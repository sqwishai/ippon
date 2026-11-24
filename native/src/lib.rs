//! ippon's native helpers, loaded through ctypes by src/ippon/native.py: the CPU work around a model that Python does
//! slowly, each giving the numbers the Python it replaced gave. On every core: `ippon_resize_u8`, torchvision's resize;
//! `ippon_patches_f32`, Qwen's normalization and patch layout; `ippon_position_table`, Qwen's vision position table for
//! a grid (79 ms in numpy for a 12-megapixel photo's 188 x 250 patches on an M5 Pro, 2.1 ms here). And
//! `ippon_gguf_header`, a GGUF file's header (gguf.rs); `ippon_decode_audio`, an audio file's samples (audio.rs).
//!
//! The creators resize with torchvision (`resize(uint8 tensor, BILINEAR | BICUBIC, antialias=True)`, ATen's
//! `_upsample_bilinear2d_aa` and `_upsample_bicubic2d_aa`), and a pixel one level off moves a probability with it, so
//! `ippon_resize_u8` reproduces ATen's arithmetic rather than resampling well in general:
//!
//! - each output pixel reads the input pixels within the filter's support around its centre, `scale * (i + 0.5)`; the
//!   support is the filter's (1 for the triangle, 2 for the cubic with a = -0.5, Pillow's) times the downscale factor,
//!   so a downscale averages every pixel it covers; the weights are normalized to sum to 1 per output pixel;
//! - the weights are computed in f64, then turned into int16 fixed point with ATen's precision search (the most
//!   fraction bits, up to 22, that keep the largest weight under 2^15), rounded half away from zero; each output is
//!   (2^(bits-1) + sum of weight x pixel) >> bits, clamped to 0..255;
//! - the width pass runs first, rounding to u8, then the height pass; an axis whose size doesn't change is not
//!   resampled.
//!
//! The f64 steps run in the order the numpy port they replace ran them (each operation rounded on its own, no fused
//! multiply-add), and the integer sums can't overflow i32 (255 x 1.25 x 2^22 at most), so the order they add in
//! doesn't matter. The cubic kernel, the negative-weight rounding and the support clamp are from ATen
//! (aten/src/ATen/native/cpu/UpSampleKernel.cpp, BSD-3-Clause); the bilinear path follows mlx-vlm 0.7.10's numpy
//! version (mlx_vlm/models/d1_omni/vision.py, MIT, Copyright (c) 2025 Prince Canuma). tests/test_media_resize.py
//! holds every path to torchvision 0.29.1's own output, bit for bit.
//!
//! Frames resize in parallel and each pass splits its output rows across the cores: on an M5 Pro, 32 frames of
//! 1280 x 720 to 576 x 320 (bicubic) took 588 ms in numpy on one core.

use std::ffi::{c_char, CStr};

use rayon::prelude::*;

mod audio;
mod gguf;

const MAX_PRECISION: u32 = 22; // ATen's ceiling on the int16 weights' fraction bits

/// For resampling n_in pixels to n_out: each output's `taps` input indices (clamped into range; a tap past the
/// support has weight 0) and int16 fixed-point weights, row after row, and the fraction bits.
struct Weights {
    taps: usize,
    index: Vec<usize>,
    fixed: Vec<i32>,
    bits: u32,
}

/// The bilinear filter, as ATen's HelperInterpLinear::aa_filter.
fn triangle(x: f64) -> f64 {
    let a = x.abs();
    if a < 1.0 {
        1.0 - a
    } else {
        0.0
    }
}

/// The bicubic filter with a = -0.5 (Pillow's, which ATen uses with antialias), in ATen's evaluation order
/// (cubic_convolution1 and 2 in UpSample.h).
fn cubic(x: f64) -> f64 {
    let a = -0.5;
    let t = x.abs();
    if t < 1.0 {
        ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
    } else if t < 2.0 {
        ((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a
    } else {
        0.0
    }
}

fn weights(n_in: usize, n_out: usize, bicubic: bool) -> Weights {
    let (interp, kernel): (f64, fn(f64) -> f64) = if bicubic { (4.0, cubic) } else { (2.0, triangle) };
    let scale = n_in as f64 / n_out as f64;
    let support = if scale >= 1.0 { interp * 0.5 * scale } else { interp * 0.5 };
    let taps = support.ceil() as usize * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut w = vec![0.0f64; n_out * taps];
    let mut index = vec![0usize; n_out * taps];
    for i in 0..n_out {
        let center = scale * (i as f64 + 0.5);
        let lo = ((center - support + 0.5) as i64).max(0); // `as` truncates toward zero, as C's cast does
        let size = (((center + support + 0.5) as i64).min(n_in as i64) - lo).clamp(0, taps as i64) as usize;
        let row = &mut w[i * taps..(i + 1) * taps];
        for (j, value) in row.iter_mut().enumerate() {
            if j < size {
                *value = kernel(((j as i64 + lo) as f64 - center + 0.5) * invscale);
            }
            index[i * taps + j] = (lo as usize + j).min(n_in - 1);
        }
        let mut total = row[0];
        for value in &row[1..] {
            total += *value; // summed in tap order, as ATen sums
        }
        if total != 0.0 {
            for value in row.iter_mut() {
                *value /= total;
            }
        }
    }
    let peak = w.iter().copied().fold(0.0f64, f64::max);
    let mut bits = 0;
    while bits < MAX_PRECISION && ((0.5 + peak * (1u64 << (bits + 1)) as f64) as i64) < 1 << 15 {
        bits += 1;
    }
    let unit = (1u64 << bits) as f64;
    let fixed = w
        .iter()
        .map(|&value| {
            let v = value * unit;
            (if v < 0.0 { v - 0.5 } else { v + 0.5 }).trunc() as i32
        })
        .collect();
    Weights { taps, index, fixed, bits }
}

/// One frame [height, width, channels] to [height, out_width, channels].
fn width_pass(src: &[u8], width: usize, channels: usize, out_width: usize, w: &Weights, dst: &mut [u8]) {
    let start = 1i32 << (w.bits - 1);
    dst.par_chunks_mut(out_width * channels).zip(src.par_chunks(width * channels)).for_each(|(out, row)| {
        for x in 0..out_width {
            let index = &w.index[x * w.taps..(x + 1) * w.taps];
            let fixed = &w.fixed[x * w.taps..(x + 1) * w.taps];
            for c in 0..channels {
                let mut acc = start;
                for (&i, &f) in index.iter().zip(fixed) {
                    acc += row[i * channels + c] as i32 * f;
                }
                out[x * channels + c] = (acc >> w.bits).clamp(0, 255) as u8;
            }
        }
    });
}

/// One frame [height, row / channels, channels] to [out_height, row / channels, channels]; `row` is a row's length.
fn height_pass(src: &[u8], row: usize, w: &Weights, dst: &mut [u8]) {
    let start = 1i32 << (w.bits - 1);
    dst.par_chunks_mut(row).enumerate().for_each(|(y, out)| {
        let mut acc = vec![start; row];
        let index = &w.index[y * w.taps..(y + 1) * w.taps];
        for (&i, &f) in index.iter().zip(&w.fixed[y * w.taps..(y + 1) * w.taps]) {
            if f == 0 {
                continue;
            }
            for (a, &p) in acc.iter_mut().zip(&src[i * row..(i + 1) * row]) {
                *a += p as i32 * f;
            }
        }
        for (o, a) in out.iter_mut().zip(&acc) {
            *o = (a >> w.bits).clamp(0, 255) as u8;
        }
    });
}

/// Resizes `frames` frames of [height, width, channels] u8 at `src` to [out_height, out_width, channels] at `dst`,
/// bilinear (`bicubic` 0) or bicubic (1), both C-contiguous and sized by the caller. 0 on success, -1 if it panicked.
///
/// # Safety
/// `src` must hold frames * height * width * channels bytes and `dst` frames * out_height * out_width * channels,
/// every size at least 1, and the two must not overlap.
#[no_mangle]
pub unsafe extern "C" fn ippon_resize_u8(
    src: *const u8,
    frames: usize,
    height: usize,
    width: usize,
    channels: usize,
    dst: *mut u8,
    out_height: usize,
    out_width: usize,
    bicubic: i32,
) -> i32 {
    let input = std::slice::from_raw_parts(src, frames * height * width * channels);
    let output = std::slice::from_raw_parts_mut(dst, frames * out_height * out_width * channels);
    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let across = (width != out_width).then(|| weights(width, out_width, bicubic != 0));
        let down = (height != out_height).then(|| weights(height, out_height, bicubic != 0));
        let mid = height * out_width * channels;
        output
            .par_chunks_mut(out_height * out_width * channels)
            .zip(input.par_chunks(height * width * channels))
            .for_each(|(out, frame)| match (&across, &down) {
                (Some(a), Some(d)) => {
                    let mut narrowed = vec![0u8; mid];
                    width_pass(frame, width, channels, out_width, a, &mut narrowed);
                    height_pass(&narrowed, out_width * channels, d, out);
                }
                (Some(a), None) => width_pass(frame, width, channels, out_width, a, out),
                (None, Some(d)) => height_pass(frame, width * channels, d, out),
                (None, None) => out.copy_from_slice(frame),
            });
    }));
    if done.is_ok() {
        0
    } else {
        -1
    }
}

/// Qwen's processor patches of `frames` frames [height, width, channels] u8 at `src` (one image, or a clip's frames,
/// as many as `temporal` pairs them), each pixel rescaled and normalized in f32 as (x - mean[c]) / std[c] (the mean
/// and std already times 1 / rescale), written to `dst` as [groups x blocks, channels x temporal x patch x patch]:
/// rows in (group, block row, block column, merge row, merge column) order, columns in (channel, frame of the
/// group, row, column) order, as transformers' Qwen2VL image and Qwen3VL video processors lay them out. One image
/// fills each group's `temporal` frames with itself. 0 on success, -1 if it panicked.
///
/// # Safety
/// `src` must hold frames * height * width * channels bytes, `mean` and `std` channels floats each, and `dst`
/// (frames, or temporal for one image) * height * width * channels floats; height and width must be multiples of
/// patch * merge and frames 1 or a multiple of temporal.
#[no_mangle]
pub unsafe extern "C" fn ippon_patches_f32(
    src: *const u8,
    frames: usize,
    height: usize,
    width: usize,
    channels: usize,
    mean: *const f32,
    std: *const f32,
    patch: usize,
    merge: usize,
    temporal: usize,
    dst: *mut f32,
) -> i32 {
    let groups = if frames == 1 { 1 } else { frames / temporal };
    let input = std::slice::from_raw_parts(src, frames * height * width * channels);
    let mean = std::slice::from_raw_parts(mean, channels);
    let std = std::slice::from_raw_parts(std, channels);
    let row = channels * temporal * patch * patch;
    let output = std::slice::from_raw_parts_mut(dst, groups * temporal * height * width * channels);
    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (blocks_h, blocks_w) = (height / patch / merge, width / patch / merge);
        output.par_chunks_mut(row).enumerate().for_each(|(r, out)| {
            let (mw, rest) = (r % merge, r / merge);
            let (mh, rest) = (rest % merge, rest / merge);
            let (bw, rest) = (rest % blocks_w, rest / blocks_w);
            let (bh, group) = (rest % blocks_h, rest / blocks_h);
            let (top, left) = ((bh * merge + mh) * patch, (bw * merge + mw) * patch);
            let mut i = 0;
            for c in 0..channels {
                for k in 0..temporal {
                    let frame = if frames == 1 { 0 } else { group * temporal + k };
                    let base = frame * height * width;
                    for y in top..top + patch {
                        for x in left..left + patch {
                            let pixel = input[((base + y * width + x) * channels) + c] as f32;
                            out[i] = (pixel - mean[c]) / std[c];
                            i += 1;
                        }
                    }
                }
            }
        });
    }));
    if done.is_ok() {
        0
    } else {
        -1
    }
}

/// Qwen's learned [side x side, width] position table resampled to an h x w patch grid, bilinear with the taps and
/// weights per axis the caller computed (src/ippon/media/qwen.py, axis_taps), written to `out` [h x w, width] in
/// merge-block order. Each cell is its four corners' rows times their float32 weights (row weight x column weight),
/// summed in f32 in the order (0, 0), (0, 1), (1, 0), (1, 1), as transformers 5.17 sums them. 0, or -1 if it panicked.
///
/// # Safety
/// `table` must hold side * side * width floats, `rows` and `row_weights` h * 2 each, `columns` and
/// `column_weights` w * 2 each, every tap below side, `out` h * w * width floats; h and w multiples of merge.
#[no_mangle]
pub unsafe extern "C" fn ippon_position_table(
    table: *const f32,
    side: usize,
    width: usize,
    rows: *const i64,
    row_weights: *const f32,
    h: usize,
    columns: *const i64,
    column_weights: *const f32,
    w: usize,
    merge: usize,
    out: *mut f32,
) -> i32 {
    let table = std::slice::from_raw_parts(table, side * side * width);
    let (rows, row_weights) = (std::slice::from_raw_parts(rows, h * 2), std::slice::from_raw_parts(row_weights, h * 2));
    let columns = std::slice::from_raw_parts(columns, w * 2);
    let column_weights = std::slice::from_raw_parts(column_weights, w * 2);
    let output = std::slice::from_raw_parts_mut(out, h * w * width);
    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let blocks_w = w / merge;
        output.par_chunks_mut(width).enumerate().for_each(|(i, cell)| {
            let (mc, rest) = (i % merge, i / merge);
            let (mr, rest) = (rest % merge, rest / merge);
            let (bc, br) = (rest % blocks_w, rest / blocks_w);
            let (r, c) = (br * merge + mr, bc * merge + mc);
            for (n, (a, b)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                let weight = row_weights[r * 2 + a] * column_weights[c * 2 + b];
                let at = (rows[r * 2 + a] as usize * side + columns[c * 2 + b] as usize) * width;
                let corner = &table[at..at + width];
                if n == 0 {
                    for (o, &t) in cell.iter_mut().zip(corner) {
                        *o = t * weight;
                    }
                } else {
                    for (o, &t) in cell.iter_mut().zip(corner) {
                        *o += t * weight;
                    }
                }
            }
        });
    }));
    if done.is_ok() {
        0
    } else {
        -1
    }
}

/// A GGUF file's header as JSON (src/gguf.rs), in a buffer the caller frees with ippon_free; its length in `len`.
///
/// # Safety
/// `path` must be a NUL-terminated UTF-8 path and `len` writable.
#[no_mangle]
pub unsafe extern "C" fn ippon_gguf_header(path: *const c_char, skip_tokenizer: i32, len: *mut usize) -> *mut u8 {
    let json = match CStr::from_ptr(path).to_str() {
        // a panic must not cross into Python, where it would end the process: it becomes the call's error
        Ok(path) => std::panic::catch_unwind(|| gguf::header(path, skip_tokenizer != 0))
            .unwrap_or_else(|_| "{\"error\":\"the header could not be read\"}".to_string()),
        Err(_) => "{\"error\":\"the path is not UTF-8\"}".to_string(),
    };
    let bytes = json.into_bytes().into_boxed_slice();
    *len = bytes.len();
    Box::into_raw(bytes) as *mut u8
}

/// An audio file's samples (audio.rs): interleaved f32, `frames` x `channels`, at `rate`, in a buffer the caller frees
/// with ippon_free_f32 (its length frames x channels). On failure null, and the reason as UTF-8 in `error`, `error_len`,
/// which the caller frees with ippon_free.
///
/// # Safety
/// `data` must hold `len` readable bytes, and the four out-pointers must be writable.
#[no_mangle]
pub unsafe extern "C" fn ippon_decode_audio(
    data: *const u8,
    len: usize,
    max_seconds: f64,
    mono: u8,
    rate: *mut u32,
    channels: *mut usize,
    frames: *mut usize,
    error: *mut *mut u8,
    error_len: *mut usize,
) -> *mut f32 {
    *error = std::ptr::null_mut();
    *error_len = 0;
    let bytes = std::slice::from_raw_parts(data, len).to_vec();
    // a panic in a decoder (symphonia's, on a malformed file) is that file's error, never the process's abort
    let decoded = std::panic::catch_unwind(move || audio::decode(bytes, max_seconds, mono != 0))
        .unwrap_or_else(|_| Err("the file could not be decoded (it is malformed)".into()));
    match decoded {
        Ok(found) => {
            *rate = found.rate;
            *channels = found.channels;
            *frames = found.samples.len() / found.channels;
            Box::into_raw(found.samples.into_boxed_slice()) as *mut f32
        }
        Err(why) => {
            let bytes = why.into_bytes().into_boxed_slice();
            *error_len = bytes.len();
            *error = Box::into_raw(bytes) as *mut u8;
            std::ptr::null_mut()
        }
    }
}

/// Frees the samples ippon_decode_audio returned.
///
/// # Safety
/// `data` and `len` must be what ippon_decode_audio returned (frames x channels), freed once.
#[no_mangle]
pub unsafe extern "C" fn ippon_free_f32(data: *mut f32, len: usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(data, len)));
}

/// Frees a buffer ippon_gguf_header returned.
///
/// # Safety
/// `data` and `len` must be what ippon_gguf_header returned, freed once.
#[no_mangle]
pub unsafe extern "C" fn ippon_free(data: *mut u8, len: usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(data, len)));
}
