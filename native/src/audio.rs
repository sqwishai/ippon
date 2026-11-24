//! An audio file's samples, decoded with symphonia: MP3, AAC (M4A, MP4), ALAC, FLAC, Ogg Vorbis, AIFF, CAF and WAV,
//! the formats a recording on a Mac or a phone comes in. The samples come out as the file holds them, interleaved
//! f32 at its own rate and channel count; mixing down and resampling to the model's rate stay with the model's front
//! end (src/ippon/media/lfm3_audio.py), so every format meets the same arithmetic.

use std::io::Cursor;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// the encoder's priming and padding trimmed, as players and FFmpeg trim them (an AAC file starts with ~2,112 silent
// samples, an MP3 with ~1,105)
const GAPLESS: FormatOptions =
    FormatOptions { enable_gapless: true, prebuild_seek_index: false, seek_index_fill_rate: 20 };

const MAX_RATE: u32 = 384_000;
const MAX_CHANNELS: usize = 32;
// Cap decoded mono frames at 135M (~540 MiB as f32): covers the full documented ~43-minute limit at 48 kHz
// regardless of channel count (stereo, 5.1, etc., mixed to mono per packet) while still refusing high-rate bombs
// (such as a 384 kHz x 10 min FLAC of silence = 230M frames) before allocating.
const MAX_DECODED_FRAMES: usize = 135_000_000;

/// The clip's samples, interleaved (or 1-channel mono when `decode(..., true)`), with its sample rate and channel count.
pub struct Decoded {
    pub samples: Vec<f32>,
    pub rate: u32,
    pub channels: usize,
}

fn check_rate(rate: u32) -> Result<(), String> {
    if rate == 0 || rate > MAX_RATE {
        return Err(format!("a sample rate of {rate} Hz; a clip is 1 Hz to {MAX_RATE} Hz"));
    }
    Ok(())
}

fn check_channels(channels: usize) -> Result<(), String> {
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(format!("a channel count of {channels}; a clip is 1 to {MAX_CHANNELS} channels"));
    }
    Ok(())
}

fn check_budget(frames: u64, rate: u32, channels: usize, max_seconds: f64, mono: bool) -> Result<(), String> {
    check_channels(channels)?;
    if rate > 0 && frames as f64 > max_seconds * rate as f64 {
        return Err(format!("the recording is longer than the {} minutes a clip may be", (max_seconds / 60.0) as u64));
    }
    if frames > MAX_DECODED_FRAMES as u64 {
        return Err(format!(
            "the recording decodes to more than {}M frames ({rate} Hz x {channels} ch)",
            MAX_DECODED_FRAMES / 1_000_000
        ));
    }
    if !mono && frames.saturating_mul(channels as u64) > (MAX_DECODED_FRAMES as u64).saturating_mul(2) {
        return Err(format!(
            "the recording decodes to more than {}M samples ({rate} Hz x {channels} ch)",
            (MAX_DECODED_FRAMES * 2) / 1_000_000
        ));
    }
    Ok(())
}

/// Refuses a recording past `max_seconds` or `MAX_DECODED_FRAMES`, or outside `1..=MAX_RATE` Hz / `1..=MAX_CHANNELS`
/// channels, from its header before decoding and again as packets decode. With `mono`, each packet is mixed down to
/// one f32 channel on the fly so a 43-minute stereo or surround recording never holds multi-channel samples in memory.
pub fn decode(data: Vec<u8>, max_seconds: f64, mono: bool) -> Result<Decoded, String> {
    let edit = mp4_edit(&data);
    let source = MediaSourceStream::new(Box::new(Cursor::new(data)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), source, &GAPLESS, &MetadataOptions::default())
        .map_err(|e| format!("not an audio file ippon decodes ({e})"))?;
    let mut format = probed.format;
    let (mut id, mut decoder, first_rate) = audio_track(format.as_ref(), max_seconds, mono)?;
    let (mut samples, mut rate, mut channels) = (Vec::new(), first_rate, 0usize);
    let mut buffer: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            // a chained recording (Ogg streams joined end to end) starts its next stream as a new track: read on
            // from it, as FFmpeg does, so the model hears the whole recording and not its first part only
            Err(Error::ResetRequired) => {
                (id, decoder, _) = audio_track(format.as_ref(), max_seconds, mono)?;
                continue;
            }
            Err(e) => return Err(format!("the file could not be read ({e})")),
        };
        if packet.track_id() != id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(Error::DecodeError(_)) => continue, // a damaged packet is skipped, as decoders do
            Err(e) => return Err(format!("the audio could not be decoded ({e})")),
        };
        let spec = *decoded.spec();
        check_rate(spec.rate)?;
        let count = spec.channels.count();
        check_channels(count)?;
        if channels != 0 && (spec.rate != rate || count != channels) {
            return Err(format!(
                "the recording changes from {rate} Hz, {channels} channels to {} Hz, {count} channels part way; \
                 convert it to one rate and channel count first",
                spec.rate,
            ));
        }
        rate = spec.rate;
        channels = count;
        let frames = decoded.frames();
        if frames == 0 {
            continue;
        }
        let out_channels = if mono { 1 } else { channels };
        let total_frames = (samples.len() / out_channels).saturating_add(frames);
        check_budget(total_frames as u64, rate, channels, max_seconds, mono)?;
        let needed = decoded.capacity() as u64;
        let fits = buffer.as_ref().is_some_and(|b| b.capacity() as u64 >= needed * channels as u64);
        if !fits {
            buffer = Some(SampleBuffer::new(needed, spec));
        }
        let out = buffer.as_mut().expect("made above");
        out.copy_interleaved_ref(decoded);
        if mono && channels > 1 {
            let inv = 1.0_f32 / channels as f32;
            samples.reserve(frames);
            for frame in out.samples().chunks_exact(channels) {
                samples.push(frame.iter().copied().sum::<f32>() * inv);
            }
        } else {
            samples.extend_from_slice(out.samples());
        }
    }
    if samples.is_empty() || channels == 0 || rate == 0 {
        return Err("the file holds no audio samples".into());
    }
    let out_channels = if mono { 1 } else { channels };
    if let Some(edit) = edit {
        let frames = samples.len() / out_channels;
        // saturating, clamped to the samples before narrowing: a file's edit list can say anything
        let start = ((edit.start as u128).saturating_mul(rate as u128) / (edit.media_scale as u128).max(1))
            .min(frames as u128) as usize;
        let end = match edit.length {
            Some(length) => (start as u128)
                .saturating_add((length as u128).saturating_mul(rate as u128) / (edit.movie_scale as u128).max(1))
                .min(frames as u128) as usize,
            None => frames,
        };
        samples = samples[start * out_channels..end * out_channels].to_vec();
    }
    if samples.is_empty() {
        return Err("the file holds no audio samples".into());
    }
    Ok(Decoded { samples, rate, channels: out_channels })
}

/// The reader's first audio track: its id, a decoder for it and its sample rate (0 when the header leaves it out).
/// Checks declared sample rate, channel count and frame budget before allocating a decoder or sample buffer.
fn audio_track(
    format: &dyn FormatReader,
    max_seconds: f64,
    mono: bool,
) -> Result<(u32, Box<dyn Decoder>, u32), String> {
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("the file has no audio track")?;
    let params = &track.codec_params;
    if let Some(rate) = params.sample_rate {
        check_rate(rate)?;
    }
    let channels = params.channels.map(|ch| ch.count()).unwrap_or(1);
    check_channels(channels)?;
    if let (Some(n_frames), Some(rate)) = (params.n_frames, params.sample_rate) {
        check_budget(n_frames, rate, channels, max_seconds, mono)?;
    }
    let decoder = symphonia::default::get_codecs()
        .make(params, &DecoderOptions::default())
        .map_err(|e| format!("its audio codec is one ippon doesn't decode ({e})"))?;
    Ok((track.id, decoder, params.sample_rate.unwrap_or(0)))
}

fn first_edit(elst: &[u8], media_scale: u32, movie_scale: u32) -> Option<Edit> {
    let (version, count) = (*elst.first()?, u32::from_be_bytes(elst.get(4..8)?.try_into().ok()?));
    let width = if version == 1 { 8 } else { 4 };
    for i in 0..count as usize {
        let at = 8 + i * (2 * width + 4);
        let read = |at: usize| -> Option<i64> {
            let b = elst.get(at..at + width)?;
            Some(if width == 8 {
                i64::from_be_bytes(b.try_into().ok()?)
            } else {
                i32::from_be_bytes(b.try_into().ok()?) as i64
            })
        };
        let (duration, media_time) = (read(at)?, read(at + width)?);
        if media_time >= 0 {
            let length = (duration > 0 && count == 1).then_some(duration as u64);
            return Some(Edit { start: media_time as u64, media_scale, length, movie_scale });
        }
    }
    None
}

/// The `iTunSMPB` tag Apple's AAC encoders write (" 00000000 <priming> <padding> <samples> ...", hex): the priming
/// to skip and the samples to keep, in the track's own samples.
fn itunes_gapless(moov: &[u8], media_scale: u32) -> Option<Edit> {
    let tag = moov.windows(8).position(|w| w == b"iTunSMPB")?;
    let data = moov[tag..].windows(4).position(|w| w == b"data")? + tag;
    let size = u32::from_be_bytes(moov.get(data - 4..data)?.try_into().ok()?) as usize;
    let text = std::str::from_utf8(moov.get(data + 12..data - 4 + size)?).ok()?;
    let fields: Vec<u64> = text.split_whitespace().filter_map(|f| u64::from_str_radix(f, 16).ok()).collect();
    let (start, samples) = (*fields.get(1)?, *fields.get(3)?);
    Some(Edit { start, media_scale, length: (samples > 0).then_some(samples), movie_scale: media_scale })
}

/// An MP4 audio track's first edit (moov > trak > edts > elst), which says where its samples start past the
/// encoder's priming and, by its duration, where they end: FFmpeg plays the samples it selects, and so does this.
struct Edit {
    start: u64, // in the track's timescale (mdhd)
    media_scale: u32,
    length: Option<u64>, // in the movie's timescale (mvhd); None for the rest of the track
    movie_scale: u32,
}

fn boxes(data: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        if at + 8 > data.len() {
            return None;
        }
        let size = u32::from_be_bytes(data[at..at + 4].try_into().ok()?) as u64;
        let kind = &data[at + 4..at + 8];
        let (header, size) = match size {
            0 => (8usize, (data.len() - at) as u64),
            1 => (16, u64::from_be_bytes(data.get(at + 8..at + 16)?.try_into().ok()?)),
            n => (8, n),
        };
        let end = at.checked_add(size as usize).filter(|&e| e <= data.len() && size >= header as u64)?;
        let body = &data[at + header..end];
        at = end;
        Some((kind, body))
    })
}

fn child<'a>(data: &'a [u8], kind: &[u8]) -> Option<&'a [u8]> {
    boxes(data).find(|(k, _)| *k == kind).map(|(_, body)| body)
}

fn timescale(header: &[u8]) -> Option<u32> {
    let at = if *header.first()? == 1 { 20 } else { 12 }; // after version, flags and the two dates
    Some(u32::from_be_bytes(header.get(at..at + 4)?.try_into().ok()?)).filter(|&t| t > 0)
}

fn mp4_edit(data: &[u8]) -> Option<Edit> {
    if data.get(4..8)? != b"ftyp" {
        return None;
    }
    let moov = child(data, b"moov")?;
    let movie_scale = timescale(child(moov, b"mvhd")?)?;
    for (kind, trak) in boxes(moov) {
        let mdia = if kind == b"trak" { child(trak, b"mdia") } else { None };
        let Some(mdia) = mdia else { continue };
        if child(mdia, b"hdlr").and_then(|h| h.get(8..12)) != Some(b"soun") {
            continue;
        }
        let media_scale = timescale(child(mdia, b"mdhd")?)?;
        if let Some(elst) = child(trak, b"edts").and_then(|e| child(e, b"elst")) {
            return first_edit(elst, media_scale, movie_scale);
        }
        return itunes_gapless(moov, media_scale); // Apple's encoders say it in a tag instead
    }
    None
}
