"""ippon's native library (libippon_native.dylib, built from native/ with the package), loaded through ctypes on first
use. Its functions take C-contiguous numpy buffers the caller sized, run on every core, and let go of the GIL for the
call (ctypes does for a CDLL)."""

from __future__ import annotations

import ctypes
import json
import math
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from .errors import RuntimeUnavailable

LIBRARY = Path(__file__).resolve().parent / "libippon_native.dylib"
_SIZE, _POINTER = ctypes.c_size_t, ctypes.c_void_p


@cache
def _library() -> ctypes.CDLL:
    if not LIBRARY.is_file():
        raise RuntimeUnavailable(
            f"ippon's native library is missing ({LIBRARY}): reinstall ippon, or in a checkout, "
            "`uv sync --reinstall-package ippon`"
        )
    try:
        library = ctypes.CDLL(str(LIBRARY))
        library.ippon_resize_u8.argtypes = [_POINTER, _SIZE, _SIZE, _SIZE, _SIZE, _POINTER, _SIZE, _SIZE, ctypes.c_int]
        library.ippon_patches_f32.argtypes = [_POINTER, *[_SIZE] * 4, _POINTER, _POINTER, *[_SIZE] * 3, _POINTER]
        library.ippon_position_table.argtypes = [_POINTER, _SIZE, _SIZE, _POINTER, _POINTER, _SIZE]
        library.ippon_position_table.argtypes += [_POINTER, _POINTER, _SIZE, _SIZE, _POINTER]
        for function in (library.ippon_resize_u8, library.ippon_patches_f32, library.ippon_position_table):
            function.restype = ctypes.c_int
        library.ippon_gguf_header.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(_SIZE)]
        library.ippon_gguf_header.restype = ctypes.POINTER(ctypes.c_char)
        library.ippon_free.argtypes = [ctypes.POINTER(ctypes.c_char), _SIZE]
        library.ippon_free.restype = None
        library.ippon_decode_audio.argtypes = [
            ctypes.c_char_p,
            _SIZE,
            ctypes.c_double,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.ippon_decode_audio.argtypes += [ctypes.POINTER(_SIZE)] * 2 + [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
            ctypes.POINTER(_SIZE),
        ]
        library.ippon_decode_audio.restype = ctypes.POINTER(ctypes.c_float)
        library.ippon_free_f32.argtypes = [ctypes.POINTER(ctypes.c_float), _SIZE]
        library.ippon_free_f32.restype = None
    except (OSError, AttributeError) as error:
        raise RuntimeUnavailable(
            f"ippon's native library cannot load ({LIBRARY}: {error}): reinstall ippon, or in a checkout, "
            "`uv sync --reinstall-package ippon`"
        ) from None

    return library


def check() -> None:
    """Check that the packaged native library loads and exposes every required function."""
    _library()


def resize(frames: np.ndarray, out: np.ndarray, bicubic: bool) -> None:
    """torchvision's antialiased resize of [frames, H, W, C] uint8 into `out` [frames, h, w, C] uint8."""
    count, height, width, channels = frames.shape
    _, out_height, out_width, _ = out.shape
    if _library().ippon_resize_u8(
        frames.ctypes.data, count, height, width, channels, out.ctypes.data, out_height, out_width, int(bicubic)
    ):
        raise RuntimeError(f"ippon's native resize failed on {frames.shape} to {out.shape}")


def patches(
    frames: np.ndarray, mean: np.ndarray, std: np.ndarray, patch: int, merge: int, temporal: int, out: np.ndarray
) -> None:
    """Qwen's normalized patches of [frames, H, W, C] uint8 into `out` (native/src/lib.rs, ippon_patches_f32)."""
    count, height, width, channels = frames.shape
    if _library().ippon_patches_f32(
        frames.ctypes.data,
        count,
        height,
        width,
        channels,
        mean.ctypes.data,
        std.ctypes.data,
        patch,
        merge,
        temporal,
        out.ctypes.data,
    ):
        raise RuntimeError(f"ippon's native patches failed on {frames.shape}")


def position_table(
    table: np.ndarray,
    rows: np.ndarray,
    row_weights: np.ndarray,
    columns: np.ndarray,
    column_weights: np.ndarray,
    merge: int,
    out: np.ndarray,
) -> None:
    """Qwen's [side x side, width] position table resampled into `out` [h x w, width] in merge-block order, from each
    axis's taps [n, 2] int64 and weights [n, 2] float32 (native/src/lib.rs, ippon_position_table)."""
    side = math.isqrt(table.shape[0])
    if _library().ippon_position_table(
        table.ctypes.data,
        side,
        table.shape[1],
        rows.ctypes.data,
        row_weights.ctypes.data,
        len(rows),
        columns.ctypes.data,
        column_weights.ctypes.data,
        len(columns),
        merge,
        out.ctypes.data,
    ):
        raise RuntimeError(f"ippon's native position table failed on a {len(rows)} x {len(columns)} grid")


def gguf_header(path: Path, skip_tokenizer: bool) -> dict[str, Any]:
    """A GGUF file's header (native/src/gguf.rs): {"version", "metadata", "tensors" [[name, ggml type, dims innermost
    first, offset]], "end"}, or {"error": why}."""
    library, size = _library(), _SIZE()
    data = library.ippon_gguf_header(str(path).encode(), int(skip_tokenizer), ctypes.byref(size))
    try:
        found: dict[str, Any] = json.loads(ctypes.string_at(data, size.value))
    finally:
        library.ippon_free(data, size.value)
    return found


def decode_audio(data: bytes, max_seconds: float, *, mono: bool = False) -> tuple[np.ndarray, int]:
    """An audio file's samples (native/src/audio.rs): float32 [frames, channels] as the file holds them (or
    [frames, 1] mixed down per packet with `mono=True`), and its sample rate. ValueError saying why for a file it
    can't decode, or one past `max_seconds`, refused as soon as its samples pass it."""
    library = _library()
    rate, channels, frames = ctypes.c_uint32(), _SIZE(), _SIZE()
    error, error_len = ctypes.POINTER(ctypes.c_char)(), _SIZE()
    found = library.ippon_decode_audio(
        data,
        len(data),
        max_seconds,
        int(mono),
        ctypes.byref(rate),
        ctypes.byref(channels),
        ctypes.byref(frames),
        ctypes.byref(error),
        ctypes.byref(error_len),
    )
    if not found:
        why = ctypes.string_at(error, error_len.value).decode()
        library.ippon_free(error, error_len.value)
        raise ValueError(why)
    count = frames.value * channels.value
    try:
        samples = np.ctypeslib.as_array(found, (count,)).reshape(frames.value, channels.value).copy()
    finally:
        library.ippon_free_f32(found, count)
    return samples, rate.value
