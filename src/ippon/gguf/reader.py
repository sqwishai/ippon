"""A GGUF file's tensors, read straight from the file into MLX (its header: header.py).

A Q8_0, Q4_0, Q4_K or Q5_K matrix becomes MLX's affine form in groups of 32, exactly: Q8_0's x = d * q (q an int8)
is stored as q + 128 with scale d and bias -128 d, Q4_0's x = d * (q - 8) as q with scale d and bias -8 d, a K-quant's
32-value sub-block x = d * sc * q - dmin * m (6-bit sc and m per sub-block, q of 4 or 5 bits) as q with scale d * sc
and bias -dmin * m, the values reordered to MLX's packing. Q6_K scales every 16 values, which MLX's groups (32 or
more) can't hold, so its matrices are unpacked to fp16, the values llama.cpp's Metal kernels multiply (a K-quant
build's Q6_K matrices are its output matrix and some of its value and down projections). Builds below 8 bits load only
when asked for (drift.py).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ..errors import UnsupportedModel
from ..host_rows import HostRows
from ..quantization import biases_of, scales_of
from .header import DENSE, GROUP_SIZE, QUANTIZED, Header, Tensor


def _dense(values: np.ndarray, kind: str) -> mx.array:
    """Values of a dense type as `DENSE` reads them, as MLX's array of that type."""
    array = mx.array(values)
    return array.view(mx.bfloat16) if kind == "BF16" else array


def load(header: Header) -> dict[str, mx.array]:
    """Every tensor of the file, as MLX arrays named as `mx.load` names them: a dense tensor by its name, a Q6_K
    matrix as fp16, a Q8_0, Q4_0, Q4_K or Q5_K matrix `<base>.weight` (packed uint32), `<base>.scales` and
    `<base>.biases` (fp16) for its name `<base>.weight`. Read through a memory map, one tensor at a time."""
    header.refuse_unread()
    data = np.memmap(header.path, dtype=np.uint8, mode="r")
    out: dict[str, mx.array] = {}
    for t in header.tensors.values():
        if t.offset + t.nbytes > data.size:
            raise UnsupportedModel(f"{header.path.name}: tensor {t.name!r} runs past the end of the file")
        raw = data[t.offset : t.offset + t.nbytes]
        if t.type in DENSE:
            out[t.name] = _dense(raw.view(DENSE[t.type]).reshape(t.shape), t.type)
        elif t.type == "Q6_K":
            out[t.name] = mx.array(_q6_k(raw, t))
        else:
            weight = t.name.removesuffix(".weight") + ".weight"
            out[weight], out[scales_of(weight)], out[biases_of(weight)] = _affine(raw, t)
    return out


def _affine(raw: np.ndarray, t: Tensor) -> tuple[mx.array, mx.array, mx.array]:
    """A Q8_0, Q4_0, Q4_K or Q5_K matrix in MLX's affine groups of 32, with the same values."""
    rows, cols = t.size // t.shape[-1], t.shape[-1]
    if t.type in ("Q4_K", "Q5_K"):
        return _k_quant(raw, t)
    if t.type == "Q4_0":
        weight, d, biases = _q4_0(mx.array(raw), rows, cols)
    else:
        weight, d, biases = affine_rows(mx.array(raw).reshape(rows, -1), cols)
    mx.eval(weight, d, biases)  # one matrix at a time: its bytes are released before the next is read
    return weight, d, biases


def _q4_0(raw: mx.array, rows: int, cols: int) -> tuple[mx.array, mx.array, mx.array]:
    """Q4_0 blocks (a float16 scale and 16 bytes, values 0-15 in their low nibbles and 16-31 in the high) in MLX's
    affine groups of 32: x = d * (q - 8) as q with scale d and bias -8 d."""
    blocks = raw.reshape(rows * cols // GROUP_SIZE, 18)
    d = mx.contiguous(blocks[:, :2]).view(mx.float16).reshape(rows, cols // GROUP_SIZE)
    packed = blocks[:, 2:]
    values = mx.concatenate([mx.bitwise_and(packed, 0x0F), mx.right_shift(packed, 4)], axis=1)
    q = mx.bitwise_or(values[:, 0::2], mx.left_shift(values[:, 1::2], 4))  # MLX: value i in bits 4 (i mod 8)
    weight = mx.contiguous(q.reshape(rows, -1)).view(mx.uint32)
    biases = (d.astype(mx.float32) * -8.0).astype(mx.float16)
    return weight, d, biases


def affine_rows(blocks: mx.array, cols: int) -> tuple[mx.array, mx.array, mx.array]:
    """Q8_0 rows (each row's blocks of a float16 scale and 32 int8 values, [rows, cols / 32 * 34] bytes) in MLX's
    affine groups of 32: the packed values, their scales and biases."""
    rows = blocks.shape[0]
    grouped = blocks.reshape(rows * cols // GROUP_SIZE, 34)
    d = mx.contiguous(grouped[:, :2]).view(mx.float16).reshape(rows, cols // GROUP_SIZE)
    q = mx.bitwise_xor(grouped[:, 2:], mx.array(0x80, mx.uint8))
    weight = mx.contiguous(q.reshape(rows, -1)).view(mx.uint32)
    biases = (d.astype(mx.float32) * -128.0).astype(mx.float16)
    return weight, d, biases


def host_rows(header: Header, t: Tensor, dtype: mx.Dtype) -> HostRows | None:
    """A 2-D tensor of the file as rows read from it on the CPU in `dtype` (`HostRows`), a Q8_0 or Q4_0 one
    converted row by row as `load` converts it whole; None for another type or shape."""
    if len(t.shape) != 2 or t.type not in ("Q8_0", "Q4_0", *DENSE):
        return None
    rows, cols = t.shape
    table = np.memmap(header.path, dtype=np.uint8, mode="r", offset=t.offset, shape=(rows, t.nbytes // rows))
    if t.type in QUANTIZED:
        kind = t.type

        def packed(ids: np.ndarray) -> tuple[mx.array, ...]:
            picked = mx.array(table[ids.reshape(-1)])
            parts = affine_rows(picked, cols) if kind == "Q8_0" else _q4_0(picked, ids.size, cols)
            return tuple(p.reshape(*ids.shape, p.shape[-1]) for p in parts)

        return HostRows(packed, dtype, (QUANTIZED[kind], GROUP_SIZE, "affine"))
    typed = table.view(DENSE[t.type])

    def dense(ids: np.ndarray) -> tuple[mx.array, ...]:
        return (_dense(typed[ids], t.type),)

    return HostRows(dense, dtype)


def _k_scales(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A K-quant block's 12 bytes of 6-bit scales and mins as [blocks, 8] each, one per 32-value sub-block
    (ggml's get_scale_min_k4)."""
    q = packed.astype(np.uint8)
    low, high = q[:, 0:4], q[:, 4:8]
    scale = np.concatenate([low & 63, (q[:, 8:12] & 0x0F) | ((low >> 6) << 4)], axis=1)
    minimum = np.concatenate([high & 63, (q[:, 8:12] >> 4) | ((high >> 6) << 4)], axis=1)
    return scale, minimum


def _pack(values: np.ndarray, bits: int) -> np.ndarray:
    """Values [rows, cols] of `bits` bits packed as MLX packs them: each run of 32 bits holds the next values from its
    lowest bit up, a value that doesn't fit in one word continuing in the next (5 bits: 8 values in 5 bytes)."""
    rows, cols = values.shape
    per = 8  # values per pack: 8 x bits is a whole number of bytes
    v = values.reshape(rows, cols // per, per).astype(np.uint64)
    word = np.zeros(v.shape[:2], np.uint64)
    for i in range(per):
        word |= v[..., i] << np.uint64(bits * i)
    out = np.empty((*word.shape, bits), np.uint8)
    for b in range(bits):
        out[..., b] = (word >> np.uint64(8 * b)) & np.uint64(0xFF)
    return out.reshape(rows, cols * bits // 8)


def _k_quant(raw: np.ndarray, t: Tensor) -> tuple[mx.array, mx.array, mx.array]:
    """A Q4_K or Q5_K matrix (ggml's dequantize_row_q4_K, _q5_K): per block of 256 an fp16 d and dmin, 6-bit scales
    and mins of its 8 sub-blocks, and 4-bit values (Q5_K: and a fifth bit each), in chunks of 64 whose first 32 values
    are the low nibbles of 32 bytes and the next 32 their high nibbles."""
    rows, cols = t.size // t.shape[-1], t.shape[-1]
    size = 144 if t.type == "Q4_K" else 176
    blocks = np.ascontiguousarray(raw).reshape(-1, size)
    d = blocks[:, 0:2].copy().view(np.float16).reshape(-1).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).reshape(-1).astype(np.float32)
    scale, minimum = _k_scales(blocks[:, 4:16])
    quants = blocks[:, -128:].reshape(-1, 4, 32)
    values = np.stack([quants & 0x0F, quants >> 4], axis=2)  # [blocks, chunk, low or high, 32]
    if t.type == "Q5_K":  # bit 2c of each of the 32 qh bytes is chunk c's low half's fifth bit, bit 2c + 1 its high's
        high = blocks[:, 16:48]
        shifts = np.arange(8, dtype=np.uint8).reshape(4, 2)
        values = values | (((high[:, None, None, :] >> shifts[None, :, :, None]) & 1) << 4)
    bits = QUANTIZED[t.type]
    weight = _pack(values.reshape(rows, cols), bits).view(np.uint32)
    scales = (d[:, None] * scale).reshape(rows, cols // GROUP_SIZE).astype(np.float16)
    biases = (-(dmin[:, None] * minimum)).reshape(rows, cols // GROUP_SIZE).astype(np.float16)
    return mx.array(weight), mx.array(scales), mx.array(biases)


def _q6_k(raw: np.ndarray, t: Tensor) -> np.ndarray:
    """A Q6_K matrix as fp16 (ggml's dequantize_row_q6_K): per block of 256, 4-bit low parts (128 bytes), 2-bit high
    parts (64 bytes), int8 scales of its 16 sub-blocks of 16 and an fp16 d, x = d * sc * (q - 32)."""
    blocks = np.ascontiguousarray(raw).reshape(-1, 210)
    low, high = blocks[:, :128].reshape(-1, 2, 64), blocks[:, 128:192].reshape(-1, 2, 32)
    sc = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = blocks[:, 208:210].copy().view(np.float16).reshape(-1).astype(np.float32)
    # each half of 128 values is four runs of 32 from low[0:32] & 15, low[32:64] & 15, low[0:32] >> 4,
    # low[32:64] >> 4, each with two bits of the half's 32 high bytes (bits 0-1, 2-3, 4-5, 6-7)
    parts = np.stack(
        [low[:, :, :32] & 0x0F, low[:, :, 32:] & 0x0F, low[:, :, :32] >> 4, low[:, :, 32:] >> 4], axis=2
    )  # [blocks, half, run, 32]
    top = (high[:, :, None, :] >> (2 * np.arange(4, dtype=np.uint8))[None, None, :, None]) & 3
    q = (parts | (top << 4)).astype(np.float32) - 32
    x = (d[:, None, None] * sc[:, :, None]) * q.reshape(-1, 16, 16)  # in ggml's order, (d * sc) * q, in fp32
    return np.asarray(x.reshape(t.shape), dtype=np.float16)
