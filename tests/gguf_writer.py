"""A minimal GGUF (version 3) writer for tests, written from the format's specification: metadata, and tensors in F32,
Q8_0 (per block of 32: an fp16 scale d and 32 int8 values, x = d * q), Q4_0 (d and 32 four-bit q, x = d * (q - 8)),
or a K-quant type: seeded random blocks whose fp16 scales are small and positive (every byte pattern is a valid block),
its values left to a reader to unpack."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

ALIGN = 32
TYPE_IDS = {"F32": 0, "Q4_0": 2, "Q8_0": 8, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14}
BLOCKS = {"F32": (1, 4), "Q4_0": (32, 18), "Q8_0": (32, 34), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176)}
BLOCKS["Q6_K"] = (256, 210)
SCALES = {"Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,)}  # where each block's fp16 scales sit


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v: Any) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return struct.pack("<Ii", 5, v)
    if isinstance(v, float):
        return struct.pack("<If", 6, v)
    if isinstance(v, str):
        return struct.pack("<I", 8) + _string(v)
    if isinstance(v, list):
        if v and all(isinstance(x, bool) for x in v):
            return struct.pack("<IIQ", 9, 7, len(v)) + b"".join(struct.pack("<?", x) for x in v)
        if all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return struct.pack("<IIQ", 9, 5, len(v)) + b"".join(struct.pack("<i", x) for x in v)
        return struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_string(str(x)) for x in v)
    raise TypeError(v)


def q8_0(x: np.ndarray) -> tuple[bytes, np.ndarray]:
    """x [rows, cols] (cols a multiple of 32) in Q8_0, and the values it stores."""
    blocks = x.astype(np.float32).reshape(-1, 32)
    d = (np.abs(blocks).max(-1) / 127).astype(np.float16)
    safe = np.where(d == 0, 1, d.astype(np.float32))
    q = np.clip(np.round(blocks / safe[:, None]), -127, 127).astype(np.int8)
    data = b"".join(struct.pack("<e", float(s)) + qs.tobytes() for s, qs in zip(d, q, strict=True))
    return data, (q.astype(np.float32) * d.astype(np.float32)[:, None]).reshape(x.shape)


def q4_0(x: np.ndarray) -> tuple[bytes, np.ndarray]:
    """x [rows, cols] in Q4_0 as ggml's quantize_row_q4_0 makes it (d = the block's signed largest value / -8,
    q = min(15, floor(x / d + 8.5))), and the values it stores, d * (q - 8)."""
    blocks = x.astype(np.float32).reshape(-1, 32)
    largest = blocks[np.arange(len(blocks)), np.abs(blocks).argmax(-1)]
    d = (largest / -8).astype(np.float16)
    inverse = np.where(d == 0, 0, 1 / np.where(d == 0, 1, d.astype(np.float32)))
    q = np.clip(np.floor(blocks * inverse[:, None] + 8.5), 0, 15).astype(np.uint8)
    packed = q[:, :16] | (q[:, 16:] << 4)  # values 0-15 in the low nibbles, 16-31 in the high
    data = b"".join(struct.pack("<e", float(s)) + qs.tobytes() for s, qs in zip(d, packed, strict=True))
    return data, ((q.astype(np.float32) - 8) * d.astype(np.float32)[:, None]).reshape(x.shape)


def k_quant(count: int, kind: str) -> bytes:
    """`count` values' blocks of a K-quant type: random bytes, each fp16 scale (d, and Q4_K's and Q5_K's dmin)
    between 0.001 and 0.01."""
    block, size = BLOCKS[kind]
    rng = np.random.default_rng(count + size)
    blocks = rng.integers(0, 256, (count // block, size), dtype=np.uint8)
    for at in SCALES[kind]:
        scale = rng.uniform(0.001, 0.01, len(blocks)).astype(np.float16)
        blocks[:, at : at + 2] = scale.view(np.uint8).reshape(-1, 2)
    return blocks.tobytes()


def _scale_min(j: int, q: bytes) -> tuple[int, int]:
    """ggml's get_scale_min_k4."""
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequantize(raw: bytes, kind: str) -> np.ndarray:
    """A K-quant tensor's values, transcribed loop for loop from ggml's dequantize_row_q4_K, _q5_K and _q6_K (fp32)."""
    _, size = BLOCKS[kind]
    out: list[float] = []
    for b in range(len(raw) // size):
        x = raw[b * size : (b + 1) * size]
        if kind == "Q6_K":
            ql, qh, sc = x[0:128], x[128:192], np.frombuffer(x[192:208], np.int8)
            d = float(np.frombuffer(x[208:210], np.float16)[0])
            for n in range(2):
                y = [0.0] * 128
                for k in range(32):
                    s = k // 16
                    q1 = ((ql[64 * n + k] & 0xF) | (((qh[32 * n + k] >> 0) & 3) << 4)) - 32
                    q2 = ((ql[64 * n + k + 32] & 0xF) | (((qh[32 * n + k] >> 2) & 3) << 4)) - 32
                    q3 = ((ql[64 * n + k] >> 4) | (((qh[32 * n + k] >> 4) & 3) << 4)) - 32
                    q4 = ((ql[64 * n + k + 32] >> 4) | (((qh[32 * n + k] >> 6) & 3) << 4)) - 32
                    for at, step, q in ((0, 0, q1), (32, 2, q2), (64, 4, q3), (96, 6, q4)):
                        y[k + at] = float(np.float32(np.float32(d) * np.float32(sc[8 * n + s + step])) * np.float32(q))
                out += y
            continue
        d, dmin = (float(v) for v in np.frombuffer(x[0:4], np.float16))
        scales, qh, qs = x[4:16], x[16:48], x[-128:]
        u1, u2 = 1, 2
        for c in range(4):
            sc1, m1 = _scale_min(2 * c, scales)
            sc2, m2 = _scale_min(2 * c + 1, scales)
            ql = qs[32 * c : 32 * c + 32]
            five = kind == "Q5_K"
            out += [d * sc1 * ((q & 0xF) + (16 if five and qh[i] & u1 else 0)) - dmin * m1 for i, q in enumerate(ql)]
            out += [d * sc2 * ((q >> 4) + (16 if five and qh[i] & u2 else 0)) - dmin * m2 for i, q in enumerate(ql)]
            u1, u2 = u1 << 2, u2 << 2
    return np.array(out, np.float32)


def write(path: Path, metadata: dict[str, Any], tensors: dict[str, tuple[np.ndarray, str]]) -> dict[str, np.ndarray]:
    """Write the GGUF file; each tensor's values as stored (Q8_0's rounding applied)."""
    infos, data, stored, offset = [], [], {}, 0
    for name, (x, kind) in tensors.items():
        if kind == "F32":
            raw, kept = x.astype(np.float32).tobytes(), x.astype(np.float32)
        elif kind == "Q8_0":
            raw, kept = q8_0(x)
        elif kind == "Q4_0":
            raw, kept = q4_0(x)
        else:
            raw, kept = k_quant(x.size, kind), x
        stored[name] = kept
        dims = tuple(reversed(x.shape))
        infos.append(
            _string(name)
            + struct.pack(f"<I{len(dims)}Q", len(dims), *dims)
            + struct.pack("<IQ", TYPE_IDS[kind], offset)
        )
        raw += bytes(-len(raw) % ALIGN)
        data.append(raw)
        offset += len(raw)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    head += b"".join(_string(k) + _value(v) for k, v in metadata.items())
    head += b"".join(infos)
    head += bytes(-len(head) % ALIGN)
    path.write_bytes(head + b"".join(data))
    return stored
