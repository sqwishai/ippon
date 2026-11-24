"""A GGUF file's header, read without loading a tensor: its metadata and each tensor's type and shape.

The header is read before any tensor is allocated, to know what the file holds: which types its tensors are (F32,
F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K and Q6_K are read, other quantized types are refused, naming the Q8_0 file), how
large it is resident (admission) and what it says it is, to check against the creator's configuration (`source.bound`).
reader.py reads the tensors themselves.

The format (version 2 and 3, little-endian): magic "GGUF", version, tensor count, metadata count, the metadata as
(key, type, value), then per tensor its name, dimensions (innermost first), type and offset.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import native
from ..errors import UnsupportedModel
from ..quantization import unsupported_format

# ggml's tensor types by number, with each one's block of weights and its size in bytes
TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}
READ = ("F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K")  # what ippon loads
QUANTIZED = {"Q8_0": 8, "Q4_0": 4, "Q4_K": 4, "Q5_K": 5}  # read as MLX's affine groups of 32, at these bits
UNPACKED = ("Q6_K",)  # read as fp16 matrices
GROUP_SIZE = 32  # MLX's group size for the GGUF types it reads quantized
DENSE = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}  # how numpy reads each dense type (bf16 as its bits)


@dataclass(frozen=True)
class Tensor:
    name: str
    type: str
    shape: tuple[int, ...]  # outermost first, as MLX and numpy index it (the file stores it innermost first)
    offset: int = 0  # where its data starts in the file

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Bytes in the file."""
        _, block, size = next(v for v in TYPES.values() if v[0] == self.type)
        return self.size // block * size


@dataclass(frozen=True)
class Header:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: dict[str, Tensor]
    digest: str  # sha256 of the header's bytes (metadata and tensor table): which build a local file is

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def get(self, key: str, default: Any = None) -> Any:
        """A metadata value under the architecture's prefix ("block_count" is "<arch>.block_count")."""
        return self.metadata.get(f"{self.architecture}.{key}", default)

    def refuse_unread(self) -> None:
        """UnsupportedModel, naming the types and the way out, when a tensor is of a type ippon doesn't load
        (MLX would unpack it to fp16: the model at full size, not the build that was asked for)."""
        unread = sorted({t.type for t in self.tensors.values() if t.type not in READ})
        if unread:
            raise unsupported_format(
                f"{self.path.name}: {', '.join(unread)} tensors",
                "the repository's :Q8_0 file or full-precision checkpoint",
            )


def read_header(path: str | Path, *, skip_tokenizer: bool = True) -> Header:
    """The header of the GGUF file at `path`, parsed by ippon's native library (native/src/gguf.rs) to the values
    Python's struct module reads. Its tokenizer's vocabulary (tokenizer.ggml.tokens and the like, often the largest
    part) is skipped unless asked for: ippon reads the creator's tokenizer.json. On an M5 Pro, Clef-Flash's Q8_0 file
    takes 106 ms in Python, which a load did three times."""
    path = Path(path)
    found = native.gguf_header(path, skip_tokenizer)
    if "error" in found:
        raise UnsupportedModel(f"{path.name}: {found['error']}")
    tensors = {}
    for name, kind, dims, offset in found["tensors"]:
        if kind not in TYPES:
            raise UnsupportedModel(f"{path.name}: tensor {name!r} has an unknown ggml type {kind}")
        tensors[name] = Tensor(name, TYPES[kind][0], tuple(reversed(dims)), offset)
    end = found["end"]
    with path.open("rb") as f:
        digest = hashlib.sha256(f.read(end)).hexdigest()
    metadata = found["metadata"]
    align = metadata.get("general.alignment", 32)
    if not isinstance(align, int) or isinstance(align, bool) or align <= 0:
        raise UnsupportedModel(f"{path.name}: its alignment {align!r} is not a positive whole number")
    start = -(-end // align) * align  # the tensors' data, after the header padded to the alignment
    tensors = {n: Tensor(t.name, t.type, t.shape, start + t.offset) for n, t in tensors.items()}
    return Header(path, found["version"], metadata, tensors, digest)


def close(a: Any, b: Any) -> bool:
    """Whether a GGUF's number `a` is the creator's `b`, within float32's rounding of it."""
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
