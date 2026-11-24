"""A safetensors file read up to its header, without importing MLX: each tensor's name, stored dtype and shape, checked
against the file, and the one table of the stored dtypes ippon reads and their widths.

Locating a model (`ippon check`, `pull`, preflight), estimating its size (admission.py), reading what its packed
weights are (quantization.inspect) and mapping a vocabulary table's rows (host_rows.py) all read headers here.
"""

from __future__ import annotations

import itertools
import json
import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import invalid_file, weight_files

# bytes per value of each dtype a header may name: the floating ones, packed U32 weights, U8 encoded scales, integers
WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1}
COMPUTED = ("F16", "BF16", "F32")  # the floating dtypes MLX computes in on the GPU
FLOATS = (*COMPUTED, "F64")  # every floating dtype a weight may be stored in


@dataclass(frozen=True)
class Tensor:
    """A stored tensor as its header describes it (or a GGUF tensor as ippon will hold it: gguf/source.py)."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


def tensor_header(path: Path) -> dict[str, Any]:
    """The header of the safetensors file `path` (each tensor's dtype, shape and data offsets, and its optional
    `__metadata__`), read only up to its bounded size and checked against the file: every dtype in WIDTH, every
    tensor's bytes inside the file, agreeing with its shape, and overlapping no other's."""
    return _read(path)[1]


def data_start(path: Path, name: str) -> tuple[Tensor, int] | None:
    """The tensor `name` of the safetensors file `path`, from its checked header, and the byte offset in the file at
    which its data starts; None when the file holds no tensor of that name."""
    size, header = _read(path)
    entry = header.get(name)
    if name == "__metadata__" or entry is None:
        return None
    return Tensor(name, entry["dtype"], tuple(entry["shape"])), 8 + size + int(entry["data_offsets"][0])


def tensors(directory: Path, files: Iterable[str] | None = None) -> list[Tensor]:
    """Every tensor's name, dtype and shape from the safetensors headers in `directory` (`files`, else the shards an
    index names and every safetensors file there), the files read only up to their header."""
    out = []
    names = (
        sorted(set(weight_files(directory)) | {p.name for p in directory.glob("*.safetensors")})
        if files is None
        else files
    )
    for name in names:
        header = tensor_header(directory / name)
        out += [Tensor(k, v["dtype"], tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"]
    return out


def _read(path: Path) -> tuple[int, dict[str, Any]]:
    """The header's size in bytes and the header itself, checked (`tensor_header`)."""
    try:
        with path.open("rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            if size > 100 * 1024 * 1024 or size > path.stat().st_size - 8:
                raise ValueError("invalid safetensors header size")
            result: dict[str, Any] = json.loads(f.read(size))
        if not isinstance(result, dict):
            raise ValueError("expected a tensor metadata object")
        if result.get("__metadata__") is None:
            result.pop("__metadata__", None)  # MLX writes null when no optional metadata was supplied
        if not isinstance(result.get("__metadata__", {}), dict):
            raise ValueError("expected a tensor metadata object")
        if any(not isinstance(v, str) for v in result.get("__metadata__", {}).values()):
            raise ValueError("tensor metadata values must be strings")
        ranges = []
        for name, value in result.items():
            if name == "__metadata__":
                continue
            shape, dtype = value["shape"], value["dtype"]
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"{name}: invalid tensor shape")
            if not isinstance(dtype, str) or dtype not in WIDTH:
                raise ValueError(f"{name}: unsupported storage dtype {dtype!r}")
            start, end = value["data_offsets"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start <= end <= path.stat().st_size - size - 8
            ):
                raise ValueError(f"{name}: tensor extends past the file")
            if end - start != Tensor(name, dtype, tuple(shape)).elements * WIDTH[dtype]:
                raise ValueError(f"{name}: tensor byte count disagrees with its shape and dtype")
            ranges.append((start, end))
        ranges.sort()
        if any(a[1] > b[0] for a, b in itertools.pairwise(ranges)):
            raise ValueError("overlapping tensor data")
        return size, result
    except (OSError, ValueError, KeyError, TypeError, RecursionError, struct.error) as error:
        raise invalid_file(path, f"invalid tensor header ({error})") from None
