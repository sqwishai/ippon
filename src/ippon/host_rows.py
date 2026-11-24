"""A vocabulary table left in its file and read row by row on the CPU (`HostRows`).

An embedding or output matrix is read a few hundred rows a prompt, so it needs no GPU memory: mapping its file and
copying the rows asked for costs microseconds, while holding Kettei Core's 250,624-row embedding takes 1.3 GB at bf16.
The pages the OS keeps of a mapped file can also be dropped under memory pressure and read again, which wired GPU
memory cannot. A packed table (an MLX 8-bit or MXFP8 build) is read as its parts' rows, which `quant.rows` unpacks
as it unpacks the whole table, so rows come out exactly as the loaded table's would.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .safetensors import COMPUTED, WIDTH, data_start

# a stored dtype (safetensors.WIDTH's names) -> the MLX dtype a load reads it as
MLX: dict[str, mx.Dtype] = {
    "F64": mx.float64,
    "F32": mx.float32,
    "F16": mx.float16,
    "BF16": mx.bfloat16,
    "I64": mx.int64,
    "I32": mx.int32,
    "U32": mx.uint32,
    "I16": mx.int16,
    "U16": mx.uint16,
    "I8": mx.int8,
    "U8": mx.uint8,
}
ROWS = (*COMPUTED, "U32", "U8")  # what a table's part is mapped from: a floating table, packed weights, MXFP8's scales
Packing = tuple[int, int, str]  # a packed table's bits, group size and mode, as QuantizedMatrix holds them


@dataclass(frozen=True)
class HostRows:
    pick: Callable[[np.ndarray], tuple[mx.array, ...]]  # each stored part's rows: one, or packed w, scales, biases
    dtype: mx.Dtype  # what rows come out in
    packing: Packing | None = None  # None: one full-precision table

    @classmethod
    def from_safetensors(
        cls, parts: Sequence[tuple[Path, str]], dtype: mx.Dtype, packing: Packing | None = None
    ) -> HostRows | None:
        """The table whose parts are the 2-D tensors `parts` (a file, a tensor's name in it), in `dtype`; None when
        one is stored in a dtype not in ROWS, or has another shape."""
        mapped = [_mapped(path, name) for path, name in parts]
        tables = [m for m in mapped if m is not None]
        if len(tables) < len(mapped):
            return None

        def pick(ids: np.ndarray) -> tuple[mx.array, ...]:
            return tuple(_rows(table, stored, ids) for table, stored in tables)

        return cls(pick, dtype, packing)


def _mapped(path: Path, name: str) -> tuple[np.ndarray, mx.Dtype] | None:
    """The 2-D tensor `name` of the file mapped as unsigned integers of its width (numpy has no bfloat16), with the
    MLX dtype its rows are viewed as; None when it is stored in a dtype not in ROWS, or has another shape."""
    found = data_start(path, name)
    if found is None:
        return None
    tensor, start = found
    if tensor.dtype not in ROWS or len(tensor.shape) != 2:
        return None
    unsigned = f"<u{WIDTH[tensor.dtype]}"
    return np.memmap(path, dtype=unsigned, mode="r", offset=start, shape=tensor.shape), MLX[tensor.dtype]


def _rows(table: np.ndarray, stored: mx.Dtype, ids: np.ndarray) -> mx.array:
    picked = mx.array(table[ids])
    return picked if picked.dtype == stored else picked.view(stored)
