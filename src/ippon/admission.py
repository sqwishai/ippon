"""Whether a model fits this Mac at a precision, before anything is loaded.

A model that does not fit the GPU's working set (about three quarters of memory) is not slow, it swaps: a 9.4B model
in fp16 needs about 19 GB, more than a 16 GB Mac keeps resident. So `Decider.load` estimates the model's resident
size from its safetensors headers (no weights read), and:

- at a precision that does not fit, auto included, refuses with `InsufficientMemory`, naming the precisions that
  fit, at full precision or 8-bit (`admit`). It never quantizes a model on its own: q8, q6 and q4 change the model's
  weights (and its answers, [accuracy.md](../../docs/accuracy.md)), so they run only when asked for.

The estimate: every floating-point tensor at the compute dtype's width; weight matrices quantized to b bits at b/8
bytes per weight plus one scale and bias per group (64 columns, or 32 where the width is not a multiple of 64, as
`quantization.group_for` groups them for quant.quantize); tensors already quantized at their stored size; the
vocabulary tables not at all, since a load reads them from the file. Room is kept for what a decider takes beside the
weights (`budget`), as `Decider` counts it once the model is loaded, so a model admitted here always reads.
"""

from __future__ import annotations

from collections.abc import Sequence

from .checkpoint import EMBEDDING, HEAD
from .errors import InsufficientMemory
from .quantization import OFFERED, PRECISIONS, Precision, group_for, is_global_scale, matrix_of
from .quantization import get as get_precision
from .safetensors import FLOATS, WIDTH, Tensor

# What a decider takes of the working set beside the model's weights, which `budget` keeps free and Decider._room
# counts: a pass's activations and the GPU's own buffers, the prefixes it keeps by default (a sixteenth of memory, at
# most PREFIX_BUDGET), and the least room for a reading's keys and values a model is admitted with.
READING_MARGIN = 2 << 30
PREFIX_BUDGET = 4 << 30
MIN_READING = 512 << 20
# the vocabulary tables' last names: a checkpoint's (checkpoint.EMBEDDING, HEAD, Gemma 4's per-layer embeddings), and a
# GGUF file's own, which gguf/names.py renames to them
TABLES = (matrix_of(EMBEDDING), "embed_tokens_per_layer", matrix_of(HEAD), "token_embd", "output")


def is_table(name: str) -> bool:
    """Whether the tensor `name` is a vocabulary table or a part of a packed one (its scales, biases)."""
    return matrix_of(name).rsplit(".", 1)[-1] in TABLES


def resident_bytes(found: Sequence[Tensor], width: int, bits: int | None = None) -> int:
    """The bytes `found` would hold computed at `width` bytes per value, its weight matrices at `bits` (None: as
    they are). A stored quantized matrix (an integer weight with its scales) counts as stored, and a vocabulary
    table not at all, as a load reads its rows from the file (`Weights.table`; the label-token format scoring every
    token holds its output matrix, which the reserve then covers)."""
    largest = max((t.elements for t in found if t.dtype in FLOATS and len(t.shape) == 2), default=0)
    total = 0
    for t in found:
        if is_table(t.name):
            continue
        group = group_for(t.shape[-1]) if bits is not None and len(t.shape) >= 2 and t.elements != largest else None
        if is_global_scale(t.name):
            total += t.elements * 4
        elif t.dtype not in FLOATS:
            total += t.elements * WIDTH.get(t.dtype, 4)
        elif bits is not None and group is not None:  # quantized as quant.quantize does; the embedding stays as it is
            total += quantized_bytes(t.elements, bits, width, group)
        else:
            total += t.elements * width
    return total


def quantized_bytes(values: int, bits: int, width: int, group: int) -> int:
    """The bytes `values` weights hold quantized here to `bits`, with a scale and a bias of `width` bytes for every
    `group` of them (quant.quantize's affine layout)."""
    return values * bits // 8 + (values // group) * 2 * width


def budget(working_set: int) -> int:
    """What a model may take of the GPU's working set: the rest holds a pass (READING_MARGIN), the prefixes a decider
    keeps by default (a sixteenth of memory, which is at most a tenth of the working set macOS gives the GPU) and a
    reading of MIN_READING at least. Admitting more loads a model with room for no prompt: an 8 GB Mac admitted Kettei
    Core at 8 bits (3.7 GB resident against 4.3 GB) and then read 1 token at a time, refusing every request."""
    return working_set - READING_MARGIN - min(PREFIX_BUDGET, working_set // 10) - MIN_READING


def admit(
    found: Sequence[Tensor],
    precision: str,
    working_set: int | None,
    *,
    can_quantize: bool = True,
) -> int:
    """The bytes `found` holds at `precision`, if it fits (or nothing is known of this Mac), else InsufficientMemory
    naming what would. It is judged by this Mac's memory, as setup judges it, never by what other apps hold now."""

    def size(meant: Precision) -> int:
        return resident_bytes(found, meant.width, meant.bits)

    meant = get_precision(precision)
    needed = size(meant)
    if working_set is None or needed <= budget(working_set):
        return needed
    offered = [get_precision(p) for p in OFFERED]
    fitting = [p.name for p in offered if (can_quantize or not p.changes_weights) and size(p) <= budget(working_set)]
    quantized = [name for name in fitting if PRECISIONS[name].changes_weights]
    gib = 1 << 30
    raise InsufficientMemory(
        f"the model needs about {needed / gib:.1f} GB at {precision}; this Mac's GPU keeps {working_set / gib:.1f} GB "
        f"resident, {budget(working_set) / gib:.1f} GB of it for a model. "
        + (f"Precisions that fit: {', '.join(fitting)}." if fitting else "No precision fits.")
        + (" Choose a smaller published build." if not can_quantize else "")
        + (
            f" {', '.join(quantized)} quantizes the model's weights, which changes its answers: choose one only if"
            " that is acceptable, and check it with ippon verify."
            if quantized
            else ""
        )
    )
