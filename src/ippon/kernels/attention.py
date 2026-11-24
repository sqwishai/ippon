"""Memory paths for head-size-256 attention where MLX 0.32.4 runs its unfused path.

    out = bounded(q, k, v, scale)                 # evaluated MLX query blocks, all keys, no mask

At head size 256 (the Qwen3.5 models' full-attention layers: pplx-decider flash, Clef-Flash, d3), MLX fuses
attention of more than 8 queries only on an M5 for 1,024 queries or more under a causal or array mask, and on M1 to
M4 only for causal self-attention of 2,048 queries or more. Every other call (both-ways attention, shorter prompts,
a branch's rows against a longer cache) multiplies q by k into a score matrix in memory, takes its softmax and
multiplies that by v. At d3-mini's shape (16 query and 4 key/value heads, fp16) that call needs 0.53, 2.06 and 4.59
GiB beyond its inputs and output at 4,096, 8,192 and 12,288 tokens (M5 Pro).

The default attention=mlx keeps MLX's operation, and reads a call whose scores MLX could not allocate in query blocks
(`score_limit`). The explicit attention=bounded memory mode runs unmasked fp16/bf16 calls in at most QUERY_ROWS
query rows at a time, each against every key, from LARGE_SCORES bytes of scores. Each block is evaluated before the
next, and selecting this mode disables layer compilation globally, even for calls that do not reach the threshold.
Full-key MLX softmax is retained, but different query-matmul shapes can change rounding: this is not a promise of
bitwise equivalence on every model or shape.

"""

from __future__ import annotations

import mlx.core as mx

HEAD = 256
QUERY_ROWS = 128
VECTOR_ROWS = 8
# attention=bounded reads in query blocks only from this many bytes of scores (`score_bytes`), 4,096 tokens of 16 query
# heads in fp16. MLX's unfused path holds about its score matrix beyond its inputs and output while it runs (M5 Pro,
# d3-mini's shape: 0.53, 2.06 and 4.59 GiB at 4,096, 8,192 and 12,288 tokens, for scores of 0.5, 2 and 4.5 GiB).
LARGE_SCORES = 512 * 2**20


def score_bytes(q: mx.array, k: mx.array) -> int:
    """The bytes of the score matrix MLX's unfused path would hold: batch x query heads x queries x keys, in q's
    dtype."""
    return q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * q.dtype.size


def score_limit() -> int:
    """The most bytes one score matrix can take now: MLX's largest buffer, and what the GPU working set has left.
    A both-ways pass past it ran out of memory (d3-mini at ~30,800 tokens a read wanted 30.3 GB against the M5 Pro's
    30.15 GB buffer limit); `bounded` reads it in query blocks instead."""
    info = mx.device_info()  # the CPU reports neither limit: nothing to cap there
    if "max_buffer_length" not in info or "max_recommended_working_set_size" not in info:
        return 1 << 62
    left = int(info["max_recommended_working_set_size"]) - mx.get_active_memory()
    return min(int(info["max_buffer_length"]), left)


def bounded(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    """Unmasked MLX attention in evaluated query blocks, each against every key, with at most QUERY_ROWS score
    rows. Softmax still covers all keys in MLX's compute dtype, although query-matmul shapes can change rounding.
    Each block completes before the next allocates intermediates; `Passes._compiled_layer` keeps this explicit
    memory mode outside compiled traces."""
    length = q.shape[2]
    starts = list(range(0, length, QUERY_ROWS))
    if len(starts) > 1 and length - starts[-1] <= VECTOR_ROWS:
        starts[-1] = length - VECTOR_ROWS - 1  # keep the final block on MLX's same unfused arithmetic path
    outputs: list[mx.array] = []
    for start, end in zip(starts, [*starts[1:], length], strict=True):
        block = q[:, :, start:end]
        outputs.append(mx.fast.scaled_dot_product_attention(block, k, v, scale=scale))
        mx.eval(outputs[-1])
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def mlx_fuses(q: mx.array, k: mx.array, mask: mx.array | str | None, nax: bool) -> bool:
    """Whether MLX 0.32 (`profiles.FITTED_MLX`) runs a fused kernel itself for this head-size-256 call (its
    `use_fallback`), running its kernels as a chip with (`nax`) or without neural accelerators: then its own runs."""
    if q.shape[2] <= VECTOR_ROWS:  # its vector kernel
        return True
    if nax:
        return q.shape[2] >= 1024 and isinstance(mask, (str, mx.array))
    return isinstance(mask, str) and q.shape[2] >= 2048 and q.shape[2] == k.shape[2]


def applies(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | str | None) -> bool:
    """Whether the bounded mode reads this call in query blocks: head size 256 for q, k and v, more than 8 queries,
    fp16 or bf16 (MLX's unfused path is exact in fp32), whole groups of query heads, a causal mask's queries no more
    than its keys, and an array mask of at most 4 dimensions, boolean or additive (of a floating dtype, taken in
    q's)."""
    if any(x.ndim != 4 or x.shape[-1] != HEAD or 0 in x.shape for x in (q, k, v)) or q.shape[2] <= VECTOR_ROWS:
        return False
    if q.dtype not in (mx.float16, mx.bfloat16) or not (q.dtype == k.dtype == v.dtype):
        return False
    if q.shape[1] % k.shape[1] or k.shape != v.shape or k.shape[0] != q.shape[0]:
        return False
    if isinstance(mask, str):
        return mask == "causal" and q.shape[2] <= k.shape[2]
    if isinstance(mask, mx.array):
        return mask.ndim <= 4 and (mask.dtype == mx.bool_ or mx.issubdtype(mask.dtype, mx.floating))
    return mask is None
