"""How a layer's operations are evaluated: MLX compositions, fused Metal kernels and bounded attention query
blocks (`kernels/`). Chosen once, at load, from the chip profile's switches (`profiles.py`, and
IPPON_SWITCHES to override them), and set on the backbone as `backbone.ops`; architectures call it and never
choose themselves.

- `norm(x, w, eps, kind)`: RMSNorm ("rms") or LayerNorm without a bias ("layer") over the last axis, in fp32,
  times the fp32 weight, rounded once to x's dtype;
- `add_norm(x, delta, w, eps, kind)`: the residual x + delta rounded to x's dtype, and its norm;
- `glu(gate_up, activation)`: silu(gate) * up or gelu(gate) * up from one projection's two halves, in fp32,
  rounded once;
- `causal_conv(x, window, taps, silu)`: a causal depthwise convolution over the first `taps.shape[0]` columns
  of x's rows, continuing `window` (the K - 1 inputs before), in fp32, through SiLU when asked, rounded once;
- `gated_conv(proj, window, taps)`: LFM2's C * conv(B * x) from proj's [B | C | x] rows (B * x rounded, the
  causal convolution rounded, then times C);
- `attention(q, k, v, scale, mask)`: MLX's scaled dot-product attention (mask None, "causal" or an array) by default.
  The explicit attention=bounded memory mode evaluates unmasked fp16/bf16 head-size-256 calls in query blocks when
  their unfused scores would occupy at least 512 MiB; every block still sees all keys, and all layers run eagerly
  in this mode. Query-matmul shapes can change rounding. A pass whose scores MLX couldn't allocate at all is read in
  such blocks under the default too (`kernels/attention.py`);
- `gated_delta(q, k, v, gamma, beta, state)`: Gated DeltaNet's recurrence, MLX's chunked
  `mx.fast.gated_delta_update`, from `state` (None: zeros; one state of batch 1 starts every sequence); (outputs,
  final state). Its inputs come in fp32 (switch gated_delta=mlx), or q, k and v in the compute dtype
  (gated_delta=half: the kernel keeps its state in fp32 and returns outputs in that dtype). Never masked: MLX's
  masked recurrence is another kernel, taking 30 to 100 times as long, and a batch is right-padded, so its padding
  reaches no real row;
- `glu_mm(x, w, activation)`: glu(x @ w.T) for w = [gate; up], through the matmul with the GLU in its epilogue
  where choices.py picks one for the shape and rows (`kernels/gemm.py`), else `glu(mm(x, w))`;
- `mm(x, w)`: x @ w.T for any weight matrix (`quant.mm`), a full-precision one through the tile choices.py picks for
  its shape and x's rows on this chip, when the profile's "gemm" switch is "kernel" (`kernels/gemm.py`); a packed
  one past the rows choices.py unpacks it from (`quant.unpacked`) as a full-precision one;
- `route(x, router, top_k)`, `expert_mm(x, w, experts)`, `combine(y, order, top)`: a mixture of experts' routing
  and grouping (every row's top_k experts, the (row, expert) pairs sorted by expert, their rows of x in that order),
  one gathered matmul by the experts' stacked matrices over the sorted pairs, and the pairs' outputs back in row
  order, weighted and summed; all on the GPU, one launch per projection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Self

import mlx.core as mx

from .. import quant
from ..errors import UnsupportedModel
from ..kernels import attention as attention_kernels
from ..kernels import conv as conv_kernels
from ..kernels import delta as delta_kernels
from ..kernels import gemm
from ..kernels import norm as norm_kernels
from ..kernels.launches import Launches
from ..profiles import Profile, Switches
from .rope import RoPE

F32 = mx.float32


ACTIVATION_NAMES = {"gelu_new": "gelu_tanh", "gelu_pytorch_tanh": "gelu_tanh"}  # transformers' names for ours


def activation(x: mx.array, name: str) -> mx.array:
    """An activation's formula, in x's dtype (pass fp32 for fp32 arithmetic): silu, exact gelu, gelu_tanh (GELU's
    tanh approximation; transformers' gelu_new and gelu_pytorch_tanh), relu, tanh."""
    name = ACTIVATION_NAMES.get(name, name)
    if name == "silu":
        return x * mx.sigmoid(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))
    if name == "gelu_tanh":
        return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    if name == "relu":
        return mx.maximum(x, 0.0)
    if name == "tanh":
        return mx.tanh(x)
    raise ValueError(f"unknown activation {name!r}")


_activation = activation  # the formula table, for methods whose parameter is named `activation`


def _compiled(name: str) -> Any:
    """The activation in fp32, rounded once to x's dtype, as one compiled kernel of any shape."""
    return mx.compile(lambda x: activation(x.astype(F32), name).astype(x.dtype), shapeless=True)


ACTIVATIONS = {name: _compiled(name) for name in ("gelu", "gelu_new", "gelu_pytorch_tanh", "relu")}


def activate(x: mx.array, name: str) -> mx.array:
    """An activation in fp32, of x's dtype, in one kernel."""
    if name not in ACTIVATIONS:
        raise UnsupportedModel(f"activation {name!r} is not supported")
    out: mx.array = ACTIVATIONS[name](x)
    return out


def layer_norm(x: mx.array, w: mx.array, b: mx.array, eps: float) -> mx.array:
    """LayerNorm (statistics in fp32), of x's dtype: one kernel when the weights are in x's dtype."""
    y = mx.fast.layer_norm(x, w, b, eps)
    return y if y.dtype == x.dtype else y.astype(x.dtype)


@dataclass(frozen=True)
class Ops:
    norm_kernel: bool = False
    conv_kernel: bool = False
    gemm_tiles: Mapping[str, gemm.Bands[gemm.Tile]] = field(
        default_factory=dict, hash=False
    )  # choices.py's, by gemm.key
    half_delta: bool = False  # Gated DeltaNet's q, k, v in the compute dtype (its state in fp32), else in fp32
    delta_kernel: bool = False  # Gated DeltaNet's q, k, v prepared in one kernel (kernels/delta.py)
    pipeline: int = 0  # hand the GPU the pass so far every this many layers, while Python builds the rest (0: off)
    pipeline_first: int = 0  # and once after this many layers too (0: no earlier hand-over)
    attention_min_scores: int = 0  # attention_bounded from this many bytes of the scores MLX would hold (0: always)
    attention_bounded: bool = False  # evaluated MLX query blocks against every key; Passes runs layers eagerly
    # MLX runs its kernels as a chip with neural accelerators (`Profile.mlx_neural_accelerators`): which
    # head-size-256 attention calls it fuses (`attention_kernels.mlx_fuses`)
    mlx_nax: bool = False
    # the MLX running is the one the facts about MLX's own kernels were measured on (`Profile.mlx_fitted`): which
    # attention calls it fuses and which Gated DeltaNet head counts it takes (FUSED_DELTA); else neither is assumed
    mlx_fitted: bool = True
    compiled: bool = False  # each layer one replayed mx.compile trace, per kind and pass shape (`common.Compiled`)
    _tiles: Launches[gemm.Tile | None] = field(default_factory=Launches, hash=False, compare=False, repr=False)

    @classmethod
    def of(cls, ops: Ops) -> Self:
        """`ops`'s settings (and chosen tiles) in this subclass, which measures or times them (precision.PeakOps,
        profile.TimedOps), its own fields at their defaults."""
        return cls(**{f.name: getattr(ops, f.name) for f in fields(Ops)})

    def _kernels_apply(self, x: mx.array) -> bool:
        return x.dtype in (mx.float16, mx.bfloat16, mx.float32) and mx.default_device() == mx.gpu

    def _tile(self, x: mx.array, w: mx.array, glu: str | None = None) -> gemm.Tile | None:
        """The chosen tile for x @ w.T (with `glu`, its fused GLU) on x's rows, looked up once per shape."""
        k = x.shape[-1]
        key = (x.dtype, w.shape[0], k, x.size // k, glu)
        if key not in self._tiles:
            self._tiles[key] = gemm.tile_for(self.gemm_tiles, x.dtype, w.shape[0], k, x.size // k, glu)
        return self._tiles[key]

    def mm(self, x: mx.array, w: quant.Matrix) -> mx.array:
        w = quant.unpacked(x, w)
        if self.gemm_tiles and isinstance(w, mx.array) and x.dtype == w.dtype and self._half_on_gpu(x):
            tile = self._tile(x, w)
            if tile is not None:
                return gemm.gemm(x, w, tile)
        return quant.mm(x, w)

    def mm_bias(self, x: mx.array, w: quant.Matrix, b: mx.array) -> mx.array:
        """x @ w.T + b: the bias in the matmul's epilogue (`mx.addmm`, added in fp32 and rounded once), or after a
        chosen tile's or a quantized matrix's product."""
        w = quant.unpacked(x, w)
        if isinstance(w, mx.array) and x.dtype == w.dtype:
            if self.gemm_tiles and self._half_on_gpu(x) and self._tile(x, w) is not None:
                return self.mm(x, w) + b
            return mx.addmm(b, x, w.T)
        return quant.mm(x, w) + b

    def glu_mm(self, x: mx.array, w: quant.Matrix, activation: str = "silu") -> mx.array:
        w = quant.unpacked(x, w)
        if self.gemm_tiles and isinstance(w, mx.array) and x.dtype == w.dtype and self._half_on_gpu(x):
            tile = self._tile(x, w, activation)
            if tile is not None:
                return gemm.glu_gemm(x, w, tile, activation)
        if (fused := quant.glu_mm(x, w, activation)) is not None:  # a quantized one's chosen fused tile
            return fused
        return self.glu(self.mm(x, w), activation)

    def _half_on_gpu(self, x: mx.array) -> bool:
        return x.dtype in (mx.bfloat16, mx.float16) and mx.default_device() == mx.gpu

    def norm_rope(
        self,
        proj: mx.array,
        offset: int,
        stride: int,
        heads: int,
        width: int,
        w: mx.array | None,
        eps: float,
        start: int,
        positions: mx.array | None,
        rope: RoPE,
    ) -> mx.array:
        """Attention's queries or keys [sequences, heads, n, width]: each head's `width` channels at `offset + head *
        stride` of proj's rows, through the per-head RMSNorm `w` (None: none) and `rope` at positions start.. (or
        each row's `positions`), as MLX's composition."""
        b, n = proj.shape[0], proj.shape[1]
        x = proj[..., offset : offset + heads * stride].reshape(b, n, heads, stride)[..., :width]
        if w is not None:
            x = self.norm(x, w, eps)
        return rope(x.transpose(0, 2, 1, 3), start, positions)

    def attention(self, q: mx.array, k: mx.array, v: mx.array, scale: float, mask: mx.array | str | None) -> mx.array:
        if (
            self.attention_bounded
            and mask is None
            and mx.default_device() == mx.gpu
            and attention_kernels.applies(q, k, v, mask)
            and attention_kernels.score_bytes(q, k) >= self.attention_min_scores
            and not self._mlx_fuses(q, k, mask)
        ):
            return attention_kernels.bounded(q, k, v, scale)
        if (
            mask is None
            and mx.default_device() == mx.gpu
            and attention_kernels.score_bytes(q, k) > attention_kernels.score_limit()
            and not self._mlx_fuses(q, k, mask)
        ):  # MLX would hold scores it cannot allocate: the request would fail, so it reads in query blocks
            return attention_kernels.bounded(q, k, v, scale)
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    def _mlx_fuses(self, q: mx.array, k: mx.array, mask: mx.array | str | None) -> bool:
        """Whether MLX runs a fused kernel itself for this attention call; never assumed under an MLX other than
        the one measured, where reading in query blocks is the side that can't run out of memory."""
        return self.mlx_fitted and attention_kernels.mlx_fuses(q, k, mask, self.mlx_nax)

    def gated_delta(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        gamma: mx.array,
        beta: mx.array,
        state: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if state is not None and state.shape[0] != q.shape[0]:  # MLX's unmasked kernel doesn't broadcast it
            state = mx.broadcast_to(state, (q.shape[0], *state.shape[1:]))
        hk, hv, dk, dv = q.shape[2], v.shape[2], q.shape[3], v.shape[3]
        fits = self.mlx_fitted and mx.default_device() == mx.gpu  # FUSED_DELTA is MLX 0.32's
        groups = fused_delta_groups(hk, hv, dk, dv) if fits else None
        if groups is None:
            o, final = mx.fast.gated_delta_update(q, k, v, gamma, beta, initial_state=state)
            return o, final
        return _padded_delta(q, k, v, gamma, beta, state, *groups)

    def delta_inputs(
        self,
        mixed: mx.array,
        proj: mx.array,
        neg_a: mx.array,
        dt_bias: mx.array,
        k_heads: int,
        k_dim: int,
        v_heads: int,
        v_dim: int,
        scale: float,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Gated DeltaNet's q, k, v in fp32 from the rows [q | k | v] of its convolution's output, q and k
        L2-normalized per head (eps 1e-6), q times `scale`; and its gates from proj's last 2 * v_heads columns (b, a):
        beta = sigmoid(b), gamma = exp(neg_a * softplus(a + dt_bias)), fp32. With `half_delta`, q, k and v stay in the
        compute dtype (the recurrence's state is fp32 either way)."""
        lead, kd = mixed.shape[:-1], k_heads * k_dim
        b, a = proj[..., -2 * v_heads : -v_heads], proj[..., -v_heads:]
        if self.half_delta and proj.dtype != F32:
            q = mixed[..., :kd].reshape(*lead, k_heads, k_dim)
            k = mixed[..., kd : 2 * kd].reshape(*lead, k_heads, k_dim)
            v = mixed[..., 2 * kd :].reshape(*lead, v_heads, v_dim)
            # l2norm(x) = rms_norm(x, eps / D) / sqrt(D): one fused norm, rounded once to the compute dtype
            q = mx.fast.rms_norm(q, None, 1e-6 / k_dim) * (scale * k_dim**-0.5)
            k = mx.fast.rms_norm(k, None, 1e-6 / k_dim) * k_dim**-0.5
            beta = mx.sigmoid(b).astype(F32)
            gamma = mx.exp(neg_a * mx.logaddexp(a.astype(F32) + dt_bias, 0.0))  # exp(-e^A * softplus)
            return q, k, v, beta, gamma
        if self.delta_kernel and self._kernels_apply(mixed) and mixed.dtype == proj.dtype:
            return delta_kernels.delta_inputs(mixed, proj, neg_a, dt_bias, k_heads, k_dim, v_heads, v_dim, scale)
        q = mixed[..., :kd].reshape(*lead, k_heads, k_dim).astype(F32)
        k = mixed[..., kd : 2 * kd].reshape(*lead, k_heads, k_dim).astype(F32)
        v = mixed[..., 2 * kd :].reshape(*lead, v_heads, v_dim).astype(F32)
        q = q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6) * scale
        k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        beta = mx.sigmoid(b).astype(F32)
        gamma = mx.exp(neg_a * mx.logaddexp(a.astype(F32) + dt_bias, 0.0))  # exp(-e^A * softplus)
        return q, k, v, beta, gamma

    def norm(self, x: mx.array, w: mx.array, eps: float, kind: str = "rms", groups: int = 1) -> mx.array:
        """The norm of x's last axis times w (fp32, one per channel), in fp32 and rounded once; with `groups`, an
        RMSNorm whose statistics are taken over each of that many equal groups of the channels."""
        if self.norm_kernel and self._kernels_apply(x):
            out = norm_kernels.add_norm(x, None, w, eps, kind, groups)
            assert isinstance(out, mx.array)
            return out
        if groups != 1:
            if kind != "rms":
                raise ValueError("grouped statistics are for RMSNorm")
            grouped = x.astype(F32).reshape(*x.shape[:-1], groups, x.shape[-1] // groups)
            normed = mx.fast.rms_norm(grouped, None, eps).reshape(x.shape)
            return (normed * w).astype(x.dtype)
        if kind == "layer":
            return mx.fast.layer_norm(x.astype(F32), w, None, eps).astype(x.dtype)
        return mx.fast.rms_norm(x.astype(F32), w, eps).astype(x.dtype)

    def add_norm(
        self, x: mx.array, delta: mx.array, w: mx.array, eps: float, kind: str = "rms", groups: int = 1
    ) -> tuple[mx.array, mx.array]:
        if self.norm_kernel and self._kernels_apply(x) and delta.dtype == x.dtype and delta.shape == x.shape:
            out = norm_kernels.add_norm(x, delta, w, eps, kind, groups)
            assert isinstance(out, tuple)
            return out
        h = (x + delta).astype(x.dtype)
        return h, self.norm(h, w, eps, kind, groups)

    def add_layer_norm(self, x: mx.array, delta: mx.array, w: mx.array, b: mx.array, eps: float) -> mx.array:
        """LayerNorm with a bias of x + delta (a post-norm encoder's residual, BERT's): the sum in x's dtype, the
        statistics in fp32, of x's dtype; one kernel when the weights are in x's dtype."""
        return layer_norm(x + delta, w, b, eps)

    def route(
        self,
        x: mx.array,
        router: quant.Matrix,
        top_k: int,
        inputs: mx.array | None = None,
        expert_scale: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """A mixture of experts' routing, as transformers' Qwen3.5-MoE computes it: router logits (x by the router,
        as stored) in x's dtype, their softmax in fp32, the top_k renormalized and rounded to x's dtype; with
        `expert_scale`, as its Gemma 4 does: the renormalized weights times their experts' scales, kept in fp32. Then
        the rows grouped by expert: (each row's weights [rows, top_k]; `order`, the (row, choice) pairs sorted by
        expert [rows * top_k]; those pairs' experts; their rows of `inputs` (x when None) [rows * top_k, 1, width])."""
        probs = mx.softmax(quant.mm(x, router).astype(F32), axis=-1)  # not self.mm: profiles time it with routing
        chosen = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[:, :top_k]
        top = mx.take_along_axis(probs, chosen, axis=-1)
        top = top / top.sum(axis=-1, keepdims=True)
        top = top.astype(x.dtype) if expert_scale is None else top * expert_scale[chosen]
        flat = chosen.reshape(-1)
        order = mx.argsort(flat)
        source = x if inputs is None else inputs
        return top, order, flat[order], source[order // top_k][:, None, :]

    def expert_mm(self, x: mx.array, w: quant.Stacked | quant.JoinedStacked, experts: mx.array) -> mx.array:
        """Each sorted pair's row of x by its expert's matrix (MLX's gather matmul, the experts' runs as segments)."""
        return w.gather(x, experts, sorted_experts=True)

    def combine(self, y: mx.array, order: mx.array, top: mx.array) -> mx.array:
        """The pairs' outputs [rows * top_k, width] in expert order back to their rows, times their weights, summed
        in fp32 and rounded once: [rows, width]."""
        k = top.shape[-1]
        back = mx.take(y, mx.argsort(order), axis=0).reshape(-1, k, y.shape[-1])
        return ((back * top[..., None]).astype(F32).sum(axis=1)).astype(y.dtype)

    def glu(self, gate_up: mx.array, activation: str = "silu") -> mx.array:
        gate, up = mx.split(gate_up, 2, axis=-1)
        return (_activation(gate.astype(F32), activation) * up.astype(F32)).astype(gate_up.dtype)

    def gated_norm(self, o: mx.array, proj: mx.array, offset: int, w: mx.array, eps: float) -> mx.array:
        """Gated DeltaNet's output, w * rmsnorm(o) * silu(z) per head, z the columns `offset`.. of proj's rows,
        rounded as Qwen3.5's reference rounds it (o first, to proj's dtype: o may come in fp32 or already in it);
        [sequences, length, heads * width] in proj's dtype."""
        dtype = proj.dtype
        if self.norm_kernel and self._kernels_apply(proj) and o.dtype in (F32, dtype):
            return norm_kernels.gated_norm(o, proj, offset, w, eps, dtype)
        batch, length, heads, width = o.shape
        normed = mx.fast.rms_norm(o.astype(dtype).astype(F32), None, eps).astype(dtype)
        zf = proj[..., offset : offset + heads * width].reshape(batch, length, heads, width).astype(F32)
        y = ((w * normed).astype(F32) * (zf * mx.sigmoid(zf))).astype(dtype)
        return y.reshape(batch, length, heads * width)

    def causal_conv(self, x: mx.array, window: mx.array, taps: mx.array, silu: bool) -> mx.array:
        if self.conv_kernel and self._kernels_apply(x):
            return conv_kernels.causal_conv(x, window, taps, silu)
        channels, k = taps.shape
        batch, length = x.shape[0], x.shape[1]
        padded = mx.concatenate(
            [mx.broadcast_to(window, (batch, k - 1, channels)).astype(x.dtype), x[..., :channels]], axis=1
        ).astype(F32)
        out = padded[:, 0:length] * taps[:, 0]
        for j in range(1, k):
            out = out + padded[:, j : j + length] * taps[:, j]
        return (out * mx.sigmoid(out) if silu else out).astype(x.dtype)

    def gated_conv(self, proj: mx.array, window: mx.array | None, taps: mx.array) -> mx.array:
        if self.conv_kernel and self._kernels_apply(proj):
            return conv_kernels.gated_conv(proj, window, taps)
        d, k = taps.shape
        batch, length = proj.shape[0], proj.shape[1]
        bx = proj[..., :d] * proj[..., 2 * d :]
        if window is None:
            window = mx.zeros((1, k - 1, d), proj.dtype)
        padded = mx.concatenate([mx.broadcast_to(window, (batch, k - 1, d)).astype(bx.dtype), bx], axis=1)
        padded = padded.astype(F32)
        z = padded[:, 0:length] * taps[:, 0]
        for j in range(1, k):
            z = z + padded[:, j : j + length] * taps[:, j]
        return proj[..., d : 2 * d] * z.astype(proj.dtype)


PIPELINE_LAYERS = 4
MLX = Ops()
FUSED = Ops(norm_kernel=True, conv_kernel=True, delta_kernel=True)


# The key/value head counts MLX's fused Gated DeltaNet kernels take, at 128-wide heads (MLX 0.32, profiles.FITTED_MLX,
# so only under it: `Ops.mlx_fitted`): any other shape makes a composition take 50 to 100 times as long (8/16 heads:
# 99 ms against 2 ms for 16/32, 2,048 tokens, M5 Pro).
FUSED_DELTA = ((16, 16), (16, 32), (16, 48), (16, 64), (24, 24), (32, 32))
DELTA_WIDTH = 128


def fused_delta_groups(hk: int, hv: int, dk: int, dv: int) -> tuple[int, int, int] | None:
    """How to run a recurrence of hk key and hv value heads (dk, dv wide) on a fused kernel: (key heads per group,
    value heads per group, groups), each group zero-padded to that shape and its heads to 128 wide; None when the
    shape is fused already or can't be (heads wider than 128, or value heads not a multiple of key heads). Zero heads
    and zero columns change nothing for the real ones: a zero key adds nothing to a state, a zero value nothing to an
    output."""
    if (hk, hv) in FUSED_DELTA and dk == dv == DELTA_WIDTH:
        return None
    if dk > DELTA_WIDTH or dv > DELTA_WIDTH or hv % hk:
        return None
    ratio = hv // hk
    fits = [(ck, cv) for ck, cv in FUSED_DELTA if cv == ck * ratio]
    if not fits:
        return None
    ck, cv = next(((ck, cv) for ck, cv in fits if ck >= hk), max(fits))  # the smallest that holds every head
    return ck, cv, -(-hk // ck)


def _padded_delta(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    gamma: mx.array,
    beta: mx.array,
    state: mx.array | None,
    ck: int,
    cv: int,
    groups: int,
) -> tuple[mx.array, mx.array]:
    """The recurrence over `groups` groups of ck key and cv value heads, zero-padded (`fused_delta_groups`)."""
    dk, hv, dv = q.shape[3], v.shape[2], v.shape[3]
    w = DELTA_WIDTH

    def pad(x: mx.array, heads: int, width: int | None) -> mx.array:
        widths = [(0, 0), (0, 0), (0, heads - x.shape[2])] + ([(0, width - x.shape[3])] if width else [])
        return mx.pad(x, widths)

    q, k = pad(q, ck * groups, w), pad(k, ck * groups, w)
    v, gamma, beta = pad(v, cv * groups, w), pad(gamma, cv * groups, None), pad(beta, cv * groups, None)
    if state is not None:  # [b, hv, dv, dk]
        state = mx.pad(state, [(0, 0), (0, cv * groups - hv), (0, w - dv), (0, w - dk)])
    outs, finals = [], []
    for g in range(groups):
        ks, vs = slice(g * ck, (g + 1) * ck), slice(g * cv, (g + 1) * cv)
        start = None if state is None else state[:, vs]
        o, f = mx.fast.gated_delta_update(
            q[:, :, ks], k[:, :, ks], v[:, :, vs], gamma[:, :, vs], beta[:, :, vs], initial_state=start
        )
        outs.append(o)
        finals.append(f)
    o = outs[0] if groups == 1 else mx.concatenate(outs, axis=2)
    final = finals[0] if groups == 1 else mx.concatenate(finals, axis=1)
    # copies: a slice of the padded state, kept in a cache or a kept prefix, would hold the whole padded array
    return mx.contiguous(o[:, :, :hv, :dv]), mx.contiguous(final[:, :hv, :dv, :dk])


def for_switches(
    switches: Switches,
    gemm_tiles: Mapping[str, gemm.Bands[gemm.Tile]] | None = None,
    mlx_nax: bool = False,
    mlx_fitted: bool = True,
) -> Ops:
    """The ops `switches` choose (`profiles.Switches`). Matmuls run on `gemm_tiles`, choices.py's tiles by
    `gemm.key`, none where `choices.active` leaves MLX's matmuls; `mlx_nax` and `mlx_fitted` are what the profile
    knows of the MLX running (`Ops`)."""
    return Ops(
        norm_kernel=switches.add_norm == "kernel",
        conv_kernel=switches.conv == "kernel",
        gemm_tiles=dict(gemm_tiles or {}),
        half_delta=switches.gated_delta == "half",
        delta_kernel=switches.delta_inputs == "kernel",
        pipeline=PIPELINE_LAYERS if switches.pipeline in ("layers", "early") else 0,
        pipeline_first=1 if switches.pipeline == "early" else 0,
        attention_min_scores=attention_kernels.LARGE_SCORES if switches.attention == "bounded" else 0,
        attention_bounded=switches.attention == "bounded",
        mlx_nax=mlx_nax,
        mlx_fitted=mlx_fitted,
        compiled=switches.compile == "layers",
    )


def for_profile(profile: Profile, gemm_tiles: Mapping[str, gemm.Bands[gemm.Tile]] | None = None) -> Ops:
    """The ops a resolved profile runs: its switches' (`for_switches`), with what it knows of the MLX running."""
    return for_switches(profile.switches, gemm_tiles, profile.mlx_neural_accelerators, profile.mlx_fitted)
