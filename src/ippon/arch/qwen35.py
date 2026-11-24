"""Qwen3.5: Gated DeltaNet layers with a full-attention layer every few layers. The base of most open decision
models: Cloudflare's Clef and Clef-Flash, Perplexity's pplx-decider, vllm-sr d3.

The numbers follow Hugging Face's Qwen3_5 text model:

- RMSNorm is zero-centred: x / rms(x) * (1 + w) in fp32, rounded once (the checkpoint stores w);
- a full-attention layer's q_proj gives each head its query and an output gate; queries and keys get a per-head
  RMSNorm and rotate_half RoPE on their first rotary_dim channels; the output is multiplied by sigmoid(gate);
- a Gated DeltaNet layer projects to q, k, v (through a causal depthwise convolution and SiLU), z, b and a; q
  and k are L2-normalized (q also scaled by its dim^-1/2); the decay is exp(-exp(A_log) * softplus(a +
  dt_bias)) and the update rate sigmoid(b); the recurrence is MLX's chunked `mx.fast.gated_delta_update`; the
  output goes through a gated RMSNorm, w * norm(o) * silu(z), whose weight is not zero-centred;
- the MLP is SwiGLU, here with the gate's SiLU and product in fp32, rounded once.

Positions are multimodal RoPE's, as Qwen3_5Model.get_rope_index and Qwen3_5TextRotaryEmbedding give them: every row
has a (t, h, w) position, and each of the rotary dims / 2 frequency pairs turns by one of them (`mrope_axes`). A text
row's three are its index plus an offset, so text alone is plain RoPE. An image's rows (`media`, placed by their
grid) start at the running position p, t = p + frame, h = p + block row, w = p + block column, and the text after
the image goes on from p + max(block rows, block columns): the offset grows by that less the image's rows (`places`).
A pass without image rows keeps plain RoPE at the shifted positions; one with them rotates its queries and keys
three times, once per axis (`_mrope`).

Matrices that read the same input are merged at load: q (with its gate), k and v; the Gated DeltaNet's four
input projections; the MLP's gate and up.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ..checkpoint import EMBEDDING
from ..errors import UnsupportedModel
from ..quant import Matrix, Table, arrays, concat
from ..weights import Weights
from .cache import Attention, HybridCache, Recurrent
from .common import (
    Held,
    Hybrid,
    Read,
    Recurrence,
    Rows,
    Snapshots,
    attend,
    embed,
    finish_layer,
    output_head,
    placeholders,
    rows_of,
    stacked_experts,
    window_before,
)
from .rope import RoPE

F32 = mx.float32
FAMILIES = ("qwen3_5_text", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_moe")  # its config.json model_types
MROPE_SECTION = (11, 11, 10)  # transformers' default for Qwen3.5: frequency pairs per axis (t, h, w)


def mrope_axes(dims: int, section: Sequence[int], width: int) -> np.ndarray:
    """Which position each of a head's `width` channels turns by (0: t, 1: h, 2: w), as Qwen3_5TextRotaryEmbedding's
    recomposition_frequencies interleaves them over the dims / 2 frequency pairs: pair j takes h when j % 3 == 1 and
    j < 3 * section[1], w when j % 3 == 2 and j < 3 * section[2], t otherwise; both channels of a pair (j and
    j + dims / 2, rotate_half's) the same. Channels past `dims` are not rotated (0)."""
    j = np.arange(dims // 2)
    pair = np.where((j % 3 == 1) & (j < 3 * section[1]), 1, np.where((j % 3 == 2) & (j < 3 * section[2]), 2, 0))
    out: np.ndarray = np.zeros(width, np.int32)
    out[:dims] = np.concatenate([pair, pair])
    return out


def places(
    tokens: np.ndarray, first: int, offset: int, grids: Mapping[int, tuple[int, int, int]]
) -> tuple[np.ndarray, int, dict[int, int]]:
    """One sequence's rows' (t, h, w) positions [3, rows], for `tokens` at indices first.. with the offset `offset`
    there: a text row at index i is at i + its offset on every axis; an image's row (its placeholder, `media_id`)
    of grid (frames, block rows, block columns) `grids[digest]` is at p + (frame, block row, block column), p its
    image's first row's index plus the offset, wherever the image's first row is (an image a pass reads the end of
    started in an earlier one). After an image's last row the offset grows by max(block rows, block columns) less its
    rows, as Qwen3_5Model.get_rope_index advances. Returns the positions, the offset after the last row, and where
    the offset changed: {the index after an image: the offset from there on}."""
    n = len(tokens)
    index = first + np.arange(n, dtype=np.int64)
    at, digests, row = placeholders(tokens)
    missing = {int(d) for d in np.unique(digests)} - set(grids)
    if missing:
        raise UnsupportedModel(f"a prompt holds {len(missing)} image(s) the pass was given no grid for")
    grid = np.array([grids[int(d)] for d in digests], np.int64).reshape(-1, 3)
    frames, high, wide = grid[:, 0], grid[:, 1], grid[:, 2]
    count = frames * high * wide
    if np.any(row >= count):
        raise ValueError("a placeholder names a row past its image's grid")
    last = row == count - 1
    change = np.zeros(n + 1, np.int64)
    np.add.at(change, at[last] + 1, (np.maximum(high, wide) - count)[last])
    running = offset + np.cumsum(change)
    out = np.broadcast_to(index + running[:n], (3, n)).copy()
    start = index[at] - row + running[at]  # each image row's p
    within = row % (high * wide)
    out[0, at] = start + row // (high * wide)
    out[1, at] = start + within // wide
    out[2, at] = start + within % wide
    changes = {int(first + e + 1): int(running[e + 1]) for e in at[last]}
    return out, int(running[n]), changes


@dataclass(frozen=True)
class Places:
    """Where a pass's rows sit for RoPE: text at positions `start`.. (or each row's `positions`, a branch pass's),
    shifted by the offset of the images before them; or, for a pass that holds image rows, every row's (t, h, w)
    positions `axes` [3, sequences x rows] int32, in the pass's row order."""

    start: int
    positions: mx.array | None = None
    axes: mx.array | None = None


@dataclass(frozen=True)
class Config:
    hidden: int
    layer_types: tuple[str, ...]  # "linear_attention" or "full_attention", per layer
    heads: int
    kv_heads: int
    head_dim: int
    rope: RoPE  # over the first rope.dims channels of each head
    eps: float
    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv_kernel: int
    tied: bool
    causal: bool = True  # False: full-attention layers see every position (vllm-sr d3)
    block: bool = False  # block attention (Backbone.block): the prefix both ways, then each tail both ways after it
    experts: int = 0  # a mixture-of-experts MLP (Qwen3.5-MoE): its experts, of which top_k run per token
    top_k: int = 0
    mrope: tuple[int, int, int] = MROPE_SECTION  # frequency pairs per position axis (t, h, w): `mrope_axes`

    @classmethod
    def from_hf(cls, config: Mapping[str, Any], *, causal: bool = True, block: bool = False) -> Config:
        """From a Hugging Face config.json, or its "text_config" for a multimodal checkpoint."""
        c = config.get("text_config", config)
        if c.get("model_type") not in FAMILIES:
            raise UnsupportedModel(f"not a Qwen3.5 text model: model_type {c.get('model_type')!r}")
        experts = int(c.get("num_experts") or 0)
        if experts and c.get("norm_topk_prob", True) is False:
            raise UnsupportedModel("a Qwen3.5 mixture of experts without renormalized top-k weights")
        if not c.get("attn_output_gate", True) or c.get("mlp_only_layers") or c.get("hidden_act", "silu") != "silu":
            raise UnsupportedModel("a Qwen3.5 variant this implementation does not cover (attention gate, MLP, act)")
        # the Gated DeltaNet output gate's activation (transformers' qwen4_exp reads it there; by default hidden_act);
        # the attention output gate is a sigmoid always.
        gate = c.get("output_gate_type") or "silu"
        if gate != "silu":
            raise UnsupportedModel(f"a Gated DeltaNet output gate of {gate!r} is not supported (only SiLU)")
        params = c.get("rope_parameters") or {}
        head_dim = int(c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"])
        partial = float(params.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
        rope = RoPE.from_hf(
            int(head_dim * partial),
            params,
            theta=float(c.get("rope_theta", 10_000.0)),
            max_positions=int(c.get("max_position_embeddings") or 0),
        )
        if rope.attention != 1.0 and rope.dims != head_dim:
            raise UnsupportedModel("YaRN over part of the head is not supported (its factor scales the rotated part)")
        section = tuple(int(s) for s in params.get("mrope_section") or MROPE_SECTION)
        if len(section) != 3:
            raise UnsupportedModel(f"an mrope_section of {list(section)}: three axes (t, h, w) are supported")
        return cls(
            hidden=int(c["hidden_size"]),
            layer_types=tuple(c["layer_types"]),
            heads=int(c["num_attention_heads"]),
            kv_heads=int(c["num_key_value_heads"]),
            head_dim=head_dim,
            rope=rope,
            eps=float(c["rms_norm_eps"]),
            k_heads=int(c["linear_num_key_heads"]),
            v_heads=int(c["linear_num_value_heads"]),
            k_dim=int(c["linear_key_head_dim"]),
            v_dim=int(c["linear_value_head_dim"]),
            conv_kernel=int(c["linear_conv_kernel_dim"]),
            tied=bool(c.get("tie_word_embeddings", False)),
            causal=causal and not block,
            block=block,
            experts=experts,
            top_k=int(c.get("num_experts_per_tok") or 0),
            mrope=(section[0], section[1], section[2]),
        )

    @property
    def conv_dim(self) -> int:
        return 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim


class Qwen35(Hybrid):
    """A loaded Qwen3.5 text model (see the module docstring)."""

    config: Config

    # each layer's weight matrices that inputs multiply (quant.mm): what quantization converts
    linears = ("gate_up", "down", "qkv", "o", "in", "out")
    gated_mlp = ("gate_up", "silu")  # the [gate; up] projection and its GLU's activation

    def __init__(self, config: Config, weights: Weights, dtype: mx.Dtype = mx.bfloat16, bits: int | None = None):
        self.config, self.dtype = config, dtype
        self.block = config.block
        c, root = config, weights.text_root()
        # A checkpoint converted to MLX stores its convolutions as [channels, taps, 1] and its zero-centred norm
        # weights already as 1 + w (mlx-vlm's sanitize); one from Hugging Face stores [channels, 1, taps] and w.
        conv = next(
            (k for k in weights.tensors if k.startswith(root) and k.endswith("linear_attn.conv1d.weight")), None
        )
        shifted = conv is not None and weights.tensors[conv].shape[-1] == 1

        def m(name: str) -> Matrix:  # a weight matrix, quantized as stored
            return weights.matrix(root + name, dtype)

        def kept(name: str) -> mx.array:  # fp32 when stored so (the reference then multiplies by it in fp32)
            return weights.get(root + name, F32 if weights.tensors[root + name].dtype == F32 else dtype)

        def w1(name: str) -> mx.array:  # zero-centred norm weight, as 1 + w in fp32
            return weights.get(root + name, F32) + (0.0 if shifted else 1.0)

        self.embed: Table = weights.table(root + EMBEDDING, dtype)
        self.final_norm = w1("norm.weight")
        head = output_head(weights, "", c.tied, dtype, "language_model.lm_head.weight")
        # None: no head stored and none tied, as in vllm-sr's d3-flash, whose readout is its own (readout.safetensors)
        self._output: Table | None = self.embed if c.tied else head
        self.layers: list[dict[str, Any]] = []  # matrices may be QuantizedMatrix
        for i, kind in enumerate(c.layer_types):
            p = f"layers.{i}."
            layer: dict[str, Any] = {
                "in_norm": w1(p + "input_layernorm.weight"),
                "post_norm": w1(p + "post_attention_layernorm.weight"),
            }
            if c.experts:
                layer |= self._experts_of(weights, root, p + "mlp.", dtype, bits)
            else:
                layer |= {
                    "gate_up": concat([m(p + "mlp.gate_proj.weight"), m(p + "mlp.up_proj.weight")]),
                    "down": m(p + "mlp.down_proj.weight"),
                }
            if kind == "full_attention":
                a = p + "self_attn."
                layer |= {
                    "qkv": concat([m(a + f"{x}_proj.weight") for x in "qkv"]),
                    "o": m(a + "o_proj.weight"),
                    "q_norm": w1(a + "q_norm.weight"),
                    "k_norm": w1(a + "k_norm.weight"),
                }
            elif kind == "linear_attention":
                g = p + "linear_attn."
                layer |= {
                    "in": concat([m(g + f"in_proj_{x}.weight") for x in ("qkv", "z", "b", "a")]),
                    "conv": weights.get(root + g + "conv1d.weight", F32).reshape(c.conv_dim, c.conv_kernel),
                    "neg_a": -mx.exp(weights.get(root + g + "A_log", F32)),
                    "dt_bias": weights.get(root + g + "dt_bias", F32),
                    "norm": kept(g + "norm.weight"),
                    "out": m(g + "out_proj.weight"),
                }
            else:
                raise UnsupportedModel(f"layer {i}: unknown layer type {kind!r}")
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(arrays([self.embed, self.final_norm, *([self._output] if self._output is not None else [])]))
        self._axes = mx.array(mrope_axes(c.rope.dims, c.mrope, c.head_dim))

    def _experts_of(self, weights: Weights, root: str, mlp: str, dtype: mx.Dtype, bits: int | None) -> dict[str, Any]:
        """A mixture-of-experts MLP's weights: the router, every expert's [gate; up] and down stacked (from one tensor
        per expert, transformers' fused [experts, rows, columns] tensors, or mlx-lm's switch_mlp ones), the shared
        expert ("gate_up" and "down", so it runs as a dense MLP does) and its gate."""
        c = self.config

        def m(name: str) -> Matrix:
            return weights.matrix(root + mlp + name, dtype)

        gate_up, down = stacked_experts(weights, root + mlp, dtype, c.experts, "switch_mlp")
        return {
            "router": weights.get(root + mlp + "gate.weight", dtype),  # never quantized: it chooses the experts
            "experts_gate_up": gate_up.quantized(bits),
            "experts_down": down.quantized(bits),
            "gate_up": concat([m("shared_expert.gate_proj.weight"), m("shared_expert.up_proj.weight")]),
            "down": m("shared_expert.down_proj.weight"),
            "shared_gate": weights.get(root + mlp + "shared_expert_gate.weight", dtype),
        }

    # -- the Backbone contract ----------------------------------------------------------------------------------

    @property
    def causal(self) -> bool:
        return self.config.causal

    @property
    def output_matrix(self) -> Table:
        if self._output is None:
            raise UnsupportedModel("this checkpoint has no language-model head (lm_head.weight) and doesn't tie it")
        return self._output

    # -- the layers ---------------------------------------------------------------------------------------------

    @property
    def layer_kinds(self) -> Sequence[str]:
        return self.config.layer_types

    def _forward(self, ids: mx.array, cache: HybridCache | None, r: Rows, keep: Read) -> mx.array:
        """Final-normed hidden states [sequences, rows, hidden] of the `keep` rows. Each
        layer's MLP output is added to the residual together with the next norm (`ops.add_norm`)."""
        at = self._places(ids, cache, r)
        plain = at.axes is None and at.start == r.start and at.positions is r.positions  # the pass's own positions
        x = embed(self.embed, ids, self.media)
        return self._decoder(x, cache, r, keep, self.final_norm, self.config.eps, at=None if plain else at)

    def _mixer(
        self,
        kind: Hashable,
        layer: dict[str, Any],
        h: mx.array,
        block: Any,
        r: Rows,
        read: Read | None,
        kept: list[Snapshots],
        at: Any,
    ) -> mx.array:
        """`at`: where the rows sit (`_places`), None for the pass's own positions."""
        if kind == "full_attention":
            assert block is None or isinstance(block, Attention)
            return self._attention(h, layer, block, r, read, Places(r.start, r.positions) if at is None else at)
        assert block is None or isinstance(block, Recurrent)
        return self._delta_net(h, layer, block, r, read, kept)

    def _feed_forward(self, kind: Hashable, layer: dict[str, Any], h: mx.array) -> mx.array:
        return self._mlp(h, layer)

    def _places(self, ids: mx.array, cache: HybridCache | None, r: Rows) -> Places:
        """Where the pass's rows sit (`places`): with no image rows read so far nor in the pass, the text positions
        as they are; after images, shifted by the cache's offset; with image rows, each row's (t, h, w). A prefill's
        and a branch head's changes of the offset go into the cache (a branch's tails continue the head but leave
        the cache where the head ends)."""
        offset = cache.offset(r.start) if cache is not None else 0
        if self.media is None and not offset:
            return Places(r.start, r.positions)
        tokens = np.array(ids)
        if not placeholders(tokens)[0].size:
            return Places(r.start + offset, None if r.positions is None else r.positions + offset)
        read = rows_of(
            self.media if self.media is not None else {}, list(dict.fromkeys(placeholders(tokens)[1].tolist()))
        )
        grids = {d: (int(x.shape[0]), int(x.shape[1]), int(x.shape[2])) for d, x in read.items() if x.ndim == 4}
        if not r.is_branch:
            found = [places(t, r.start, offset, grids) for t in tokens]
            if cache is not None:
                cache.offsets.update(found[0][2])
            return Places(r.start, axes=mx.array(np.concatenate([f[0] for f in found], axis=1).astype(np.int32)))
        assert r.valid is not None
        head, offset, changes = places(tokens[0, : r.head], r.start, offset, grids)
        if cache is not None:
            cache.offsets.update(changes)
        parts, at = [head], r.head
        for n in np.array(r.valid).sum(axis=1).tolist():  # each tail from the head's end
            parts.append(places(tokens[0, at : at + n], r.start + r.head, offset, grids)[0])
            at += n
        return Places(r.start, axes=mx.array(np.concatenate(parts, axis=1).astype(np.int32)))

    def _mlp(self, h: mx.array, layer: Mapping[str, Any]) -> mx.array:
        dense = self.ops.mm(self.ops.glu_mm(h, layer["gate_up"], "silu"), layer["down"])
        if "router" not in layer:
            return dense
        return self._moe(h, layer) + mx.sigmoid(h @ layer["shared_gate"].T) * dense  # the shared expert, gated

    def _moe(self, h: mx.array, layer: Mapping[str, Any]) -> mx.array:
        """The routed experts, as transformers' Qwen3.5-MoE computes them: router logits in the compute dtype, their
        softmax in fp32, the top_k renormalized and rounded to the compute dtype, each chosen expert's SwiGLU times
        its weight, summed (here in fp32, rounded once)."""
        c, lead, width = self.config, h.shape[:-1], h.shape[-1]
        top, order, experts, pairs = self.ops.route(h.reshape(-1, width), layer["router"], c.top_k)
        g = self.ops.glu(self.ops.expert_mm(pairs, layer["experts_gate_up"], experts), "silu")
        y = self.ops.expert_mm(g, layer["experts_down"], experts)[:, 0]
        return self.ops.combine(y, order, top).reshape(*lead, width)

    def _attention(
        self,
        h: mx.array,
        layer: Mapping[str, mx.array],
        block: Attention | None,
        r: Rows,
        read: Read | None,
        at: Places,
    ) -> mx.array:
        c = self.config
        b, n, qw, kw = h.shape[0], r.n, c.heads * c.head_dim * 2, c.kv_heads * c.head_dim
        proj = self.ops.mm(h, layer["qkv"])
        d = c.head_dim
        gate = proj[..., :qw].reshape(b, n, c.heads, 2 * d)[..., d:]
        v = proj[..., qw + kw :].reshape(b, n, c.kv_heads, d)
        if at.axes is None:  # one position per row: the fused norm and RoPE, when the profile has it
            q = self.ops.norm_rope(proj, 0, 2 * d, c.heads, d, layer["q_norm"], c.eps, at.start, at.positions, c.rope)
            k = self.ops.norm_rope(proj, qw, d, c.kv_heads, d, layer["k_norm"], c.eps, at.start, at.positions, c.rope)
        else:
            q = self._mrope(proj, 0, 2 * d, c.heads, layer["q_norm"], at.axes)
            k = self._mrope(proj, qw, d, c.kv_heads, layer["k_norm"], at.axes)
        o = attend(q, k, v.transpose(0, 2, 1, 3), block, r, read, c.causal, c.head_dim**-0.5, ops=self.ops)
        if read is not None:
            if read.size == 0:
                return mx.zeros((b, 0, c.hidden), dtype=self.dtype)
            gate = read.take(gate)
        o = o * mx.sigmoid(gate)
        return self.ops.mm(o.reshape(b, o.shape[1], c.heads * c.head_dim), layer["o"])

    def _mrope(self, proj: mx.array, offset: int, stride: int, heads: int, w: mx.array, axes: mx.array) -> mx.array:
        """Queries or keys [sequences, heads, rows, head_dim] of a pass with image rows (`ops.norm_rope`'s layout):
        the per-head RMSNorm, then RoPE at each row's t, h and w positions in turn (MLX's, a row a batch of its
        own), each frequency pair kept from its axis's turn (`mrope_axes`). A text row turns by the same position
        three times, so it gets exactly the RoPE of a pass without images."""
        c = self.config
        b, n, d = proj.shape[0], proj.shape[1], c.head_dim
        x = self.ops.norm(proj[..., offset : offset + heads * stride].reshape(b, n, heads, stride)[..., :d], w, c.eps)
        flat = x.transpose(2, 0, 1, 3).reshape(1, heads, b * n, d)
        by_t, by_h, by_w = (c.rope(flat, 0, axes[axis]) for axis in range(3))
        turned = mx.where(self._axes == 1, by_h, mx.where(self._axes == 2, by_w, by_t))
        return turned.reshape(heads, b, n, d).transpose(1, 0, 2, 3)

    def _delta_net(
        self,
        h: mx.array,
        layer: Mapping[str, mx.array],
        block: Recurrent | None,
        r: Rows,
        read: Read | None,
        snapshots: list[Snapshots],
    ) -> mx.array:
        proj = self.ops.mm(h, layer["in"])  # [sequences, n, conv | z | b | a]
        y = self._recurrent(
            proj, block, r, snapshots, lambda rows, held, marks: self._recurrence(rows, layer, held, marks)
        )
        if read is not None:
            y = read.take(y)
        return self.ops.mm(y, layer["out"])

    def _recurrence(
        self,
        proj: mx.array,
        layer: Mapping[str, mx.array],
        held: Held,
        marks: Sequence[int],
    ) -> tuple[mx.array, Held, Snapshots]:
        """A Gated DeltaNet over [sequences, length] rows continuing (state, window), broadcast to every sequence:
        its output before out_proj, the state and window after the last row, and the state and window at each
        mark (the recurrence runs piecewise there). Padding after a sequence's last real row changes only the
        rows after it and the final state, which a padded batch doesn't keep."""
        c, (state, window) = self.config, held
        conv, length = c.conv_dim, proj.shape[1]
        k1 = c.conv_kernel - 1
        if window is None:
            window = mx.zeros((1, k1, conv), dtype=proj.dtype)
        inputs = proj[..., :conv]  # the convolution's inputs, for its window at marks and at the end
        mixed = self.ops.causal_conv(proj, window, layer["conv"], silu=True)  # over proj's first `conv` columns
        q, k, v, beta, gamma = self.ops.delta_inputs(
            mixed, proj, layer["neg_a"], layer["dt_bias"], c.k_heads, c.k_dim, c.v_heads, c.v_dim, c.k_dim**-0.5
        )
        kept: Snapshots = {}
        if marks:
            pieces, bounds = [], [0, *marks, length]
            for lo, hi in itertools.pairwise(bounds):
                o_piece, state = self.ops.gated_delta(
                    q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], gamma[:, lo:hi], beta[:, lo:hi], state
                )
                pieces.append(o_piece)
                if hi < length:
                    kept[hi] = (state, window_before(window, inputs, k1, hi))
            o = mx.concatenate(pieces, axis=1)
        else:
            o, state = self.ops.gated_delta(q, k, v, gamma, beta, state)
        y = self.ops.gated_norm(o, proj, conv, layer["norm"], c.eps)  # z: proj's columns conv..
        return y, (state, window_before(window, inputs, k1, length)), kept


def load(
    model_dir: str,
    config: Mapping[str, Any],
    *,
    causal: bool = True,
    block: bool = False,
    recurrence: Recurrence | None = None,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    bits: int | None = None,
) -> Qwen35:
    """The Qwen3.5 text model of a downloaded checkpoint: `causal=False` for full-attention layers that read every
    position, `block` for block attention (Backbone.block), and `recurrence` for blocks of layers read more than once
    a pass (`Recurrence`), which only a model read both ways takes: a causal pass continues a cache, which a block read
    again would overwrite."""
    settings = Config.from_hf(config, causal=causal, block=block)
    if recurrence is not None:
        if settings.causal or settings.block:
            raise UnsupportedModel("recurrent depth reads a model both ways only (causal=False, no block attention)")
        last = len(settings.layer_types) - 2
        if any(not 0 <= a <= b <= last for a, b in recurrence.blocks):
            raise UnsupportedModel(f"recurrent depth {recurrence}: blocks must lie within layers 0 to {last}")
    model = Qwen35(settings, weights or Weights.from_dir(model_dir), dtype, bits)
    model.recurrence = recurrence
    return model
