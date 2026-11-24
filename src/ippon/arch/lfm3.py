"""LFM3 (Liquid AI), as Liquid's d1-2B and d1-Omni-2B read it: LFM2's gated short convolutions with attention layers
of two kinds between them, under d1's "set" attention.

The numbers follow Liquid's modeling_lfm3.py:

- RMSNorm is x / rms(x) times the checkpoint's weight (not 1 + w), here in fp32 and rounded once; every operator's
  and every MLP's output is normed again before the residual (sandwich norms);
- a convolution layer is LFM2's: B, C and x, a causal depthwise convolution of `conv_L_cache` taps over B * x, C
  times that;
- an attention layer is GQA with a per-head RMSNorm on queries and keys, and a sigmoid gate per head on its output
  (`gate_proj`, from the layer's normed input) before the output projection. A `sliding_attention` layer reads the
  `sliding_window` keys ending at each token, causally, with RoPE over the whole head; a `full_attention` layer has
  its own RoPE, which rotates the first `partial_rotary_factor` of the angle pairs at the whole head's frequencies and
  leaves the rest (the "zero tail", `rope.proportional_freqs`);
- the MLP is SwiGLU, w2(silu(w1 x) * w3 x).

Under "set" attention a prompt is a state followed by questions (`plan.Prompt.groups`): on the full attention layers
a state token reads every token of its prompt and a question token the state and all of its own question, never
another; the convolutions and the sliding layers stay causal over the whole prompt. Nothing a prompt reads can be
kept for another, so every prompt is read whole, as an encoder's is.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..checkpoint import EMBEDDING
from ..errors import UnsupportedModel
from ..quant import Matrix, Table, arrays, concat
from ..weights import Weights
from .cache import HybridCache
from .common import Encoder, Read, Rows, Snapshots, attend, embed, finish_layer
from .rope import RoPE, proportional_freqs

F32 = mx.float32
FAMILIES = ("lfm3",)  # its config.json model_types
KINDS = ("conv", "full_attention", "sliding_attention")


@dataclass(frozen=True)
class Config:
    hidden: int
    layer_types: tuple[str, ...]  # one of KINDS per layer
    heads: int
    kv_heads: int
    head_dim: int
    ropes: dict[str, RoPE]  # per attention kind
    window: int  # keys before a query a sliding layer reads (transformers' sliding_window less the query's own)
    eps: float
    conv_kernel: int

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> Config:
        """From a Hugging Face config.json, or its "text_config" for a multimodal checkpoint."""
        c = config.get("text_config", config)
        if c.get("model_type") not in FAMILIES or "conv_L_cache" not in c or "layer_types" not in c:
            raise UnsupportedModel(f"not an LFM3 text model: model_type {c.get('model_type')!r}")
        if c.get("conv_bias") or not c.get("use_sandwich_norm") or not c.get("tie_word_embeddings", True):
            raise UnsupportedModel("an LFM3 variant this implementation does not cover (conv bias, norms, head)")
        if any(t not in KINDS for t in c["layer_types"]):
            raise UnsupportedModel(f"unknown LFM3 layer types {sorted(set(c['layer_types']))}")
        heads = int(c["num_attention_heads"])
        head_dim = int(c.get("head_dim") or c["hidden_size"] // heads)
        ropes = {}
        for kind in ("full_attention", "sliding_attention"):
            p = dict(c["rope_parameters"][kind])
            if p.get("rope_type", "default") != "default":
                raise UnsupportedModel(f"LFM3 {kind} RoPE of type {p['rope_type']!r} is not supported")
            theta = float(p["rope_theta"])
            whole = float(p.get("partial_rotary_factor", 1.0)) == 1.0
            ropes[kind] = (
                RoPE(head_dim, theta) if whole else RoPE(head_dim, theta, proportional_freqs(head_dim, theta, p))
            )
        return cls(
            hidden=int(c["hidden_size"]),
            layer_types=tuple(c["layer_types"]),
            heads=heads,
            kv_heads=int(c["num_key_value_heads"]),
            head_dim=head_dim,
            ropes=ropes,
            window=int(c["sliding_window"]) - 1,
            eps=float(c.get("norm_eps", 1e-5)),
            conv_kernel=int(c["conv_L_cache"]),
        )


class LFM3(Encoder):
    """A loaded LFM3 text model under set attention (see the module docstring): every prompt read whole."""

    config: Config

    # each layer's weight matrices that inputs multiply (quant.mm): what quantization converts
    linears = ("gate_up", "down", "qkvg", "o", "in", "out")
    gated_mlp = ("gate_up", "silu")  # the [gate; up] projection and its GLU's activation

    def __init__(
        self,
        config: Config,
        weights: Weights,
        dtype: mx.Dtype = mx.bfloat16,
        root: str | None = None,
        bits: int | None = None,
    ):
        self.config, self.dtype = config, dtype
        c = config
        root = weights.text_root() if root is None else root

        def n(name: str) -> mx.array:  # a norm weight, in fp32
            return weights.get(root + name, F32)

        def m(name: str) -> Matrix:  # a weight matrix, quantized as stored
            return weights.matrix(root + name, dtype)

        self.attention_share = sum(t != "conv" for t in c.layer_types) / len(c.layer_types)
        self.embed: Table = weights.table(root + EMBEDDING, dtype)
        self.final_norm = n("embedding_norm.weight")
        self.layers: list[dict[str, Any]] = []  # matrices may be QuantizedMatrix
        for i, kind in enumerate(c.layer_types):
            p = f"layers.{i}."
            if (root + p + ("conv.in_proj.weight" if kind == "conv" else "self_attn.q_proj.weight")) not in weights:
                raise UnsupportedModel(f"layer {i}: the config says {kind!r}, the weights do not")
            layer: dict[str, Any] = {
                "in_norm": n(p + "operator_norm.weight"),
                "mixer_norm": n(p + "post_operator_norm.weight"),
                "post_norm": n(p + "ffn_norm.weight"),
                "mlp_norm": n(p + "post_ffn_norm.weight"),
                "gate_up": concat([m(p + "feed_forward.w1.weight"), m(p + "feed_forward.w3.weight")]),
                "down": m(p + "feed_forward.w2.weight"),
            }
            if kind == "conv":
                layer |= {
                    "in": m(p + "conv.in_proj.weight"),
                    # [channels, 1, taps] from Hugging Face, [channels, taps, 1] from MLX: the same order of taps
                    "conv": weights.get(root + p + "conv.conv.weight", F32).reshape(c.hidden, c.conv_kernel),
                    "out": m(p + "conv.out_proj.weight"),
                }
            else:  # queries, keys, values and the heads' gates in one projection, as Liquid fuses them
                a = p + "self_attn."
                layer |= {
                    "qkvg": concat([m(a + f"{x}_proj.weight") for x in ("q", "k", "v", "gate")]),
                    "o": m(a + "o_proj.weight"),
                    "q_norm": n(a + "q_norm.weight"),
                    "k_norm": n(a + "k_norm.weight"),
                }
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(arrays([self.embed, self.final_norm]))

    # -- the Backbone contract ----------------------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.config.hidden

    @property
    def output_matrix(self) -> Table:
        return self.embed  # tied

    # -- the layers ---------------------------------------------------------------------------------------------

    @property
    def layer_kinds(self) -> Sequence[str]:
        return self.config.layer_types

    def mask_keys(self, r: Rows) -> Sequence[tuple[bool, int | None]]:
        return [(False, None), (True, self.config.window)]

    def _forward(self, ids: mx.array, cache: HybridCache | None, r: Rows, keep: Read) -> mx.array:
        """Final-normed hidden states [sequences, rows, hidden] of the `keep` rows."""
        x = embed(self.embed, ids, self.media)
        return self._decoder(x, None, r, keep, self.final_norm, self.config.eps)

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
        """The convolution or attention, its output normed (row by row, so the read rows' own)."""
        if kind == "conv":
            y = self.ops.gated_conv(self.ops.mm(h, layer["in"]), None, layer["conv"])
            mixed = self.ops.mm(y if read is None else read.take(y), layer["out"])
        else:
            assert isinstance(kind, str)
            mixed = self._attention(h, layer, kind, r, read)
        return self.ops.norm(mixed, layer["mixer_norm"], self.config.eps)

    def _feed_forward(self, kind: Hashable, layer: dict[str, Any], h: mx.array) -> mx.array:
        ops = self.ops
        return ops.norm(
            ops.mm(ops.glu_mm(h, layer["gate_up"], "silu"), layer["down"]), layer["mlp_norm"], self.config.eps
        )

    def _attention(self, h: mx.array, layer: Mapping[str, mx.array], kind: str, r: Rows, read: Read | None) -> mx.array:
        c = self.config
        b, n, qw, kw = h.shape[0], r.n, c.heads * c.head_dim, c.kv_heads * c.head_dim
        proj = self.ops.mm(h, layer["qkvg"])
        d, rope, at = c.head_dim, c.ropes[kind], (r.start, r.positions)
        q = self.ops.norm_rope(proj, 0, d, c.heads, d, layer["q_norm"], c.eps, *at, rope)
        k = self.ops.norm_rope(proj, qw, d, c.kv_heads, d, layer["k_norm"], c.eps, *at, rope)
        v = proj[..., qw + kw : qw + 2 * kw].reshape(b, n, c.kv_heads, d)
        sliding = kind == "sliding_attention"
        window = c.window if sliding else None
        o = attend(q, k, v.transpose(0, 2, 1, 3), None, r, read, sliding, d**-0.5, window, ops=self.ops)
        gate = proj[..., qw + 2 * kw :]
        if read is not None:
            gate = read.take(gate)
        gated = (o.astype(F32) * mx.sigmoid(gate.astype(F32))[..., None]).astype(o.dtype)
        return self.ops.mm(gated.reshape(b, o.shape[1], qw), layer["o"])


def load(
    model_dir: str,
    config: Mapping[str, Any],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    root: str | None = None,
    bits: int | None = None,
) -> LFM3:
    """The LFM3 text model of a downloaded checkpoint, read under set attention."""
    return LFM3(Config.from_hf(config), weights or Weights.from_dir(model_dir), dtype, root, bits)
