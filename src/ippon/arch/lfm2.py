"""LFM2 (Liquid AI): gated short convolutions with a GQA attention layer every few layers. The speech encoder of
Liquid's d1-Omni-2B is one.

The numbers follow Hugging Face's LFM2 and Liquid's own d1 code:

- RMSNorm is x / rms(x) times the checkpoint's weight (not 1 + w); here in fp32 and rounded once (Hugging Face
  rounds before weighting);
- a convolution layer projects to B, C and x, multiplies B by x, runs a causal depthwise convolution of
  `conv_L_cache` taps over that, multiplies by C and projects out;
- an attention layer is GQA with a per-head RMSNorm on queries and keys and rotate_half RoPE over the whole head;
  it has no output gate;
- the MLP is SwiGLU, w2(silu(w1 x) * w3 x), here in fp32 and rounded once; its width is read from the weights.

d1-Omni-2B's speech encoder is a causal LFM2 over the embeddings its downsampler gives (no token embedding) whose
attention reads at most `window` positions back (`read_embeddings`).
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
from .cache import Attention, HybridCache, Recurrent
from .common import Held, Hybrid, Read, Rows, Snapshots, attend, embed, finish_layer, window_before
from .rope import RoPE

F32 = mx.float32
FAMILIES = ("lfm2",)  # its config.json model_types


@dataclass(frozen=True)
class Config:
    hidden: int
    layer_types: tuple[str, ...]  # "conv" or "full_attention", per layer
    heads: int
    kv_heads: int
    head_dim: int
    rope: RoPE
    eps: float
    conv_kernel: int
    window: int | None = None  # keys before a query its attention reads at most; None: all

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> Config:
        """From a Hugging Face config.json, or its "text_config" for a multimodal checkpoint."""
        c = config.get("text_config", config)
        if c.get("model_type", "lfm2") not in FAMILIES or "conv_L_cache" not in c or "layer_types" not in c:
            raise UnsupportedModel(f"not an LFM2 text model: model_type {c.get('model_type')!r}")
        if c.get("conv_bias") or c.get("block_use_swiglu", True) is not True:
            raise UnsupportedModel("an LFM2 variant this implementation does not cover (conv bias, MLP)")
        if any(t not in ("conv", "full_attention") for t in c["layer_types"]):
            raise UnsupportedModel(f"unknown LFM2 layer types {sorted(set(c['layer_types']))}")
        kernel = int(c["conv_L_cache"])
        heads = int(c.get("num_attention_heads") or c["num_heads"])
        head_dim = int(c.get("head_dim") or c["hidden_size"] // heads)
        rope = RoPE.from_hf(
            head_dim,
            c.get("rope_parameters"),
            theta=float(c.get("rope_theta", 1_000_000.0)),
            max_positions=int(c.get("max_position_embeddings") or 0),
        )
        return cls(
            hidden=int(c["hidden_size"]),
            layer_types=tuple(c["layer_types"]),
            heads=heads,
            kv_heads=int(c["num_key_value_heads"]),
            head_dim=head_dim,
            rope=rope,
            eps=float(c.get("norm_eps", 1e-5)),
            conv_kernel=kernel,
        )


class LFM2(Hybrid):
    """A loaded LFM2 text model (see the module docstring)."""

    config: Config

    # each layer's weight matrices that inputs multiply (quant.mm): what quantization converts
    linears = ("gate_up", "down", "qkv", "o", "in", "out")
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

        # an encoder over embeddings (d1-Omni-2B's speech encoder) has none
        self.embed: Table | None = weights.table(root + EMBEDDING, dtype) if root + EMBEDDING in weights else None
        self.final_norm = n("embedding_norm.weight")
        self.layers: list[dict[str, Any]] = []  # matrices may be QuantizedMatrix
        for i, kind in enumerate(c.layer_types):
            p = f"layers.{i}."
            if (root + p + ("self_attn.q_proj.weight" if kind == "full_attention" else "conv.in_proj.weight")) not in (
                weights
            ):
                raise UnsupportedModel(f"layer {i}: the config says {kind!r}, the weights do not")
            layer: dict[str, Any] = {
                "in_norm": n(p + "operator_norm.weight"),
                "post_norm": n(p + "ffn_norm.weight"),
                "gate_up": concat([m(p + "feed_forward.w1.weight"), m(p + "feed_forward.w3.weight")]),
                "down": m(p + "feed_forward.w2.weight"),
            }
            if kind == "full_attention":
                a = p + "self_attn."
                layer |= {
                    "qkv": concat([m(a + f"{x}_proj.weight") for x in "qkv"]),
                    "o": m(a + "out_proj.weight"),
                    "q_norm": n(a + "q_layernorm.weight"),
                    "k_norm": n(a + "k_layernorm.weight"),
                }
            else:
                layer |= {
                    "in": m(p + "conv.in_proj.weight"),
                    # [channels, 1, taps] from Hugging Face, [channels, taps, 1] from MLX: the same order of taps
                    "conv": weights.get(root + p + "conv.conv.weight", F32).reshape(c.hidden, c.conv_kernel),
                    "out": m(p + "conv.out_proj.weight"),
                }
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(arrays([self.embed, self.final_norm]))

    def read_embeddings(self, x: mx.array) -> mx.array:
        """The final-normed hidden states [n, hidden] of embeddings x [n, hidden], read causally in one pass from
        nothing."""
        n = x.shape[0]
        return self._decoder(x[None], None, Rows.prefill(n, 0), Read(None, 0, n), self.final_norm, self.config.eps)[0]

    # -- the Backbone contract ----------------------------------------------------------------------------------

    @property
    def output_matrix(self) -> Table:
        if self.embed is None:
            raise UnsupportedModel("an LFM2 over embeddings has no output matrix")
        return self.embed  # tied

    # -- the layers ---------------------------------------------------------------------------------------------

    @property
    def layer_kinds(self) -> Sequence[str]:
        return self.config.layer_types

    def mask_keys(self, r: Rows) -> Sequence[tuple[bool, int | None]]:
        return [(True, self.config.window)]

    def _forward(self, ids: mx.array, cache: HybridCache | None, r: Rows, keep: Read) -> mx.array:
        """Final-normed hidden states [sequences, rows, hidden] of the `keep` rows. Each
        layer's MLP output is added to the residual together with the next norm (`ops.add_norm`)."""
        if self.embed is None:
            raise UnsupportedModel("an LFM2 over embeddings reads no tokens")
        x = embed(self.embed, ids, self.media)
        return self._decoder(x, cache, r, keep, self.final_norm, self.config.eps)

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
        if kind == "full_attention":
            assert block is None or isinstance(block, Attention)
            return self._attention(h, layer, block, r, read)
        assert block is None or isinstance(block, Recurrent)
        return self._conv(h, layer, block, r, read, kept)

    def _feed_forward(self, kind: Hashable, layer: dict[str, Any], h: mx.array) -> mx.array:
        return self.ops.mm(self.ops.glu_mm(h, layer["gate_up"], "silu"), layer["down"])

    def _attention(
        self, h: mx.array, layer: Mapping[str, mx.array], block: Attention | None, r: Rows, read: Read | None
    ) -> mx.array:
        c = self.config
        b, n, qw, kw = h.shape[0], r.n, c.heads * c.head_dim, c.kv_heads * c.head_dim
        proj = self.ops.mm(h, layer["qkv"])
        d, at = c.head_dim, (r.start, r.positions)
        q = self.ops.norm_rope(proj, 0, d, c.heads, d, layer["q_norm"], c.eps, *at, c.rope)
        k = self.ops.norm_rope(proj, qw, d, c.kv_heads, d, layer["k_norm"], c.eps, *at, c.rope)
        v = proj[..., qw + kw :].reshape(b, n, c.kv_heads, c.head_dim)
        scale = c.head_dim**-0.5 * c.rope.attention**2  # YaRN's factor on both queries and keys
        o = attend(q, k, v.transpose(0, 2, 1, 3), block, r, read, True, scale, c.window, ops=self.ops)
        return self.ops.mm(o.reshape(b, o.shape[1], c.heads * c.head_dim), layer["o"])

    def _conv(
        self,
        h: mx.array,
        layer: Mapping[str, mx.array],
        block: Recurrent | None,
        r: Rows,
        read: Read | None,
        snapshots: list[Snapshots],
    ) -> mx.array:
        proj = self.ops.mm(h, layer["in"])  # [sequences, n, B | C | x]
        y = self._recurrent(
            proj, block, r, snapshots, lambda rows, held, marks: self._short_conv(rows, layer, held, marks)
        )
        if read is not None:
            y = read.take(y)
        return self.ops.mm(y, layer["out"])

    def _short_conv(
        self, proj: mx.array, layer: Mapping[str, mx.array], held: Held, marks: Sequence[int]
    ) -> tuple[mx.array, Held, Snapshots]:
        """C * conv(B * x) over [sequences, length] rows continuing the held window (broadcast to every sequence),
        before out_proj; the window after the last row, and the window at each mark."""
        c, d, window = self.config, self.config.hidden, held[1]
        k1 = c.conv_kernel - 1
        y = self.ops.gated_conv(proj, window, layer["conv"])

        def before(at: int) -> mx.array:  # the convolution's inputs B * x before position `at`
            rows = proj[:, max(0, at - k1) : at]
            return window_before(window, rows[..., :d] * rows[..., 2 * d :], k1, min(at, k1))

        return y, (None, before(proj.shape[1])), {m: (None, before(m)) for m in marks}


def load(
    model_dir: str,
    config: Mapping[str, Any],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    root: str | None = None,
    bits: int | None = None,
) -> LFM2:
    """The LFM2 text model of a downloaded checkpoint; `root` names its tensors' prefix when it isn't a standard
    one."""
    return LFM2(Config.from_hf(config), weights or Weights.from_dir(model_dir), dtype, root, bits)
