"""Gemma 3 text decoders (270M, 1B and the text path of the larger ones), a base of small local fine-tunes. The
numbers follow Hugging Face's modeling_gemma3:

- the embedding is scaled by sqrt(hidden) (the scale rounded to the weights' dtype, as transformers does);
- RMSNorm multiplies by 1 + w in fp32 and rounds once; every layer has four: before attention, on its output,
  before the MLP and on its output (x = x + norm(attention(norm(x))), the same for the MLP);
- attention is GQA with a per-head RMSNorm on queries and keys, scaled by query_pre_attn_scalar ** -0.5, with
  rotate_half RoPE of the layer type's base (`rope_parameters`: sliding layers 10,000, full ones 1,000,000 and
  their scaling); a sliding layer's query sees the `sliding_window` keys ending at itself;
- the MLP is GeGLU with GELU's tanh approximation, down(gelu_tanh(gate x) * up x), in fp32, rounded once.

Logit soft-capping (Gemma 2) and the bidirectional variant (EmbeddingGemma) are refused.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..checkpoint import EMBEDDING
from ..errors import UnsupportedModel
from ..quant import Matrix, Table, arrays, concat, rows
from ..weights import Weights
from .cache import Attention, HybridCache
from .common import AttentionDecoder, Read, Rows, Snapshots, attend, finish_layer, output_head
from .rope import RoPE

F32 = mx.float32
FAMILIES = ("gemma3_text", "gemma3")


@dataclass(frozen=True)
class Config:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    eps: float
    query_scale: float
    window: int  # keys before a query a sliding layer reads (transformers' sliding_window less the query's own)
    layer_types: tuple[str, ...]  # "sliding_attention" or "full_attention"
    ropes: Mapping[str, RoPE]  # by layer type
    tied: bool

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> Config:
        c = config.get("text_config", config)
        if c.get("model_type") not in FAMILIES:
            raise UnsupportedModel(f"not a Gemma 3 text model: model_type {c.get('model_type')!r}")
        if c.get("attn_logit_softcapping") or c.get("final_logit_softcapping") or c.get("use_bidirectional_attention"):
            raise UnsupportedModel("a Gemma variant with soft-capping or bidirectional attention is not supported")
        if c.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise UnsupportedModel(f"hidden_activation {c.get('hidden_activation')!r} is not supported")
        n = int(c["num_hidden_layers"])
        pattern = int(c.get("sliding_window_pattern", 6))
        types = tuple(
            c.get("layer_types") or ["sliding_attention" if (i + 1) % pattern else "full_attention" for i in range(n)]
        )
        params = dict(c.get("rope_parameters") or {})
        full = dict(params.get("full_attention") or {})
        sliding = dict(params.get("sliding_attention") or {})
        if c.get("rope_scaling"):
            full |= dict(c["rope_scaling"])
        full.setdefault("rope_theta", c.get("rope_theta", 1_000_000.0))
        sliding.setdefault("rope_theta", c.get("rope_local_base_freq", 10_000.0))
        heads = int(c["num_attention_heads"])
        head_dim = int(c.get("head_dim") or c["hidden_size"] // heads)
        positions = int(c.get("max_position_embeddings") or 0)
        ropes = {
            kind: RoPE.from_hf(head_dim, p, max_positions=positions)
            for kind, p in (("full_attention", full), ("sliding_attention", sliding))
        }
        return cls(
            hidden=int(c["hidden_size"]),
            layers=n,
            heads=heads,
            kv_heads=int(c.get("num_key_value_heads") or heads),
            head_dim=head_dim,
            eps=float(c.get("rms_norm_eps", 1e-6)),
            query_scale=float(c.get("query_pre_attn_scalar", head_dim)) ** -0.5,
            window=int(c.get("sliding_window") or 4096) - 1,
            layer_types=types,
            ropes=ropes,
            tied=bool(c.get("tie_word_embeddings", True)),
        )


class Gemma3(AttentionDecoder):
    """A loaded Gemma 3 text decoder (see the module docstring); the cache and passes are the dense decoders'."""

    config: Config

    linears = ("gate_up", "down", "qkv", "o")
    gated_mlp = ("gate_up", "gelu_tanh")

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

        def n(name: str) -> mx.array:  # a norm weight, as 1 + w in fp32
            return 1.0 + weights.get(root + name, F32)

        def m(name: str) -> Matrix:
            return weights.matrix(root + name, dtype)

        self.embed: Table = weights.table(root + EMBEDDING, dtype)
        # rounded to the weights' dtype, as transformers rounds it (exact in fp32)
        self.embed_scale = mx.array(math.sqrt(c.hidden), dtype=dtype).astype(F32)
        self.final_norm = n("norm.weight")
        self.lm_head = output_head(weights, root, c.tied, dtype)
        self.layers: list[dict[str, Any]] = []
        for i in range(c.layers):
            p, a = f"layers.{i}.", f"layers.{i}.self_attn."
            layer: dict[str, Any] = {
                "in_norm": n(p + "input_layernorm.weight"),
                "attn_out_norm": n(p + "post_attention_layernorm.weight"),
                "mlp_norm": n(p + "pre_feedforward_layernorm.weight"),
                "mlp_out_norm": n(p + "post_feedforward_layernorm.weight"),
                "qkv": concat([m(a + f"{x}_proj.weight") for x in "qkv"]),
                "o": m(a + "o_proj.weight"),
                "q_norm": n(a + "q_norm.weight"),
                "k_norm": n(a + "k_norm.weight"),
                "gate_up": concat([m(p + "mlp.gate_proj.weight"), m(p + "mlp.up_proj.weight")]),
                "down": m(p + "mlp.down_proj.weight"),
            }
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(arrays([self.embed, self.final_norm, self.embed_scale, self.lm_head]))

    @property
    def max_tokens(self) -> int | None:
        """No limit: its sliding layers attend within their window, as Gemma's do."""
        return None

    def _forward(self, ids: mx.array, cache: HybridCache | None, r: Rows, keep: Read) -> mx.array:
        x = (rows(self.embed, ids).astype(F32) * self.embed_scale).astype(self.dtype)
        return self._decoder(x, cache, r, keep, self.final_norm, self.config.eps, post="mlp_norm")

    @property
    def layer_kinds(self) -> Sequence[str]:
        return self.config.layer_types

    def mask_keys(self, r: Rows) -> Sequence[tuple[bool, int | None]]:
        return [(True, None), (True, self.config.window)]

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
        """Attention, its output normed (Gemma's sandwich norm: row by row, so the read rows' own)."""
        assert block is None or isinstance(block, Attention)
        assert isinstance(kind, str)
        attention = self._attention(h, layer, kind, block, r, read)
        return self.ops.norm(attention, layer["attn_out_norm"], self.config.eps)

    def _feed_forward(self, kind: Hashable, layer: dict[str, Any], h: mx.array) -> mx.array:
        ops = self.ops
        return ops.norm(
            ops.mm(ops.glu_mm(h, layer["gate_up"], "gelu_tanh"), layer["down"]), layer["mlp_out_norm"], self.config.eps
        )

    def _attention(
        self, h: mx.array, layer: Mapping[str, mx.array], kind: str, block: Attention | None, r: Rows, read: Read | None
    ) -> mx.array:
        c = self.config
        b, n, qw, kw = h.shape[0], r.n, c.heads * c.head_dim, c.kv_heads * c.head_dim
        proj = self.ops.mm(h, layer["qkv"])
        v = proj[..., qw + kw :].reshape(b, n, c.kv_heads, c.head_dim)
        d, rope, at = c.head_dim, c.ropes[kind], (r.start, r.positions)
        q = self.ops.norm_rope(proj, 0, d, c.heads, d, layer["q_norm"], c.eps, *at, rope)
        k = self.ops.norm_rope(proj, qw, d, c.kv_heads, d, layer["k_norm"], c.eps, *at, rope)
        window = c.window if kind == "sliding_attention" else None
        o = attend(
            q, k, v.transpose(0, 2, 1, 3), block, r, read, True, c.query_scale * rope.attention**2, window, ops=self.ops
        )
        out: mx.array = self.ops.mm(o.reshape(b, o.shape[1], qw), layer["o"])
        return out


def load(
    model_dir: str,
    config: Mapping[str, Any],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    root: str | None = None,
    bits: int | None = None,
) -> Gemma3:
    """The Gemma 3 text decoder of a downloaded checkpoint (a causal LM's, or a sequence classifier's)."""
    return Gemma3(Config.from_hf(config), weights or Weights.from_dir(model_dir), dtype, root, bits)
