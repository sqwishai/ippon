"""Dense decoders: Llama, Mistral, Qwen2 (and 2.5) and Qwen3, the bases most people fine-tune their own decision
models from. One module covers them, as Hugging Face's code differs between them only in a few places:

- RMSNorm is x / rms(x) times the checkpoint's weight (not 1 + w); here in fp32 and rounded once (Hugging Face
  rounds before weighting); `norm_groups` takes its statistics over equal groups of the channels instead, for
  models that normalize so;
- attention is GQA with rotate_half RoPE over the head (Llama's "llama3", "linear" and YaRN frequency scaling
  included; YaRN's attention factor, which scales the rotated queries and keys alike, as its square on the scores);
  Qwen2's queries, keys and values have biases, Qwen3's queries and keys a per-head RMSNorm before RoPE, and
  Llama's `attention_bias` gives every projection one;
- the MLP is SwiGLU, down(silu(gate x) * up x), here in fp32 and rounded once;
- the output matrix is `lm_head`, or the embedding when tied; a classifier's `score` belongs to its format.

Every layer is attention, so a kept prefix holds only keys and values. A Mistral or Qwen2 sliding window is
honoured by refusing prompts longer than the window (the model sees nothing beyond it, which full attention
would).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..checkpoint import EMBEDDING
from ..errors import ContextTooLong, UnsupportedModel
from ..quant import Matrix, Table, arrays, concat, rows
from ..weights import Weights
from .cache import Attention, HybridCache
from .common import AttentionDecoder, Read, Rows, Snapshots, attend, finish_layer, output_head
from .rope import RoPE

F32 = mx.float32
FAMILIES = ("llama", "mistral", "qwen2", "qwen3")


@dataclass(frozen=True)
class Config:
    family: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    eps: float
    qkv_bias: bool  # biases on the query, key and value projections
    o_bias: bool  # and on the output projection
    qk_norm: bool  # a per-head RMSNorm on queries and keys (Qwen3)
    tied: bool  # the output matrix is the embedding
    rope: RoPE  # over the whole head
    window: int | None = None  # a sliding window over the keys
    norm_groups: int = 1  # RMSNorm statistics over this many equal groups of the channels

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> Config:
        """From a Hugging Face config.json (or its "text_config")."""
        c = config.get("text_config", config)
        family = c.get("model_type")
        if family not in FAMILIES:
            raise UnsupportedModel(f"not a dense decoder this module covers ({', '.join(FAMILIES)}): {family!r}")
        if c.get("mlp_bias") or c.get("hidden_act", "silu") != "silu":
            raise UnsupportedModel(f"a {family} variant this implementation does not cover (MLP bias or activation)")
        heads = int(c["num_attention_heads"])
        head_dim = int(c.get("head_dim") or c["hidden_size"] // heads)
        if c.get("partial_rotary_factor", 1.0) != 1.0:
            raise UnsupportedModel("RoPE over part of the head is not supported for dense decoders")
        rope = RoPE.from_hf(
            head_dim,
            c.get("rope_parameters") or c.get("rope_scaling"),
            theta=float(c.get("rope_theta", 10_000.0)),
            max_positions=int(c.get("max_position_embeddings") or 0),
        )
        window = c.get("sliding_window")
        if family in ("qwen2", "qwen3") and not c.get("use_sliding_window", False):
            window = None  # Qwen's window applies only when use_sliding_window says so
        attention_bias = bool(c.get("attention_bias", False))
        return cls(
            family=family,
            hidden=int(c["hidden_size"]),
            layers=int(c["num_hidden_layers"]),
            heads=heads,
            kv_heads=int(c.get("num_key_value_heads") or heads),
            head_dim=head_dim,
            eps=float(c.get("rms_norm_eps", 1e-6)),
            qkv_bias=family == "qwen2" or attention_bias,
            o_bias=attention_bias,
            qk_norm=family == "qwen3",
            tied=bool(c.get("tie_word_embeddings", False)),
            rope=rope,
            window=int(window) if window else None,
        )


class Dense(AttentionDecoder):
    """A loaded dense decoder (see the module docstring)."""

    config: Config

    linears = ("gate_up", "down", "qkv", "o")  # each layer's weight matrices that inputs multiply (quant.mm)
    gated_mlp = ("gate_up", "silu")

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

        def w(name: str) -> mx.array:
            return weights.get(root + name, dtype)

        def n(name: str) -> mx.array:  # a norm weight, in fp32
            return weights.get(root + name, F32)

        def m(name: str) -> Matrix:  # a weight matrix, quantized as stored
            return weights.matrix(root + name, dtype)

        self.embed: Table = weights.table(root + EMBEDDING, dtype)
        self.final_norm = n("norm.weight")
        self.lm_head = output_head(weights, root, c.tied, dtype)
        self.layers: list[dict[str, Any]] = []
        for i in range(c.layers):
            p, a = f"layers.{i}.", f"layers.{i}.self_attn."
            if root + a + "q_proj.weight" not in weights:
                raise UnsupportedModel(f"layer {i} has no attention projections under {root!r}")
            layer: dict[str, Any] = {
                "in_norm": n(p + "input_layernorm.weight"),
                "post_norm": n(p + "post_attention_layernorm.weight"),
                "qkv": concat([m(a + f"{x}_proj.weight") for x in "qkv"]),
                "o": m(a + "o_proj.weight"),
                "gate_up": concat([m(p + "mlp.gate_proj.weight"), m(p + "mlp.up_proj.weight")]),
                "down": m(p + "mlp.down_proj.weight"),
            }
            if c.qkv_bias:
                layer["qkv_bias"] = mx.concatenate([w(a + f"{x}_proj.bias") for x in "qkv"])
            if c.o_bias:
                layer["o_bias"] = w(a + "o_proj.bias")
            if c.qk_norm:
                layer |= {"q_norm": n(a + "q_norm.weight"), "k_norm": n(a + "k_norm.weight")}
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(arrays([self.embed, self.final_norm] + ([self.lm_head] if self.lm_head is not None else [])))

    # -- the Backbone contract ----------------------------------------------------------------------------------

    @property
    def max_tokens(self) -> int | None:
        """A Mistral or Qwen2 sliding window, which these layers don't implement: longer prompts are refused."""
        return self.config.window

    # -- the layers ---------------------------------------------------------------------------------------------

    def _forward(self, ids: mx.array, cache: HybridCache | None, r: Rows, keep: Read) -> mx.array:
        """Final-normed hidden states [sequences, rows, hidden] of the `keep` rows; each layer's MLP output is added
        to the residual together with the next norm (`ops.add_norm`)."""
        c = self.config
        if c.window is not None and r.start + r.n > c.window:
            raise ContextTooLong(f"a prompt longer than the model's sliding window ({c.window} tokens)")
        return self._decoder(rows(self.embed, ids), cache, r, keep, self.final_norm, c.eps, c.norm_groups)

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
        assert block is None or isinstance(block, Attention)
        return self._attention(h, layer, block, r, read)

    def _feed_forward(self, kind: Hashable, layer: dict[str, Any], h: mx.array) -> mx.array:
        return self.ops.mm(self.ops.glu_mm(h, layer["gate_up"], "silu"), layer["down"])

    def _attention(
        self, h: mx.array, layer: Mapping[str, mx.array], block: Attention | None, r: Rows, read: Read | None
    ) -> mx.array:
        c = self.config
        b, n, qw, kw = h.shape[0], r.n, c.heads * c.head_dim, c.kv_heads * c.head_dim
        proj = self.ops.mm(h, layer["qkv"])
        if c.qkv_bias:
            proj = proj + layer["qkv_bias"]
        v = proj[..., qw + kw :].reshape(b, n, c.kv_heads, c.head_dim)
        qn, kn = (layer["q_norm"], layer["k_norm"]) if c.qk_norm else (None, None)
        d, at = c.head_dim, (r.start, r.positions)
        q = self.ops.norm_rope(proj, 0, d, c.heads, d, qn, c.eps, *at, c.rope)
        k = self.ops.norm_rope(proj, qw, d, c.kv_heads, d, kn, c.eps, *at, c.rope)
        scores_scale = c.head_dim**-0.5 * c.rope.attention**2  # YaRN's factor on both queries and keys
        o = attend(q, k, v.transpose(0, 2, 1, 3), block, r, read, True, scores_scale, ops=self.ops)
        out = self.ops.mm(o.reshape(b, o.shape[1], c.heads * c.head_dim), layer["o"])
        return out + layer["o_bias"] if c.o_bias else out


def load(
    model_dir: str,
    config: Mapping[str, Any] | Config,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    root: str | None = None,
    bits: int | None = None,
) -> Dense:
    """The dense decoder of a downloaded checkpoint (a causal LM's, or a sequence classifier's), from its config.json
    or from the settings an adapter built for a family of its own (`Config`)."""
    settings = config if isinstance(config, Config) else Config.from_hf(config)
    return Dense(settings, weights or Weights.from_dir(model_dir), dtype, root, bits)
