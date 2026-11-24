"""ModernBERT: a bidirectional encoder whose attention layers alternate between global and sliding-window ones
(ModernBERT and mmBERT checkpoints, under a classifier head: adapters/classifier.py).

The numbers follow Hugging Face's ModernBertModel:

- the token embedding goes through a LayerNorm; every layer adds attention over a LayerNorm of its input (the
  first layer has no norm there), then a GeGLU MLP over another; a final LayerNorm. The norms have no bias;
- attention projects q, k and v with one matrix, rotates them with rotate_half RoPE over the whole head in fp32,
  the global layers with their base and the sliding ones with theirs; a sliding layer's query sees the keys at most
  `local_attention / 2` positions away on either side;
- the MLP is Wo(gelu(a) * b), a and b the two halves of Wi x, with the exact GELU; here in fp32, rounded once.

Every pass reads whole prompts: a single prompt, or a right-padded batch whose padding no query sees.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..errors import UnsupportedModel
from ..quant import Matrix, arrays, rows
from ..weights import Weights
from .common import Encoder, Read, Rows, Snapshots, attend, banded_pass, finish_layer
from .rope import RoPE

F32 = mx.float32
FAMILIES = ("modernbert",)  # its config.json model_types


@dataclass(frozen=True)
class Config:
    hidden: int
    layer_types: tuple[str, ...]  # "full_attention" or "sliding_attention", per layer
    heads: int
    head_dim: int
    ropes: Mapping[str, RoPE]  # by layer type: full and sliding attention
    window: int  # how far a sliding layer's query sees, either side
    eps: float

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> Config:
        c = config
        if c.get("model_type") not in FAMILIES:
            raise UnsupportedModel(f"not a ModernBERT model: model_type {c.get('model_type')!r}")
        if c.get("attention_bias") or c.get("mlp_bias") or c.get("norm_bias"):
            raise UnsupportedModel("a ModernBERT variant with biases, which this implementation does not cover")
        if c.get("hidden_activation", "gelu") != "gelu":
            raise UnsupportedModel(f"hidden_activation {c.get('hidden_activation')!r} is not supported")
        n = int(c["num_hidden_layers"])
        every = int(c.get("global_attn_every_n_layers", 3))
        types = c.get("layer_types") or ["full_attention" if i % every == 0 else "sliding_attention" for i in range(n)]
        rope_parameters = c.get("rope_parameters") or {}
        heads = int(c["num_attention_heads"])
        head_dim = int(c["hidden_size"]) // heads
        thetas = {
            "full_attention": c.get("global_rope_theta", 160_000.0),
            "sliding_attention": c.get("local_rope_theta", 10_000.0),
        }
        ropes = {kind: RoPE.from_hf(head_dim, rope_parameters.get(kind), theta=float(t)) for kind, t in thetas.items()}
        if any(rope.attention != 1.0 for rope in ropes.values()):
            raise UnsupportedModel("YaRN RoPE is not supported for ModernBERT")
        return cls(
            hidden=int(c["hidden_size"]),
            layer_types=tuple(types),
            heads=heads,
            head_dim=head_dim,
            ropes=ropes,
            window=int(c.get("local_attention", 128)) // 2,
            eps=float(c.get("norm_eps", 1e-5)),
        )


class ModernBert(Encoder):
    """A loaded ModernBERT encoder (see the module docstring)."""

    config: Config

    # each layer's weight matrices that inputs multiply (quant.mm): what quantization converts
    linears = ("qkv", "o", "wi", "wo")
    gated_mlp = ("wi", "gelu")  # the [gate; up] projection and its GLU's activation

    def __init__(
        self,
        config: Config,
        weights: Weights,
        dtype: mx.Dtype = mx.bfloat16,
        root: str = "",
        bits: int | None = None,
    ):
        self.config, self.dtype = config, dtype

        def w(name: str, as_dtype: mx.Dtype | None = None) -> mx.array:
            return weights.get(root + name, as_dtype or dtype)

        def m(name: str) -> Matrix:  # a weight matrix, quantized as stored
            return weights.matrix(root + name, dtype)

        self.embed: Matrix = m("embeddings.tok_embeddings.weight")
        self.embed_norm = w("embeddings.norm.weight", F32)
        self.final_norm = w("final_norm.weight", F32)
        self.layers: list[dict[str, Any]] = []  # matrices may be QuantizedMatrix
        for i in range(len(config.layer_types)):
            p = f"layers.{i}."
            layer: dict[str, Any] = {
                "qkv": m(p + "attn.Wqkv.weight"),
                "o": m(p + "attn.Wo.weight"),
                "mlp_norm": w(p + "mlp_norm.weight", F32),
                "wi": m(p + "mlp.Wi.weight"),
                "wo": m(p + "mlp.Wo.weight"),
            }
            if i > 0:
                layer["attn_norm"] = w(p + "attn_norm.weight", F32)
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(arrays([self.embed, self.embed_norm, self.final_norm]))

    # -- the Backbone contract ----------------------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.config.hidden

    @property
    def output_matrix(self) -> Matrix:
        return self.embed

    # -- the layers ---------------------------------------------------------------------------------------------

    @property
    def layer_kinds(self) -> Sequence[str]:
        return self.config.layer_types

    def mask_keys(self, r: Rows) -> Sequence[tuple[bool, int | None]]:
        window = self.config.window
        return [(False, None)] + ([] if banded_pass(r, window) else [(False, window)])

    def _forward(self, ids: mx.array, cache: Any, r: Rows, keep: Read) -> mx.array:
        """Final-normed hidden states of the `keep` rows; each MLP output is added to the residual together with the
        next norm (`ops.add_norm`)."""
        eps = self.config.eps
        x = self.ops.norm(rows(self.embed, ids), self.embed_norm, eps, "layer")
        x, pending, _ = self._layers(x, None, r, keep, (eps,))
        if pending is None:  # the last layer read no row
            return x
        return self.ops.add_norm(x, pending, self.final_norm, eps, "layer")[1]

    def _layer(
        self,
        kind: Hashable,
        layer: dict[str, Any],
        x: mx.array,
        pending: mx.array | None,
        block: Any,
        r: Rows,
        read: Read | None,
        kept: list[Snapshots],
        constants: tuple[Any, ...],
        at: Any,
    ) -> tuple[mx.array, mx.array | None]:
        """One encoder layer: attention over a LayerNorm of its input (the first layer's input as it is, its
        previous MLP output `pending` added in one `ops.add_norm`), then the GeGLU MLP over another."""
        (eps,) = constants
        ops = self.ops
        if pending is None:  # the first layer has no norm before its attention
            h = x
        else:
            x, h = ops.add_norm(x, pending, layer["attn_norm"], eps, "layer")
        assert isinstance(kind, str)
        mixed = self._attention(h, layer, kind, r, read)
        if read is not None:
            if read.size == 0:
                return mx.zeros((x.shape[0], 0, self.config.hidden), dtype=self.dtype), None
            x = read.take(x)
        x, h = ops.add_norm(x, mixed, layer["mlp_norm"], eps, "layer")
        return x, ops.mm(ops.glu_mm(h, layer["wi"], "gelu"), layer["wo"])

    def _attention(self, h: mx.array, layer: Mapping[str, mx.array], kind: str, r: Rows, read: Read | None) -> mx.array:
        c = self.config
        b, n = h.shape[0], r.n
        qkv = self.ops.mm(h, layer["qkv"]).reshape(b, n, 3, c.heads, c.head_dim)
        sliding = kind == "sliding_attention"
        rope = c.ropes["sliding_attention" if sliding else "full_attention"]
        q, k, v = (qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3))
        q = rope(q, r.start, r.positions)  # MLX's RoPE computes in fp32 and rounds once to the input's dtype
        k = rope(k, r.start, r.positions)
        o = attend(q, k, v, None, r, read, False, c.head_dim**-0.5, c.window if sliding else None, self.ops)
        return self.ops.mm(o.reshape(b, o.shape[1], c.hidden), layer["o"])


def load(
    model_dir: str,
    config: Mapping[str, Any],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    root: str = "",
    bits: int | None = None,
) -> ModernBert:
    """The ModernBERT encoder of a downloaded checkpoint; `root` is its tensors' prefix ("model." for a classifier)."""
    return ModernBert(Config.from_hf(config), weights or Weights.from_dir(model_dir), dtype, root, bits)
