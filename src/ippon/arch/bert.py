"""BERT-family encoders: BERT, RoBERTa, XLM-RoBERTa (bge rerankers among them) and DistilBERT, the bases of most
text classifiers people fine-tune. The numbers follow Hugging Face's modeling_bert / modeling_roberta /
modeling_distilbert:

- the embedding is the token's, plus a learned absolute position's (RoBERTa's from padding_idx + 1), plus a token
  type's (BERT: 1 after the first [SEP], as a pair's tokenizer marks it; none in DistilBERT), through a LayerNorm;
- every layer is post-norm: x = LN(x + attention(x)), x = LN(x + W2 act(W1 x)), with biases on every projection and
  norm; attention is full, bidirectional, over every real token of a prompt;
- the activation is the exact GELU (or the config's: gelu_new, relu); the last layer's output is the result, with
  no final norm.

Every pass reads whole prompts: one, or a right-padded batch whose padding no query sees.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..errors import ContextTooLong, UnsupportedModel
from ..quant import Matrix, arrays, concat, rows
from ..weights import Weights
from .common import Encoder, Read, Rows, attend, finish_layer, hand_over
from .ops import activate

F32 = mx.float32
FAMILIES = ("bert", "roberta", "xlm-roberta", "distilbert")
ROOTS = {"bert": "bert.", "roberta": "roberta.", "xlm-roberta": "roberta.", "distilbert": "distilbert."}


@dataclass(frozen=True)
class Config:
    family: str
    hidden: int
    layers: int
    heads: int
    eps: float
    positions: int  # learned positions, RoBERTa's offset included
    position_offset: int  # RoBERTa: padding_idx + 1
    types: int  # token types (0: none)
    activation: str

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> Config:
        c = config
        family = c.get("model_type")
        if family not in FAMILIES:
            raise UnsupportedModel(f"not a BERT-family encoder this module covers ({', '.join(FAMILIES)}): {family!r}")
        if family == "distilbert":
            return cls(
                family,
                int(c["dim"]),
                int(c["n_layers"]),
                int(c["n_heads"]),
                1e-12,
                int(c["max_position_embeddings"]),
                0,
                0,
                str(c.get("activation", "gelu")),
            )
        if c.get("position_embedding_type", "absolute") != "absolute":
            raise UnsupportedModel(f"position_embedding_type {c.get('position_embedding_type')!r} is not supported")
        roberta = family in ("roberta", "xlm-roberta")
        return cls(
            family,
            int(c["hidden_size"]),
            int(c["num_hidden_layers"]),
            int(c["num_attention_heads"]),
            float(c.get("layer_norm_eps", 1e-12)),
            int(c["max_position_embeddings"]),
            int(c.get("pad_token_id", 1)) + 1 if roberta else 0,
            int(c.get("type_vocab_size", 2)),
            str(c.get("hidden_act", "gelu")),
        )

    @property
    def max_tokens(self) -> int:
        return self.positions - self.position_offset


def encoder_layer(
    m: Callable[[str], Matrix],
    w: Callable[[str], mx.array],
    norm: Callable[[str], tuple[mx.array, mx.array]],
    names: Sequence[str],
    norms: tuple[str, str],
) -> dict[str, Any]:
    """A post-norm encoder layer's weights, from the full names of its query, key, value, attention output, up and
    down projections (each with a bias) and of its two LayerNorms (after attention, after the MLP)."""
    q, k, v, o, up, down = names
    return {
        "qkv": concat([m(x + ".weight") for x in (q, k, v)]),
        "qkv_bias": mx.concatenate([w(x + ".bias") for x in (q, k, v)]),
        "o": m(o + ".weight"),
        "o_bias": w(o + ".bias"),
        "up": m(up + ".weight"),
        "up_bias": w(up + ".bias"),
        "down": m(down + ".weight"),
        "down_bias": w(down + ".bias"),
        "attn_norm": norm(norms[0]),
        "mlp_norm": norm(norms[1]),
    }


def post_norm_layers(
    model: Any, x: mx.array, keep: Read, attention: Callable[[mx.array, dict[str, Any], Read | None], mx.array]
) -> mx.array:
    """The last layer's hidden states of the `keep` rows of a post-norm encoder (BERT's and DeBERTa's), from its
    embeddings x: each layer's `attention`, then LayerNorm of the residual, the MLP and LayerNorm again."""
    c, ops, last = model.config, model.ops, len(model.layers) - 1
    for i, layer in enumerate(model.layers):
        read = keep if i == last else None
        mixed = attention(x, layer, read)
        if read is not None:
            if read.size == 0:
                return mx.zeros((x.shape[0], 0, c.hidden), dtype=model.dtype)
            x = read.take(x)
        (aw, ab), (mw, mb) = layer["attn_norm"], layer["mlp_norm"]
        x = ops.add_layer_norm(x, mixed, aw, ab, c.eps)
        f = ops.mm_bias(
            activate(ops.mm_bias(x, layer["up"], layer["up_bias"]), c.activation), layer["down"], layer["down_bias"]
        )
        x = ops.add_layer_norm(x, f, mw, mb, c.eps)
        hand_over(ops, i, last, x)
    return x


class Bert(Encoder):
    """A loaded BERT-family encoder (see the module docstring)."""

    config: Config

    linears = ("qkv", "o", "up", "down")  # each layer's weight matrices that inputs multiply (quant.mm)
    gated_mlp = None  # an MLP without a gate

    def __init__(
        self,
        config: Config,
        weights: Weights,
        dtype: mx.Dtype = mx.bfloat16,
        root: str | None = None,
        bits: int | None = None,
        sep: int | None = None,
    ):
        self.config, self.dtype, self.sep = config, dtype, sep
        c = config
        root = ROOTS[c.family] if root is None else root

        def w(name: str, as_dtype: mx.Dtype | None = None) -> mx.array:
            return weights.get(root + name, as_dtype or dtype)

        def m(name: str) -> Matrix:
            return weights.matrix(root + name, dtype)

        def norm(name: str, as_dtype: mx.Dtype = dtype) -> tuple[mx.array, mx.array]:
            """Weight and bias (older checkpoints: gamma and beta), in the compute dtype as transformers' half-precision
            runs hold them (the embedding's, whose sum is fp32, in fp32)."""
            if root + name + ".weight" in weights:
                return w(name + ".weight", as_dtype), w(name + ".bias", as_dtype)
            return w(name + ".gamma", as_dtype), w(name + ".beta", as_dtype)

        e = "embeddings."
        self.embed: Matrix = m(e + "word_embeddings.weight")
        self.position = w(e + "position_embeddings.weight", F32)
        self.token_type = w(e + "token_type_embeddings.weight", F32) if c.types else None
        self.embed_norm = norm(e + "LayerNorm", F32)
        self.layers: list[dict[str, Any]] = []
        for i in range(c.layers):
            if c.family == "distilbert":
                p = f"transformer.layer.{i}."
                names = ("q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2")
                q, k, v, o, up, down = (n if n.startswith("ffn") else "attention." + n for n in names)
                norms = (p + "sa_layer_norm", p + "output_layer_norm")
            else:
                p = f"encoder.layer.{i}."
                q, k, v = (f"attention.self.{x}" for x in ("query", "key", "value"))
                o, up, down = "attention.output.dense", "intermediate.dense", "output.dense"
                norms = (p + "attention.output.LayerNorm", p + "output.LayerNorm")
            layer = encoder_layer(m, w, norm, [p + x for x in (q, k, v, o, up, down)], norms)
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(arrays([self.embed, self.position, *self.embed_norm, self.token_type]))

    # -- the Backbone contract ----------------------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.config.hidden

    @property
    def max_tokens(self) -> int | None:
        """Its learned positions (less RoBERTa's offset)."""
        return self.config.max_tokens

    @property
    def output_matrix(self) -> Matrix:
        return self.embed

    # -- the layers ---------------------------------------------------------------------------------------------

    def _embed(self, ids: mx.array) -> mx.array:
        c = self.config
        n = ids.shape[1]
        if n > c.max_tokens:
            raise ContextTooLong(f"{n} tokens, more than the model's {c.max_tokens}")
        x = rows(self.embed, ids).astype(F32) + self.position[c.position_offset : c.position_offset + n][None]
        if self.token_type is not None:
            if self.sep is not None and self.token_type.shape[0] > 1:  # a pair: type 1 after the first [SEP]
                sep = (ids == self.sep).astype(mx.int32)
                x = x + mx.take(self.token_type, mx.minimum(mx.cumsum(sep, axis=1) - sep, 1), axis=0)
            else:
                x = x + self.token_type[0]
        return mx.fast.layer_norm(x, *self.embed_norm, c.eps).astype(self.dtype)  # summed and normed in fp32

    def _forward(self, ids: mx.array, cache: Any, r: Rows, keep: Read) -> mx.array:
        """The last layer's hidden states of the `keep` rows."""
        return post_norm_layers(self, self._embed(ids), keep, lambda h, layer, read: self._attention(h, layer, r, read))

    def _attention(self, h: mx.array, layer: Mapping[str, mx.array], r: Rows, read: Read | None) -> mx.array:
        c = self.config
        b, n, d = h.shape[0], r.n, c.hidden // c.heads
        qkv = self.ops.mm_bias(h, layer["qkv"], layer["qkv_bias"]).reshape(b, n, 3, c.heads, d)
        q, k, v = (qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3))
        o = attend(q, k, v, None, r, read, False, d**-0.5, ops=self.ops)
        return self.ops.mm_bias(o.reshape(b, o.shape[1], c.hidden), layer["o"], layer["o_bias"])


def load(
    model_dir: str,
    config: Mapping[str, Any],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    root: str | None = None,
    bits: int | None = None,
    sep: int | None = None,
) -> Bert:
    """The BERT-family encoder of a downloaded checkpoint; `sep` is the tokenizer's [SEP] id, which starts a pair's
    second text."""
    return Bert(Config.from_hf(config), weights or Weights.from_dir(model_dir), dtype, root, bits, sep)
