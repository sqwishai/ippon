"""DeBERTa-v2 and -v3 encoders, the base of the prompt-injection and jailbreak guards (Llama Prompt Guard, protectai's
deberta-v3 injection detectors) and of many NLI and zero-shot classifiers. The numbers follow Hugging Face's
modeling_deberta_v2:

- the embedding is the token's (plus a learned absolute position's when `position_biased_input`, plus a token
  type's when the model has types; projected when `embedding_size` differs), through a LayerNorm;
- every layer is post-norm, as BERT's, with disentangled attention: the scores are q.k plus content-to-position
  q_i . Pk[d(i, j)] plus position-to-content k_j . Pq[d(i, j)], all over sqrt(head_dim * 3), where d is the
  log-bucketed relative distance i - j shifted into the relative embeddings (`rel_embeddings`, LayerNormed when
  `norm_rel_ebd` says so) and Pq, Pk are those embeddings through the layer's own query and key projections
  (`share_att_key`). Here the two position terms are an additive mask of MLX's fused attention;
- the buckets are computed as transformers computes them, in float32 (`make_log_bucket_position`).

Every pass reads whole prompts: one, or a right-padded batch whose padding no query sees.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx
import numpy as np

from ..errors import ContextTooLong, UnsupportedModel
from ..quant import Matrix, arrays, rows
from ..weights import Weights
from .bert import encoder_layer, post_norm_layers
from .common import Encoder, Read, Rows, finish_layer, mask
from .ops import layer_norm

F32 = mx.float32
FAMILIES = ("deberta-v2",)  # its config.json model_types (DeBERTa-v2 and v3)


@dataclass(frozen=True)
class Config:
    hidden: int
    layers: int
    heads: int
    eps: float
    positions: int  # max_position_embeddings
    absolute: bool  # position_biased_input: learned absolute positions added to the embedding
    types: int
    buckets: int  # position_buckets (0: none)
    max_relative: int  # max_relative_positions
    norm_rel: bool  # the relative embeddings LayerNormed
    activation: str

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> Config:
        c = config
        if c.get("model_type") not in FAMILIES:
            raise UnsupportedModel(f"not a DeBERTa-v2/v3 encoder: model_type {c.get('model_type')!r}")
        if not c.get("relative_attention", False) or not c.get("share_att_key", False):
            raise UnsupportedModel("a DeBERTa-v2 variant without relative attention or shared keys is not supported")
        if sorted(c.get("pos_att_type") or []) != ["c2p", "p2c"] or c.get("conv_kernel_size", 0):
            raise UnsupportedModel("a DeBERTa-v2 variant this implementation does not cover (pos_att_type, conv)")
        if c.get("embedding_size", c["hidden_size"]) != c["hidden_size"]:
            raise UnsupportedModel("a DeBERTa-v2 embedding narrower than the model is not supported")
        positions = int(c["max_position_embeddings"])
        max_relative = int(c.get("max_relative_positions", -1))
        return cls(
            hidden=int(c["hidden_size"]),
            layers=int(c["num_hidden_layers"]),
            heads=int(c["num_attention_heads"]),
            eps=float(c.get("layer_norm_eps", 1e-7)),
            positions=positions,
            absolute=bool(c.get("position_biased_input", True)),
            types=int(c.get("type_vocab_size", 0)),
            buckets=max(0, int(c.get("position_buckets", -1))),
            max_relative=max_relative if max_relative >= 1 else positions,
            norm_rel="layer_norm" in str(c.get("norm_rel_ebd", "none")).lower(),
            activation=str(c.get("hidden_act", "gelu")),
        )

    @property
    def span(self) -> int:
        return self.buckets if self.buckets > 0 else self.max_relative


def buckets(relative: np.ndarray, bucket_size: int, max_position: int) -> np.ndarray:
    """transformers' make_log_bucket_position, in float32 as torch computes it."""
    sign = np.sign(relative)
    mid = bucket_size // 2
    near = (relative < mid) & (relative > -mid)
    abs_pos = np.where(near, mid - 1, np.abs(relative)).astype(np.float32)
    ratio = np.float32((max_position - 1) / mid)
    log_pos = (np.ceil(np.log(abs_pos / np.float32(mid)) / np.log(ratio) * np.float32(mid - 1)) + mid).astype(
        np.float32
    )
    return np.where(abs_pos <= mid, relative.astype(np.float32), log_pos * sign).astype(np.int64)


@lru_cache(maxsize=64)
def relative_index(n: int, bucket_size: int, max_position: int, span: int) -> mx.array:
    """[n, n] int32: each (query i, key j)'s row of the relative embeddings, clamp(bucket(i - j) + span)."""
    rel = np.arange(n)[:, None] - np.arange(n)[None, :]
    if bucket_size > 0 and max_position > 0:
        rel = buckets(rel, bucket_size, max_position)
    return mx.array(np.clip(rel + span, 0, 2 * span - 1).astype(np.int32))


class Deberta(Encoder):
    """A loaded DeBERTa-v2/v3 encoder (see the module docstring)."""

    config: Config

    linears = ("qkv", "o", "up", "down")
    gated_mlp = None  # an MLP without a gate

    def __init__(
        self,
        config: Config,
        weights: Weights,
        dtype: mx.Dtype = mx.bfloat16,
        root: str = "deberta.",
        bits: int | None = None,
        sep: int | None = None,
    ):
        self.config, self.dtype, self.sep = config, dtype, sep
        c = config

        def w(name: str, as_dtype: mx.Dtype | None = None) -> mx.array:
            return weights.get(root + name, as_dtype or dtype)

        def m(name: str) -> Matrix:
            return weights.matrix(root + name, dtype)

        def norm(name: str, as_dtype: mx.Dtype = dtype) -> tuple[mx.array, mx.array]:
            """Weight and bias, in the compute dtype (the embedding's and the relative embeddings', fp32)."""
            return w(name + ".weight", as_dtype), w(name + ".bias", as_dtype)

        self.embed: Matrix = m("embeddings.word_embeddings.weight")
        self.position = w("embeddings.position_embeddings.weight", F32) if c.absolute else None
        self.token_type = w("embeddings.token_type_embeddings.weight", F32) if c.types else None
        self.embed_norm = norm("embeddings.LayerNorm", F32)
        rel = w("encoder.rel_embeddings.weight", F32)[: 2 * c.span]
        self.rel = (
            layer_norm(rel, *norm("encoder.LayerNorm", F32), c.eps) if c.norm_rel else rel
        )  # [2 span, hidden] fp32
        self.layers: list[dict[str, Any]] = []
        for i in range(c.layers):
            p = f"encoder.layer.{i}."
            names = [p + f"attention.self.{x}_proj" for x in ("query", "key", "value")]
            names += [p + x for x in ("attention.output.dense", "intermediate.dense", "output.dense")]
            layer = encoder_layer(m, w, norm, names, (p + "attention.output.LayerNorm", p + "output.LayerNorm"))
            # the relative embeddings through this layer's query and key projections (biases included), once
            qkv_w = layer["qkv"] if isinstance(layer["qkv"], mx.array) else None
            if qkv_w is None:
                raise UnsupportedModel("a quantized DeBERTa checkpoint is not supported")
            h = c.hidden
            q_w, k_w = qkv_w[:h].astype(F32), qkv_w[h : 2 * h].astype(F32)
            q_b, k_b = layer["qkv_bias"][:h].astype(F32), layer["qkv_bias"][h : 2 * h].astype(F32)
            layer["pos_q"], layer["pos_k"] = self.rel @ q_w.T + q_b, self.rel @ k_w.T + k_b  # [2 span, hidden]
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(arrays([self.embed, self.position, self.token_type, *self.embed_norm, self.rel]))

    # -- the Backbone contract ----------------------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.config.hidden

    @property
    def max_tokens(self) -> int | None:
        """Its learned absolute positions, when it adds them to the embedding; else no limit (relative positions)."""
        return self.config.positions if self.config.absolute else None

    @property
    def output_matrix(self) -> Matrix:
        return self.embed

    # -- the layers ---------------------------------------------------------------------------------------------

    def _embed(self, ids: mx.array) -> mx.array:
        c = self.config
        n = ids.shape[1]
        x = rows(self.embed, ids).astype(F32)
        if self.position is not None:
            if n > c.positions:
                raise ContextTooLong(f"{n} tokens, more than the model's {c.positions}")
            x = x + self.position[:n][None]
        if self.token_type is not None:
            if self.sep is not None and self.token_type.shape[0] > 1:  # a pair: type 1 after the first [SEP]
                sep = (ids == self.sep).astype(mx.int32)
                x = x + mx.take(self.token_type, mx.minimum(mx.cumsum(sep, axis=1) - sep, 1), axis=0)
            else:
                x = x + self.token_type[0]
        return mx.fast.layer_norm(x, *self.embed_norm, c.eps).astype(self.dtype)

    def _forward(self, ids: mx.array, cache: Any, r: Rows, keep: Read) -> mx.array:
        c = self.config
        index = relative_index(r.n, c.buckets, c.max_relative, c.span)

        def attention(h: mx.array, layer: dict[str, Any], read: Read | None) -> mx.array:
            return self._attention(h, layer, r, read, index)

        return post_norm_layers(self, self._embed(ids), keep, attention)

    def _attention(
        self, h: mx.array, layer: Mapping[str, Any], r: Rows, read: Read | None, index: mx.array
    ) -> mx.array:
        c = self.config
        b, n, d = h.shape[0], r.n, c.hidden // c.heads
        qkv = self.ops.mm_bias(h, layer["qkv"], layer["qkv_bias"]).reshape(b, n, 3, c.heads, d)
        q, k, v = (qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3))  # [b, heads, n, d]
        pos_q = layer["pos_q"].reshape(-1, c.heads, d).transpose(1, 0, 2)  # [heads, 2 span, d] fp32
        pos_k = layer["pos_k"].reshape(-1, c.heads, d).transpose(1, 0, 2)
        idx = index[None]  # [1 or b, rows, n]: each read query's row of the relative embeddings, per key
        if read is not None:
            q = read.take(q, axis=2)
            idx = (
                index[read.start : read.start + read.count][None]
                if read.index is None
                else mx.take(index, read.index, axis=0)
            )
        qf, kf = q.astype(F32), k.astype(F32)
        rows_, keys = idx.shape[1], idx.shape[2]
        # content to position: q_i . Pk[idx(i, j)]; position to content: k_j . Pq[idx(i, j)]
        c2p = mx.take_along_axis(
            qf @ pos_k.swapaxes(-1, -2), mx.broadcast_to(idx[:, None], (b, c.heads, rows_, keys)), axis=-1
        )
        p2c_all = kf @ pos_q.swapaxes(-1, -2)  # [b, heads, key j, 2 span]
        by_key = mx.broadcast_to(idx.swapaxes(-1, -2)[:, None], (b, c.heads, keys, rows_))
        p2c = mx.take_along_axis(p2c_all, by_key, axis=-1).swapaxes(-1, -2)  # [b, heads, rows, n]
        scale = 1.0 / math.sqrt(d * 3)
        bias = (c2p + p2c) * scale
        m = mask(r, False)
        if isinstance(m, mx.array):  # padding: no query sees a padded key
            bias = mx.where(m, bias, -mx.inf)
        o = self.ops.attention(q, k, v, scale, bias.astype(q.dtype))
        out: mx.array = self.ops.mm_bias(
            o.transpose(0, 2, 1, 3).reshape(b, o.shape[2], c.hidden), layer["o"], layer["o_bias"]
        )
        return out


def load(
    model_dir: str,
    config: Mapping[str, Any],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    root: str = "deberta.",
    bits: int | None = None,
    sep: int | None = None,
) -> Deberta:
    """The DeBERTa-v2/v3 encoder of a downloaded checkpoint."""
    return Deberta(Config.from_hf(config), weights or Weights.from_dir(model_dir), dtype, root, bits, sep)
