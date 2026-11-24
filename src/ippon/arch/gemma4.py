"""Gemma 4 text decoders: the 26B-A4B mixture of experts, the dense ones (12B, 31B), and the per-layer-embedding /
shared-KV ones (E2B, E4B), for decision models built on Gemma 4 (Jeff-1.1-Medium, adapters/jeff.py; Strands Decider
v2, adapters/strands.py; tested against transformers on tiny models). The numbers follow Hugging Face's
modeling_gemma4 (transformers 5.17):

- the embedding is scaled by sqrt(hidden), the scale rounded to the weights' dtype as transformers rounds it;
- RMSNorm multiplies by the checkpoint's weight (not 1 + w, as Gemma 3 does), in fp32, rounded once; every layer
  has four, before attention, on its output, before the MLP and on its output;
- attention has no 1/sqrt(d) scale (its per-head norms on queries and keys set the scale) and two kinds of layers:
  sliding ones (`head_dim`, `num_key_value_heads`, RoPE over the head at the sliding base, a query sees the
  `sliding_window` keys ending at itself) and full ones (`global_head_dim`, `num_global_key_value_heads`, and with
  `attention_k_eq_v` no value projection: the values are the key projection, before the key norm). Values get an
  RMSNorm without a weight. Full layers use "proportional" RoPE: the first `partial_rotary_factor` of the head's
  angle pairs rotate, at base ** (2i / head_dim), the rest not at all;
- with `num_kv_shared_layers` (E2B, E4B), the last `num_kv_shared_layers` layers project only queries and attend over
  the keys and values produced by the last unshared layer of the same layer type;
- the MLP is GeGLU with GELU's tanh approximation (possibly double-wide on shared layers, taken from each layer's
  projection shapes). With `enable_moe_block` a layer also runs its experts on the same residual: the router reads it
  through a weightless RMSNorm times `router.scale / sqrt(hidden)`, takes the softmax's top_k (in fp32), renormalizes
  them and multiplies each by its expert's `per_expert_scale`; the experts (GeGLU each) read it through their own
  norm. The MLP's and the experts' outputs are normed apart, added, and normed again before the residual;
- with `hidden_size_per_layer_input` (E2B, E4B), each token carries a per-layer input vector combining a per-layer
  embedding table (`embed_tokens_per_layer`, kept on host via `weights.table` and scaled by `sqrt(per_layer_dim)`)
  and a normed projection of the token embedding (`per_layer_model_projection * hidden**-0.5`), averaged with
  `2**-0.5`. After the MLP residual, each layer applies `gelu_tanh(ple_gate(x)) * ple[i]`, projects back through
  `ple_proj` and `post_per_layer_input_norm`, and adds to the residual;
- each layer's output (residual included) is multiplied by its stored `layer_scalar`.

With `causal=False` the full layers read every position and the sliding ones stay causal (Jeff 1.1's
`"bidirectional": "full"`). With `block=True` every layer (both sliding and full) attends without a sliding window in
block-bidirectional mode: the prefix both ways, then each tail over the prefix and both ways within itself (Strands
Decider v2's `"attention_mode": "prefix_bidir"`). The vision and audio towers are not read: decisions here are text
only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..checkpoint import EMBEDDING
from ..errors import UnsupportedModel
from ..quant import Matrix, Table, arrays, concat, rows
from ..weights import Weights
from .cache import Attention, HybridCache, Recurrent
from .common import (
    AttentionDecoder,
    Read,
    Rows,
    attend,
    finish_layer,
    hand_over,
    output_head,
    remember,
    stacked_experts,
)
from .ops import activate
from .rope import RoPE, proportional_freqs

F32 = mx.float32
FAMILIES = ("gemma4_text", "gemma4", "gemma4_unified_text", "gemma4_unified")  # unified: the 12B
SLIDING, FULL = "sliding_attention", "full_attention"


@dataclass(frozen=True)
class Config:
    hidden: int
    layers: int
    heads: int
    eps: float
    window: int  # keys before a query a sliding layer reads (transformers' sliding_window less the query's own)
    layer_types: tuple[str, ...]  # SLIDING or FULL
    head_dim: Mapping[str, int]  # by layer type
    kv_heads: Mapping[str, int]
    ropes: Mapping[str, RoPE]  # by layer type ("proportional" frequencies on full layers)
    k_eq_v: bool  # full layers' values are their key projection
    experts: int = 0
    top_k: int = 0
    tied: bool = True
    causal: bool = True  # False: full layers read every position; sliding layers stay causal (unless block=True)
    block: bool = False  # block attention (Backbone.block): every layer prefix-bidirectional, no sliding window
    per_layer_dim: int = 0  # hidden_size_per_layer_input (E2B, E4B)
    per_layer_vocab: int = 0  # vocab_size_per_layer_input (E2B, E4B)
    kv_shared_layers: int = 0  # num_kv_shared_layers (E2B, E4B)

    @property
    def first_shared(self) -> int:
        return self.layers - self.kv_shared_layers

    @property
    def kv_producers(self) -> dict[str, int]:
        if not self.kv_shared_layers:
            return {}
        cut = self.first_shared
        return {kind: max(i for i in range(cut) if self.layer_types[i] == kind) for kind in set(self.layer_types[cut:])}

    @classmethod
    def from_hf(cls, config: Mapping[str, Any], *, causal: bool = True, block: bool = False) -> Config:
        c = config.get("text_config", config)
        if c.get("model_type") not in FAMILIES:
            raise UnsupportedModel(f"not a Gemma 4 text model: model_type {c.get('model_type')!r}")
        if c.get("use_bidirectional_attention") == "all" or c.get("attention_bias"):
            raise UnsupportedModel("a Gemma 4 variant with bidirectional attention or attention biases")
        if c.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise UnsupportedModel(f"hidden_activation {c.get('hidden_activation')!r} is not supported")
        n, heads = int(c["num_hidden_layers"]), int(c["num_attention_heads"])
        types = tuple(c.get("layer_types") or [SLIDING if (i + 1) % 6 else FULL for i in range(n)])
        if set(types) - {SLIDING, FULL}:
            raise UnsupportedModel(f"layer types {sorted(set(types))} are not supported")
        per_layer_dim = int(c.get("hidden_size_per_layer_input") or 0)
        per_layer_vocab = int(c.get("vocab_size_per_layer_input") or c.get("vocab_size") or 0) if per_layer_dim else 0
        kv_shared = int(c.get("num_kv_shared_layers") or 0)
        if kv_shared < 0 or kv_shared >= n:
            raise UnsupportedModel(f"num_kv_shared_layers={kv_shared} is invalid for {n} layers")
        if kv_shared and (set(types[n - kv_shared :]) - set(types[: n - kv_shared])):
            raise UnsupportedModel("shared KV layer type has no unshared producer layer")
        k_eq_v = bool(c.get("attention_k_eq_v", False))
        head_dim = {SLIDING: int(c.get("head_dim", 256)), FULL: int(c.get("global_head_dim") or c.get("head_dim", 256))}
        kv = int(c.get("num_key_value_heads") or heads)
        global_kv = c.get("num_global_key_value_heads")
        kv_heads = {SLIDING: kv, FULL: int(global_kv) if k_eq_v and global_kv is not None else kv}
        head_dim, kv_heads = per_layer(c.get("per_layer_config"), types, head_dim, kv_heads)
        params = dict(c.get("rope_parameters") or {})
        ropes: dict[str, RoPE] = {}
        for kind, default_theta in ((SLIDING, 10_000.0), (FULL, 1_000_000.0)):
            p = dict(params.get(kind) or {})
            theta = float(p.get("rope_theta", default_theta))
            rope_type = p.get("rope_type", "default")
            if rope_type == "proportional":
                ropes[kind] = RoPE(head_dim[kind], theta, proportional_freqs(head_dim[kind], theta, p))
            elif rope_type == "default" and float(p.get("factor", 1.0)) == 1.0:
                ropes[kind] = RoPE(head_dim[kind], theta)
            else:
                raise UnsupportedModel(f"rope_type {rope_type!r} is not supported for Gemma 4")
        moe = bool(c.get("enable_moe_block", False))
        return cls(
            hidden=int(c["hidden_size"]),
            layers=n,
            heads=heads,
            eps=float(c.get("rms_norm_eps", 1e-6)),
            window=int(c.get("sliding_window") or 1024) - 1,
            layer_types=types,
            head_dim=head_dim,
            kv_heads=kv_heads,
            ropes=ropes,
            k_eq_v=k_eq_v,
            experts=int(c["num_experts"]) if moe else 0,
            top_k=int(c["top_k_experts"]) if moe else 0,
            tied=bool(c.get("tie_word_embeddings", True)),
            causal=causal and not block,
            block=block,
            per_layer_dim=per_layer_dim,
            per_layer_vocab=per_layer_vocab,
            kv_shared_layers=kv_shared,
        )


def per_layer(
    overrides: Mapping[str, Any] | None, types: tuple[str, ...], head_dim: dict[str, int], kv_heads: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    """Head widths and key-value heads by layer type from `per_layer_config` (transformers 5.17's form, by layer
    index, which the 12B's config stores instead of `global_head_dim` and `num_global_key_value_heads`): the layers of
    one type must agree."""
    if not overrides:
        return head_dim, kv_heads
    found: dict[str, set[tuple[int, int]]] = {}
    for i, kind in enumerate(types):
        o = overrides.get(str(i)) or overrides.get(f"{i:02d}") or {}
        found.setdefault(kind, set()).add(
            (int(o.get("head_dim", head_dim[kind])), int(o.get("num_key_value_heads", kv_heads[kind])))
        )
    for kind, shapes in found.items():
        if len(shapes) != 1:
            raise UnsupportedModel(f"{kind} layers of different widths ({sorted(shapes)}) are not supported")
        ((head_dim[kind], kv_heads[kind]),) = shapes
    return head_dim, kv_heads


class Gemma4(AttentionDecoder):
    """A loaded Gemma 4 text decoder (see the module docstring); the passes are the dense decoders'."""

    config: Config

    linears = ("gate_up", "down", "qkv", "q", "o", "ple_gate", "ple_proj")
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
        self.block = config.block
        c = config
        root = weights.text_root() if root is None else root

        def n(name: str) -> mx.array:  # a norm weight, in fp32
            return weights.get(root + name, F32)

        def m(name: str) -> Matrix:
            return weights.matrix(root + name, dtype)

        self.embed: Table = weights.table(root + EMBEDDING, dtype)
        # rounded to the weights' dtype, as transformers rounds it (exact in fp32)
        self.embed_scale = mx.array(math.sqrt(c.hidden), dtype=dtype).astype(F32)
        self.embed_per_layer: Table | None = None
        self.per_layer_embed_scale = mx.array(1.0, dtype=F32)
        self.per_layer_model_proj: Matrix | None = None
        self.per_layer_proj_norm: mx.array | None = None
        if c.per_layer_dim:
            self.embed_per_layer = weights.table(root + "embed_tokens_per_layer.weight", dtype)
            self.per_layer_embed_scale = mx.array(math.sqrt(c.per_layer_dim), dtype=dtype).astype(F32)
            self.per_layer_model_proj = finish_layer({"w": m("per_layer_model_projection.weight")}, ("w",), bits)["w"]
            self.per_layer_proj_norm = n("per_layer_projection_norm.weight")
        self.final_norm = n("norm.weight")
        self.lm_head = output_head(weights, root, c.tied, dtype)
        self.scalars: list[float] = []
        self.layers: list[dict[str, Any]] = []
        first_shared = c.first_shared
        for i, kind in enumerate(c.layer_types):
            p, a = f"layers.{i}.", f"layers.{i}.self_attn."
            layer: dict[str, Any] = {
                "in_norm": n(p + "input_layernorm.weight"),
                "attn_out_norm": n(p + "post_attention_layernorm.weight"),
                "mlp_norm": n(p + "pre_feedforward_layernorm.weight"),
                "mlp_out_norm": n(p + "post_feedforward_layernorm.weight"),
                "o": m(a + "o_proj.weight"),
                "q_norm": n(a + "q_norm.weight"),
                "gate_up": concat([m(p + "mlp.gate_proj.weight"), m(p + "mlp.up_proj.weight")]),
                "down": m(p + "mlp.down_proj.weight"),
            }
            if i < first_shared:
                values = c.k_eq_v and kind == FULL
                layer |= {
                    "qkv": concat([m(a + f"{x}_proj.weight") for x in ("qk" if values else "qkv")]),
                    "k_norm": n(a + "k_norm.weight"),
                    "v_norm": mx.ones((c.head_dim[kind],), dtype=F32),  # weightless
                }
            else:
                layer["q"] = m(a + "q_proj.weight")
            if c.per_layer_dim:
                layer |= {
                    "ple_gate": m(p + "per_layer_input_gate.weight"),
                    "ple_proj": m(p + "per_layer_projection.weight"),
                    "ple_norm": n(p + "post_per_layer_input_norm.weight"),
                }
            if c.experts:
                layer |= self._experts_of(weights, root, p, dtype, bits)
            scalar = root + p + "layer_scalar"
            value = weights.get(scalar, F32).reshape(-1)[0].item() if scalar in weights else 1.0
            assert isinstance(value, float)
            self.scalars.append(value)
            self.layers.append(finish_layer(layer, self.linears, bits))
        mx.eval(
            arrays(
                [
                    self.embed,
                    self.final_norm,
                    self.embed_scale,
                    self.lm_head,
                    self.embed_per_layer,
                    self.per_layer_embed_scale,
                    self.per_layer_model_proj,
                    self.per_layer_proj_norm,
                ]
            )
        )

    def _experts_of(self, weights: Weights, root: str, p: str, dtype: mx.Dtype, bits: int | None) -> dict[str, Any]:
        """A layer's router and experts, every expert's [gate; up] and down stacked (`stacked_experts`)."""
        c = self.config
        gate_up, down = stacked_experts(weights, root + p, dtype, c.experts, "experts.switch_glu")
        router_scale = weights.get(root + p + "router.scale", F32) * c.hidden**-0.5
        return {
            "router": weights.matrix(
                root + p + "router.proj.weight", dtype
            ),  # as stored (mlx-lm keeps it at 8 bits): it chooses the experts
            "router_norm": router_scale,
            "expert_scale": weights.get(root + p + "router.per_expert_scale", F32),
            "experts_norm": weights.get(root + p + "pre_feedforward_layernorm_2.weight", F32),
            "mlp_out_norm_1": weights.get(root + p + "post_feedforward_layernorm_1.weight", F32),
            "experts_out_norm": weights.get(root + p + "post_feedforward_layernorm_2.weight", F32),
            "experts_gate_up": gate_up.quantized(bits),
            "experts_down": down.quantized(bits),
        }

    def _blocks(self, capacity: int) -> list[Attention | Recurrent]:
        """Unshared layers' keys and values, each of its own heads and width."""
        c = self.config
        return [
            Attention.empty(c.kv_heads[t], capacity, c.head_dim[t], self.dtype) for t in c.layer_types[: c.first_shared]
        ]

    @property
    def max_tokens(self) -> int | None:
        """No limit: its sliding layers attend within their window, as Gemma's do."""
        return None

    @property
    def causal(self) -> bool:
        return self.config.causal

    def _forward(self, ids: mx.array, cache: HybridCache | None, r: Rows, keep: Read) -> mx.array:
        c, ops, last = self.config, self.ops, len(self.layers) - 1
        x = (rows(self.embed, ids).astype(F32) * self.embed_scale).astype(self.dtype)
        ple: mx.array | None = None
        if c.per_layer_dim:
            assert self.embed_per_layer is not None
            assert self.per_layer_model_proj is not None
            assert self.per_layer_proj_norm is not None
            ple_ids = mx.where(ids < c.per_layer_vocab, ids, mx.zeros_like(ids)) if c.per_layer_vocab else ids
            tok_ple = (rows(self.embed_per_layer, ple_ids).astype(F32) * self.per_layer_embed_scale).astype(self.dtype)
            tok_ple = tok_ple.reshape(*ids.shape, c.layers, c.per_layer_dim)
            mod_ple = ops.mm(x, self.per_layer_model_proj) * mx.array(c.hidden**-0.5, dtype=self.dtype)
            mod_ple = ops.norm(mod_ple.reshape(*ids.shape, c.layers, c.per_layer_dim), self.per_layer_proj_norm, c.eps)
            ple = (mod_ple + tok_ple) * mx.array(2.0**-0.5, dtype=self.dtype)
        pending: mx.array | None = None  # the previous layer's final residual update, not yet added to x
        first_shared, producers = c.first_shared, c.kv_producers
        shared_kv: dict[str, tuple[mx.array, mx.array]] = {}
        for i, (kind, layer) in enumerate(zip(c.layer_types, self.layers, strict=True)):
            if i == first_shared and keep.size == 0:
                remember(cache, r, [])
                return mx.zeros((x.shape[0], 0, c.hidden), dtype=self.dtype)
            read = keep if i == last else None
            if pending is None:
                h = ops.norm(x, layer["in_norm"], c.eps)
            elif self.scalars[i - 1] == 1.0:
                x, h = ops.add_norm(x, pending, layer["in_norm"], c.eps)
            else:  # the residual rounded, then times the layer's scalar, then rounded (as transformers)
                x = (x + pending) * mx.array(self.scalars[i - 1], dtype=x.dtype)
                h = ops.norm(x, layer["in_norm"], c.eps)
            if i < first_shared:
                block = cache.layers[i] if cache is not None else None
                assert block is None or isinstance(block, Attention)
                mixed, kv = self._attention(h, layer, kind, block, r, read, None)
                if producers.get(kind) == i:
                    shared_kv[kind] = kv
            else:
                mixed, _ = self._attention(h, layer, kind, None, r, read, shared_kv[kind])
            if read is not None:
                if read.size == 0:
                    remember(cache, r, [])
                    return mx.zeros((x.shape[0], 0, c.hidden), dtype=self.dtype)
                x = read.take(x)
            x, h = ops.add_norm(x, ops.norm(mixed, layer["attn_out_norm"], c.eps), layer["mlp_norm"], c.eps)
            f = ops.mm(ops.glu_mm(h, layer["gate_up"], "gelu_tanh"), layer["down"])
            if c.experts:
                f = ops.norm(f, layer["mlp_out_norm_1"], c.eps)
                e = ops.norm(self._moe(x, layer), layer["experts_out_norm"], c.eps)
                f = ops.add_norm(f, e, layer["mlp_out_norm"], c.eps)[1]
            else:
                f = ops.norm(f, layer["mlp_out_norm"], c.eps)
            if ple is not None:
                x = x + f
                ple_i = ple[:, :, i] if read is None else read.take(ple[:, :, i])
                g = activate(ops.mm(x, layer["ple_gate"]), "gelu_pytorch_tanh") * ple_i
                f = ops.norm(ops.mm(g, layer["ple_proj"]), layer["ple_norm"], c.eps)
            pending = f
            hand_over(ops, i, last, x, pending)
        remember(cache, r, [])
        assert pending is not None
        x = x + pending
        if self.scalars[last] != 1.0:
            x = x * mx.array(self.scalars[last], dtype=x.dtype)
        return ops.norm(x, self.final_norm, c.eps)

    def _moe(self, x: mx.array, layer: Mapping[str, Any]) -> mx.array:
        """The routed experts, as transformers' Gemma 4 computes them: router logits in the compute dtype from the
        router's norm of the residual, their softmax in fp32, the top_k renormalized and times their experts' scales
        (kept in fp32), each chosen expert's GeGLU on the experts' norm of the residual times its weight, summed (here
        in fp32, rounded once)."""
        c, ops, lead, width = self.config, self.ops, x.shape[:-1], x.shape[-1]
        flat = x.reshape(-1, width)
        scores_in = ops.norm(flat, layer["router_norm"], c.eps)
        inputs = ops.norm(flat, layer["experts_norm"], c.eps)
        top, order, experts, pairs = ops.route(
            scores_in, layer["router"], c.top_k, inputs=inputs, expert_scale=layer["expert_scale"]
        )
        g = ops.glu(ops.expert_mm(pairs, layer["experts_gate_up"], experts), "gelu_tanh")
        y = ops.expert_mm(g, layer["experts_down"], experts)[:, 0]
        return ops.combine(y, order, top).reshape(*lead, width)

    def _attention(
        self,
        h: mx.array,
        layer: Mapping[str, Any],
        kind: str,
        block: Attention | None,
        r: Rows,
        read: Read | None,
        shared: tuple[mx.array, mx.array] | None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        c = self.config
        d, kvh = c.head_dim[kind], c.kv_heads[kind]
        b, n, qw, kw = h.shape[0], r.n, c.heads * d, kvh * d
        rope, where = c.ropes[kind], (r.start, r.positions)
        if shared is None:
            proj = self.ops.mm(h, layer["qkv"])
            at = qw if c.k_eq_v and kind == FULL else qw + kw  # the values: the keys' projection, or their own
            v = self.ops.norm(proj[..., at : at + kw].reshape(b, n, kvh, d), layer["v_norm"], c.eps)
            q = self.ops.norm_rope(proj, 0, d, c.heads, d, layer["q_norm"], c.eps, *where, rope)
            k = self.ops.norm_rope(proj, qw, d, kvh, d, layer["k_norm"], c.eps, *where, rope)
            v_t = v.transpose(0, 2, 1, 3)
            keys, values = block.write(r.start, k, v_t) if block is not None else (k, v_t)
        else:
            keys, values = shared
            proj = self.ops.mm(h, layer["q"])
            q = self.ops.norm_rope(proj, 0, d, c.heads, d, layer["q_norm"], c.eps, *where, rope)
        window = None if c.block else (c.window if kind == SLIDING else None)
        causal = False if c.block else (c.causal or kind != FULL)
        o = attend(q, keys, values, None, r, read, causal, 1.0, window, ops=self.ops)
        out: mx.array = self.ops.mm(o.reshape(b, o.shape[1], qw), layer["o"])
        return out, (keys, values)


def load(
    model_dir: str,
    config: Mapping[str, Any],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    weights: Weights | None = None,
    root: str | None = None,
    bits: int | None = None,
    causal: bool = True,
    block: bool = False,
) -> Gemma4:
    """The Gemma 4 text decoder of a downloaded checkpoint (`causal=False`: full layers read both ways; `block=True`:
    every layer attends in prefix-bidirectional mode with no sliding window)."""
    settings = Config.from_hf(config, causal=causal, block=block)
    return Gemma4(settings, weights or Weights.from_dir(model_dir), dtype, root, bits)
