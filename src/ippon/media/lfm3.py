"""LFM3-VL's pictures, as Liquid's d1-2B and d1-Omni-2B read them (media.py, image_processing_lfm3_vl.py and
modeling_lfm3_vl.py in their checkpoints).

Preprocessing: a picture over the checkpoint's pixel cap (config.json's d1_vision_max_pixels) is first downscaled to
that many pixels with Pillow's bicubic (`capped`, as media.py's cap_pixels), then goes through Qwen2-VL's processor,
which LFM3-VL's is line for line (`qwen.preprocess`: each side a multiple of 32, torchvision's antialiased bicubic,
(x - 127.5) / 127.5, 16 x 16 patches in merge-block order over the 2 frames of a temporal patch).

The tower (`Lfm3Vision`, Lfm3VlVisionModel and its projector):

- the patches through a linear patchifier (no bias) and an RMSNorm;
- pre-norm layers, each its operator then a SwiGLU MLP, and no final norm. A `conv` layer projects to B, C and x,
  convolves B * x over the picture's patch grid with a 3 x 3 depthwise kernel (zero padding), multiplies by C and
  projects out; an attention layer is GQA over the whole picture both ways, a per-head RMSNorm on queries and keys and
  the axial RoPE: each patch's row turns the first quarter of a head's channels and its column the second, the pair
  repeated for rotate_half (`qwen.rotary_angles`, theta 1e6);
- the projector: a LayerNorm per patch, each merge block's 4 patches as one row, a Linear, the exact GELU and a
  Linear to the text model's width.

A picture becomes (height / 32) x (width / 32) rows after its resize: at most about 1,600 under d1's cap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ..arch.common import finish_layer
from ..arch.ops import MLX, Ops, activate, layer_norm
from ..errors import UnsupportedModel
from ..quant import Matrix, arrays, concat
from ..weights import Weights
from .decode import pillow
from .qwen import ImageConfig, Patches, QwenVision, block_order, rotary_angles, smart_resize

F32 = mx.float32
TOWER, PROJECTOR = "model.vision_tower.", "model.multi_modal_projector."


def capped(rgb: np.ndarray, max_pixels: int) -> np.ndarray:
    """An H x W x 3 uint8 picture downscaled to at most `max_pixels` with Pillow's bicubic, each side scaled by
    sqrt(max_pixels / pixels) and truncated (at least 1); as it is when it has no more."""
    height, width = rgb.shape[:2]
    if width * height <= max_pixels:
        return rgb
    scale = (max_pixels / (width * height)) ** 0.5
    image, _ = pillow()
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return np.asarray(image.fromarray(rgb).resize(size, image.Resampling.BICUBIC))


def capped_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """The (width, height) `capped` gives a picture of that size."""
    if width * height <= max_pixels:
        return width, height
    scale = (max_pixels / (width * height)) ** 0.5
    return max(1, int(width * scale)), max(1, int(height * scale))


def picture_rows(width: int, height: int, image: ImageConfig, max_pixels: int) -> int:
    """The rows a width x height picture becomes, from its size alone."""
    w, h = capped_size(width, height, max_pixels)
    h, w = smart_resize(h, w, image.factor, image.min_pixels, image.max_pixels)
    return (h // image.factor) * (w // image.factor)


@dataclass(frozen=True)
class VisionConfig:
    """The tower's shape (config.json's vision_config) and the projector's (its top level)."""

    hidden: int
    heads: int
    kv_heads: int
    layer_types: tuple[str, ...]
    intermediate: int
    kernel: int
    eps: float
    theta: float
    patch: int
    temporal: int
    merge: int
    channels: int
    projected: int  # the projector's hidden width
    projector_eps: float
    out: int  # the text model's width

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> VisionConfig:
        v = config["vision_config"]
        if (v.get("rope_parameters") or {}).get("rope_type") != "axial" or config.get("projector_hidden_act") != "gelu":
            raise UnsupportedModel("an LFM3-VL vision tower this implementation does not cover (rope, projector)")
        if not config.get("projector_bias", True) or any(t not in ("conv", "full_attention") for t in v["layer_types"]):
            raise UnsupportedModel("an LFM3-VL vision tower this implementation does not cover (layers, projector)")
        return cls(
            hidden=int(v["hidden_size"]),
            heads=int(v["num_attention_heads"]),
            kv_heads=int(v["num_key_value_heads"]),
            layer_types=tuple(v["layer_types"]),
            intermediate=int(v["intermediate_size"]),
            kernel=int(v["conv_kernel_size"]),
            eps=float(v["norm_eps"]),
            theta=float(v["rope_parameters"]["rope_theta"]),
            patch=int(v["patch_size"]),
            temporal=int(v["temporal_patch_size"]),
            merge=int(v["spatial_merge_size"]),
            channels=int(v.get("in_channels", 3)),
            projected=int(config["projector_hidden_size"]),
            projector_eps=float(config["projector_layer_norm_eps"]),
            out=int(config["text_config"]["hidden_size"]),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_dim(self) -> int:
        return self.channels * self.temporal * self.patch * self.patch

    def check(self, image: ImageConfig) -> None:
        """UnsupportedModel when the processor's patches are not the tower's."""
        if (image.patch, image.temporal, image.merge) != (self.patch, self.temporal, self.merge):
            raise UnsupportedModel("the image processor's patches are not the vision tower's")


class Lfm3Vision:
    """The LFM3-VL vision tower and projector (see the module docstring): a picture's patches to its rows."""

    linears = ("in", "out", "qkv", "o", "gate_up", "down")
    gated_mlp = ("gate_up", "silu")
    ops: Ops = MLX  # how its matmuls and attention run: set from the chip profile, as a backbone's

    def __init__(self, config: VisionConfig, weights: Weights, dtype: mx.Dtype = mx.float32):
        self.config, self.dtype = config, dtype
        c = config

        def m(name: str) -> Matrix:
            return weights.matrix(name, dtype)

        def n(name: str) -> mx.array:  # a norm weight, in fp32
            return weights.get(name, F32)

        if TOWER + "patchifier.weight" not in weights:
            raise UnsupportedModel(f"{weights.source}: no LFM3-VL vision tower under {TOWER!r}")
        self.patchifier = m(TOWER + "patchifier.weight")
        self.embedding_norm = n(TOWER + "embedding_norm.weight")
        self.layers: list[dict[str, Any]] = []
        for i, kind in enumerate(c.layer_types):
            p = f"{TOWER}layers.{i}."
            layer: dict[str, Any] = {
                "in_norm": n(p + "operator_norm.weight"),
                "mlp_norm": n(p + "ffn_norm.weight"),
                "gate_up": concat([m(p + "feed_forward.w1.weight"), m(p + "feed_forward.w3.weight")]),
                "down": m(p + "feed_forward.w2.weight"),
            }
            if kind == "conv":
                taps = weights.get(p + "conv.spatial_conv.weight", F32)  # [channels, 1, k, k]
                layer |= {
                    "in": m(p + "conv.in_proj.weight"),
                    "taps": taps.reshape(c.hidden, c.kernel, c.kernel, 1),  # mx.conv2d's [out, k, k, in / groups]
                    "out": m(p + "conv.out_proj.weight"),
                }
            else:
                a = p + "self_attn."
                layer |= {
                    "qkv": concat([m(a + f"{x}_proj.weight") for x in "qkv"]),
                    "o": m(a + "out_proj.weight"),
                    "q_norm": n(a + "q_layernorm.weight"),
                    "k_norm": n(a + "k_layernorm.weight"),
                }
            self.layers.append(finish_layer(layer, self.linears, None))
        self.projector_norm = (weights.get(PROJECTOR + "layer_norm.weight"), weights.get(PROJECTOR + "layer_norm.bias"))
        self.up, self.up_bias = m(PROJECTOR + "linear_1.weight"), weights.get(PROJECTOR + "linear_1.bias", dtype)
        self.down, self.down_bias = m(PROJECTOR + "linear_2.weight"), weights.get(PROJECTOR + "linear_2.bias", dtype)
        mx.eval(arrays([self.patchifier, self.embedding_norm, *self.projector_norm, self.up, self.up_bias]))
        mx.eval(arrays([self.down, self.down_bias]))

    @classmethod
    def load(cls, weights: Weights, config: Mapping[str, Any], dtype: mx.Dtype = mx.float32) -> Lfm3Vision:
        return cls(VisionConfig.from_hf(config), weights, dtype)

    @property
    def hidden_size(self) -> int:
        """The width of the rows it returns: the text model's."""
        return self.config.out

    def __call__(self, pictures: Sequence[Patches]) -> mx.array:
        """Picture and video rows in order, in the compute dtype. Each temporal patch has its own spatial
        attention and convolution, as modeling_lfm3_vl.get_vision_frame_shapes and its attention lengths specify."""
        if not pictures:
            return mx.zeros((0, self.config.out), self.dtype)
        rows = []
        for p in pictures:
            t, h, w = p.grid
            for frame in range(t):
                rows.append(self.picture(Patches(p.pixels[frame * h * w : (frame + 1) * h * w], (1, h, w))))
        return rows[0] if len(rows) == 1 else mx.concatenate(rows)

    def picture(self, picture: Patches) -> mx.array:
        """One picture's rows [rows, out]: its own attention and convolutions."""
        c, ops = self.config, self.ops
        _, h, w = picture.grid
        if picture.pixels.shape != (h * w, c.patch_dim) or picture.grid[0] != 1 or h % c.merge or w % c.merge:
            raise ValueError(f"a picture of grid {picture.grid} has patches {picture.pixels.shape}")
        x = ops.norm(ops.mm(mx.array(picture.pixels).astype(self.dtype), self.patchifier), self.embedding_norm, c.eps)
        cos, sin = (mx.array(a)[:, None] for a in rotary_angles(h, w, c.head_dim, c.theta, c.merge))
        raster = mx.array(np.argsort(block_order(np.arange(h * w).reshape(h, w), c.merge)))  # block order -> raster
        blocks = mx.array(block_order(np.arange(h * w).reshape(h, w), c.merge))  # raster -> block order
        for kind, layer in zip(c.layer_types, self.layers, strict=True):
            y = ops.norm(x, layer["in_norm"], c.eps)
            if kind == "conv":
                proj = ops.mm(y, layer["in"])
                d = c.hidden
                gated = (proj[:, :d] * proj[:, 2 * d :])[raster].reshape(1, h, w, d).astype(F32)
                conv = mx.conv2d(gated, layer["taps"], padding=c.kernel // 2, groups=d).reshape(h * w, d)[blocks]
                y = ops.mm(proj[:, d : 2 * d] * conv.astype(proj.dtype), layer["out"])
            else:
                y = self._attention(y, layer, cos, sin)
            x = x + y
            x = x + ops.mm(ops.glu_mm(ops.norm(x, layer["mlp_norm"], c.eps), layer["gate_up"], "silu"), layer["down"])
        weight, bias = (t.astype(x.dtype) for t in self.projector_norm)
        x = layer_norm(x, weight, bias, c.projector_eps)
        x = activate(ops.mm_bias(x.reshape(-1, c.hidden * c.merge**2), self.up, self.up_bias), "gelu")
        return ops.mm_bias(x, self.down, self.down_bias)

    def _attention(self, y: mx.array, layer: Mapping[str, Any], cos: mx.array, sin: mx.array) -> mx.array:
        c, ops = self.config, self.ops
        n, hd, qw, kw = y.shape[0], c.head_dim, c.heads * c.head_dim, c.kv_heads * c.head_dim
        proj = ops.mm(y, layer["qkv"])
        q = ops.norm(proj[:, :qw].reshape(n, c.heads, hd), layer["q_norm"], c.eps)
        k = ops.norm(proj[:, qw : qw + kw].reshape(n, c.kv_heads, hd), layer["k_norm"], c.eps)
        q, k = (QwenVision.rotate(t, cos, sin).transpose(1, 0, 2)[None] for t in (q, k))
        v = proj[:, qw + kw :].reshape(n, c.kv_heads, hd).transpose(1, 0, 2)[None]
        o = ops.attention(q, k, v, hd**-0.5, None)[0]
        return ops.mm(o.transpose(1, 0, 2).reshape(n, qw), layer["o"])
