"""Qwen3.5's images: the image processor, exact to transformers' Qwen2VLImageProcessor (its torchvision backend), the
vision tower and patch merger (`QwenVision`), exact to transformers' Qwen3_5VisionModel, and a checkpoint's image
and video paths through them (`QwenImages`, `QwenVideos`), which LFM3's pictures reuse with a tower of their own (the
multimodal RoPE positions the text model gives image rows are arch/qwen35.py's). Perplexity's
pplx-decider flash and Cloudflare's Clef-Flash both read images through it, each with its own tower size and
pixel ceiling (`VisionConfig.from_hf`, `ImageConfig.from_hf`).

Preprocessing (`smart_resize`, `preprocess`), as the processor does it:

- each side rounded to a multiple of patch x merge (32) with Python's round, so halves go to even; above max_pixels
  both sides scaled down by sqrt(area / max), floored; below min_pixels scaled up, ceiled; an aspect over 200 is
  refused;
- the uint8 image resized to that size by torchvision's antialiased bicubic (`resize.resize_frames`, bit for
  bit), the resample its processor_config.json names (3);
- rescale and normalize fused as transformers fuses them: (x - mean / rescale) / (std / rescale) in float32, so
  (x - 127.5) / 127.5 for mean = std = 0.5;
- 16 x 16 patches in merge-block order (each 2 x 2 block of patches contiguous, the blocks row-major), each
  flattened as (channel, frame, row, column) with the image repeated over the 2 frames of a temporal patch: 1,536
  values per patch. Its grid (t, h, w) in patches is (1, height / 16, width / 16).

Each image becomes (h / 2) x (w / 2) rows, (height / 32) x (width / 32): 300 for a 640 x 480 photo.

Videos (`VideoConfig`, `video_size`, `preprocess_video`), as Qwen3VLVideoProcessor does them for frames already sampled
(media/video.py, by the creator's own decoder): every frame resized to one size, chosen from the frame's height and
width and the frame count (its smart_resize: a side under 32 pixels scaled up first, the budget the whole clip's, with a
per-frame cap when the creator sets one), each by the same antialiased bicubic; normalized as an image; patches of two
consecutive frames, (channel, frame, row, column), in merge-block order frame pair by frame pair. A clip of T frames (T
even) is a grid (T / 2, h, w): T / 2 x (h / 2) x (w / 2) rows. The text model reads each frame pair as an image of its
own (Qwen3_5Model.get_rope_index splits video_grid_thw by frame pair: timestamps separate them in the prompt).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx
import numpy as np

from .. import native
from ..arch.common import Kept, expand, finish_layer, media_digest, media_id
from ..arch.ops import ACTIVATIONS, MLX, Ops, activate, layer_norm
from ..errors import ContextTooLong, InvalidRequest, UnsupportedModel
from ..quant import Matrix, arrays
from ..weights import Weights
from .decode import Media, image_size, load_rgb
from .paths import Sampling, Seen, config_token, encode_kept, encoder_loader
from .resize import Mode, resize_frames
from .video import Clip, Shrink, timestamps

if TYPE_CHECKING:
    from ..adapter import LoadContext, Rendering
    from ..arch.base import Backbone
    from ..model_dir import ModelDir
    from ..tokenizer import Tokenizer

F32 = mx.float32
MAX_ASPECT = 200  # the processor's ceiling on the longer side over the shorter
# Encoded vision tokens across all images in one request at most: allows e.g. 5 full-res photos on 1.64 MP models
# (d3, d1) or 12 phone photos on 262k-pixel models while blocking multi-image prefill OOMs (12 x 1,600 = 19,200 tokens).
IMAGE_MAX_TOKENS = 8_192
RESAMPLE: dict[int, Mode] = {2: "bilinear", 3: "bicubic"}  # PIL's resample codes, as torchvision maps them
ROOTS = ("model.visual.", "visual.", "vision_tower.")  # Qwen3_5ForConditionalGeneration's, Qwen3_5Model's, MLX's


@dataclass(frozen=True)
class ImageConfig:
    """Qwen2VLImageProcessor's settings (processor_config.json's "image_processor"): the patch, the merge and the
    temporal patch, the pixel range a resized image falls in, the per-channel mean and std, the rescale factor, and
    the resample (PIL's code: 3 bicubic, 2 bilinear)."""

    patch: int
    merge: int
    temporal: int
    min_pixels: int
    max_pixels: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    rescale: float
    resample: int = 3

    @classmethod
    def from_hf(cls, processor: Mapping[str, Any]) -> ImageConfig:
        """From processor_config.json (its "image_processor") or a preprocessor_config.json. min_pixels and
        max_pixels, when given, take the place of size's shortest_edge and longest_edge, as the processor's own
        constructor does."""
        p = processor.get("image_processor", processor)
        if not isinstance(p, Mapping):
            raise UnsupportedModel("the processor config has no image_processor")
        kind = p.get("image_processor_type", "Qwen2VLImageProcessor")
        # Liquid's LFM3-VL processor is Qwen2-VL's line for line (its smart_resize, normalization and patches)
        if kind not in ("Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast", "Lfm3VlImageProcessor"):
            raise UnsupportedModel(f"not Qwen's image processor: {kind!r}")
        for flag in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
            if not p.get(flag, True):
                raise UnsupportedModel(f"a Qwen image processor with {flag} off is not supported")
        size = dict(p.get("size") or {})
        least = p.get("min_pixels", size.get("shortest_edge"))
        most = p.get("max_pixels", size.get("longest_edge"))
        if not least or not most:
            raise UnsupportedModel("the image processor's size needs shortest_edge and longest_edge (pixels)")
        resample = int(p.get("resample", 3))
        if resample not in RESAMPLE:
            raise UnsupportedModel(f"resample {resample} is not supported (3, bicubic; 2, bilinear)")

        def channels(name: str) -> tuple[float, float, float]:
            values = p.get(name)
            if not isinstance(values, Sequence) or len(values) != 3:
                raise UnsupportedModel(f"the image processor's {name} must give 3 channels")
            return float(values[0]), float(values[1]), float(values[2])

        c = cls(
            int(p.get("patch_size", 14)),
            int(p.get("merge_size", 2)),
            int(p.get("temporal_patch_size", 2)),
            int(least),
            int(most),
            channels("image_mean"),
            channels("image_std"),
            float(p.get("rescale_factor", 1 / 255)),
            resample,
        )
        if min(c.patch, c.merge, c.temporal) < 1 or not 0 < c.min_pixels <= c.max_pixels or 0.0 in c.std:
            raise UnsupportedModel(f"an image processor of {c} is not one Qwen's code can run")
        return c

    def pixels(self, min_pixels: int | None = None, max_pixels: int | None = None) -> ImageConfig:
        """These settings with another pixel range, as a creator sets it on its processor (pplx-decider: its
        max_pixels)."""
        return replace(
            self,
            min_pixels=self.min_pixels if min_pixels is None else int(min_pixels),
            max_pixels=self.max_pixels if max_pixels is None else int(max_pixels),
        )

    @property
    def factor(self) -> int:
        """What each resized side is a multiple of: patch x merge."""
        return self.patch * self.merge


@dataclass(frozen=True)
class VideoConfig:
    """Qwen3VLVideoProcessor's settings (video_preprocessor_config.json) as a creator sets them: `image` holds the
    patch, merge, temporal patch, normalization and resample, and the pixel range of the whole clip (its size's
    shortest_edge and longest_edge); `frame_tokens`, when set, caps each frame at that many rows' worth of pixels
    (the processor's cap_pixels_per_frame with max_video_tokens)."""

    image: ImageConfig
    frame_tokens: int | None = None

    @classmethod
    def from_hf(cls, processor: Mapping[str, Any]) -> VideoConfig:
        """From video_preprocessor_config.json (or a processor_config.json's "video_processor"), with the class's
        defaults: bicubic, mean and std 0.5, patch 16, merge 2, temporal patch 2."""
        p = processor.get("video_processor", processor)
        if not isinstance(p, Mapping):
            raise UnsupportedModel("the processor config has no video_processor")
        kind = p.get("video_processor_type", "Qwen3VLVideoProcessor")
        if kind != "Qwen3VLVideoProcessor":
            raise UnsupportedModel(f"not Qwen3-VL's video processor: {kind!r}")
        if p.get("do_center_crop") or not p.get("do_resize", True):
            raise UnsupportedModel("a Qwen video processor that crops or doesn't resize is not supported")
        defaults: dict[str, Any] = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "resample": 3}
        defaults |= {"image_mean": [0.5] * 3, "image_std": [0.5] * 3}
        image = ImageConfig.from_hf({**defaults, **p, "image_processor_type": "Qwen2VLImageProcessor"})
        return cls(image, None)

    def limits(self, min_pixels: int, max_pixels: int, frame_tokens: int | None) -> VideoConfig:
        """These settings with the clip's pixel range and per-frame cap a creator sets on its processor."""
        return VideoConfig(self.image.pixels(min_pixels, max_pixels), frame_tokens)


def video_size(frames: int, height: int, width: int, config: VideoConfig) -> tuple[int, int]:
    """The (height, width) every frame of a clip of `frames` frames is resized to: Qwen3VLVideoProcessor.resize's
    budget (with cap_pixels_per_frame, each frame's pixels min(frame_tokens x factor^2, longest_edge // frames) but
    at least shortest_edge x 1.05, times the frame count) and its smart_resize, line for line: a side under the
    factor scales both up first (truncated), an aspect over 200 is refused, each side rounded to the factor, then
    scaled down within the budget over the clip's frames (floored) or up to its floor (ceiled)."""
    c = config.image
    factor, temporal = c.factor, c.temporal
    min_pixels, max_pixels = c.min_pixels, c.max_pixels
    if config.frame_tokens is not None:
        per_frame = max(min(config.frame_tokens * factor * factor, max_pixels // frames), int(min_pixels * 1.05))
        max_pixels = per_frame * frames
    if frames < temporal:
        raise InvalidRequest(f"a video of {frames} frames: Qwen's video processor needs at least {temporal}")
    if min(height, width) < 1:
        raise InvalidRequest("an empty video frame")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise InvalidRequest(
            f"{width} x {height} video frames: their aspect ratio is over {MAX_ASPECT}, which Qwen's video processor "
            "refuses"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(frames / temporal) * temporal
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def video_grid(frames: int, height: int, width: int, config: VideoConfig) -> tuple[int, int, int]:
    """A clip's grid (t, h, w) in patches, from its frame count and frame size alone: its frames paired (the last
    repeated for an odd count), each pair one temporal patch."""
    h, w = video_size(frames, height, width, config)
    temporal = config.image.temporal
    return -(-frames // temporal), h // config.image.patch, w // config.image.patch


def preprocess_video(frames: np.ndarray, config: VideoConfig) -> Patches:
    """A clip's sampled frames [frames, H, W, 3] uint8 RGB (media/video.Clip, as a creator's decoder gives them) as
    Qwen's video processor gives them to the tower: grid (frames / 2, h / 16, w / 16), each patch two consecutive
    frames."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3 or not len(frames):
        raise ValueError(f"expected [frames, H, W, 3] uint8 frames, not {frames.dtype} {frames.shape}")
    c = config.image
    count, height, width = frames.shape[:3]
    h, w = video_size(count, height, width, config)
    if (h, w) != (height, width):
        frames = resize_frames(frames, h, w, RESAMPLE[c.resample])
    if pad := -count % c.temporal:
        frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])
    return patches(frames, c)


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """transformers' smart_resize, line for line: the (height, width) an image is resized to. InvalidRequest for an
    empty image or one whose aspect is over 200, which the processor refuses."""
    if min(height, width) < 1:
        raise InvalidRequest("an empty image")
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise InvalidRequest(
            f"a {width} x {height} image: its aspect ratio is over {MAX_ASPECT}, which Qwen's image processor refuses"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@dataclass(frozen=True)
class Patches:
    """One image as the tower takes it: its patches [t * h * w, channels * temporal * patch * patch] float32, in
    merge-block order, and its grid (t, h, w) in patches (the processor's image_grid_thw)."""

    pixels: np.ndarray
    grid: tuple[int, int, int]

    def rows(self, merge: int = 2) -> int:
        """The rows it becomes in the text model: one per merge x merge patches."""
        t, h, w = self.grid
        return t * (h // merge) * (w // merge)


def image_tokens(width: int, height: int, config: ImageConfig) -> int:
    """The rows a width x height image becomes, from its size alone."""
    h, w = smart_resize(height, width, config.factor, config.min_pixels, config.max_pixels)
    return (h // config.factor) * (w // config.factor)


def patches(frames: np.ndarray, config: ImageConfig) -> Patches:
    """[frames, H, W, 3] uint8, one image or a clip's frames in pairs (sides multiples of patch x merge), as the
    processor's patches: rescaled and normalized as the processor fuses the two, the mean and std in float32 times
    1 / rescale (rounded to float32), then (x - mean) / std in float32, laid out in merge-block order by the native
    library (native/src/lib.rs, ippon_patches_f32). An image fills its temporal patch's frames with itself."""
    count, height, width, channels = frames.shape
    p, m, t = config.patch, config.merge, config.temporal
    inverse = np.float32(1.0 / config.rescale)
    mean = (np.asarray(config.mean, np.float32) * inverse).astype(np.float32)
    std = (np.asarray(config.std, np.float32) * inverse).astype(np.float32)
    groups, gh, gw = 1 if count == 1 else count // t, height // p, width // p
    out = np.empty((groups * gh * gw, channels * t * p * p), np.float32)
    native.patches(np.ascontiguousarray(frames), mean, std, p, m, t, out)
    return Patches(out, (groups, gh, gw))


def preprocess(rgb: np.ndarray, config: ImageConfig) -> Patches:
    """An H x W x 3 uint8 RGB image (`decode.load_rgb`) as Qwen's processor gives it to the tower. InvalidRequest
    for an empty image or an aspect over 200."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 uint8 image, not {rgb.dtype} {rgb.shape}")
    height, width = rgb.shape[:2]
    h, w = smart_resize(height, width, config.factor, config.min_pixels, config.max_pixels)
    resized = resize_frames(rgb[None], h, w, RESAMPLE[config.resample]) if (h, w) != (height, width) else rgb[None]
    return patches(resized, config)


@dataclass(frozen=True)
class VisionConfig:
    """The Qwen3.5 vision tower's shape (config.json's vision_config, with Qwen3_5VisionConfig's defaults)."""

    depth: int = 27
    hidden: int = 1152
    intermediate: int = 4304
    heads: int = 16
    out: int = 3584  # the merger's output: the text model's width
    activation: str = "gelu_pytorch_tanh"
    patch: int = 16
    temporal: int = 2
    merge: int = 2
    channels: int = 3
    positions: int = 2304  # a side x side learned table
    theta: float = 10_000.0
    eps: float = 1e-6  # every LayerNorm's, fixed in transformers' code

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> VisionConfig:
        v = config.get("vision_config")
        if not isinstance(v, Mapping) or v.get("model_type") != "qwen3_5_vision":
            raise UnsupportedModel("not a Qwen3.5 vision tower (vision_config.model_type 'qwen3_5_vision')")
        rope = v.get("rope_parameters") or {}
        if rope.get("rope_type", "axial") != "axial":
            raise UnsupportedModel(f"a Qwen3.5 vision tower with {rope.get('rope_type')!r} RoPE (only axial)")
        sizes = {k: v.get(k) for k in ("patch_size", "temporal_patch_size")}
        if any(isinstance(s, Sequence) for s in sizes.values()):
            raise UnsupportedModel(f"a Qwen3.5 vision tower with non-square patches: {sizes}")
        c = cls(
            int(v.get("depth", 27)),
            int(v.get("hidden_size", 1152)),
            int(v.get("intermediate_size", 4304)),
            int(v.get("num_heads", v.get("num_attention_heads", 16))),
            int(v.get("out_hidden_size", 3584)),
            str(v.get("hidden_act", "gelu_pytorch_tanh")),
            int(v.get("patch_size", 16)),
            int(v.get("temporal_patch_size", 2)),
            int(v.get("spatial_merge_size", 2)),
            int(v.get("in_channels", 3)),
            int(v.get("num_position_embeddings", 2304)),
            float(rope.get("rope_theta", 10_000.0)),
        )
        if c.side**2 != c.positions or c.hidden % c.heads or c.head_dim % 4:
            raise UnsupportedModel(
                f"a vision tower of {c.positions} positions and {c.heads} heads of {c.hidden}: the table must be "
                "square and each head's width a multiple of 4 (its two axes' rotary halves)"
            )
        if c.activation not in ACTIVATIONS:
            raise UnsupportedModel(f"the vision tower's activation {c.activation!r} is not supported")
        return c

    @property
    def side(self) -> int:
        return math.isqrt(self.positions)

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_dim(self) -> int:
        """The values in one patch: channels x temporal x patch x patch."""
        return self.channels * self.temporal * self.patch * self.patch

    def check(self, image: ImageConfig) -> None:
        """UnsupportedModel when the processor's patches are not the tower's."""
        if (image.patch, image.temporal, image.merge) != (self.patch, self.temporal, self.merge):
            raise UnsupportedModel(
                f"the image processor's patch, temporal patch and merge {image.patch, image.temporal, image.merge} "
                f"are not the vision tower's {self.patch, self.temporal, self.merge}"
            )


def axis_taps(n: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    """transformers' bilinear, align_corners=True taps along one axis of the side x side table for n patches: each
    patch's two table indices [n, 2] (floor and floor + 1, clamped) and weights [n, 2], in float32 as torch computes
    them (src = i * (side - 1) / (n - 1), the weights 1 - |src - tap|)."""
    index = np.arange(n, dtype=np.float32)
    src = (index * np.float32(side - 1)) / np.float32(max(n - 1, 1))
    floor = np.floor(src)
    offsets = np.arange(2)
    taps = np.clip(floor.astype(np.int64)[:, None] + offsets, 0, side - 1)
    distance = np.abs((src - floor)[:, None] - offsets.astype(np.float32))
    weights = np.maximum(np.float32(1) - distance, np.float32(0)).astype(np.float32)
    return taps, weights


def block_order(x: np.ndarray, merge: int) -> np.ndarray:
    """An [h, w, ...] grid's cells in merge-block order: [h * w, ...]."""
    h, w = x.shape[:2]
    blocks = x.reshape(h // merge, merge, w // merge, merge, *x.shape[2:]).swapaxes(1, 2)
    out: np.ndarray = blocks.reshape(h * w, *x.shape[2:])
    return out


def position_embeddings(table: np.ndarray, h: int, w: int, merge: int) -> np.ndarray:
    """The learned [side * side, d] float32 table resampled to an h x w patch grid, in merge-block order: [h * w, d]
    float32, as Qwen3_5VisionModel computes it in transformers 5.17 (each patch's four corners' rows times their
    float32 weights, the products summed in order). transformers 5.10.2, which Clef-Flash's card names, placed the
    coordinates with torch.linspace instead: for grid sides 2 to 1,024 that moves 17% of them by one float32 ulp
    (at most 3.8e-6 of a table step), far under the tower's float32 noise."""
    side = math.isqrt(table.shape[0])
    rows, row_weights = axis_taps(h, side)
    columns, column_weights = axis_taps(w, side)
    out = np.empty((h * w, table.shape[1]), np.float32)
    native.position_table(
        np.ascontiguousarray(table, np.float32), rows, row_weights, columns, column_weights, merge, out
    )
    return out


def rotary_angles(h: int, w: int, head_dim: int, theta: float, merge: int) -> tuple[np.ndarray, np.ndarray]:
    """The axial RoPE's cos and sin [h * w, head_dim] float32 for an h x w patch grid, in merge-block order:
    head_dim / 4 frequencies theta ** (-2i / (head_dim / 2)), the row's angles then the column's, the pair repeated
    for rotate_half's two halves (Qwen3_5VisionRotaryEmbedding; LFM3-VL's axial RoPE)."""
    half = head_dim // 2
    exponent = (np.arange(0, half, 2, dtype=np.float32) / np.float32(half)).astype(np.float32)
    inv = (np.float32(1) / (theta ** exponent.astype(np.float64)).astype(np.float32)).astype(np.float32)
    r, c = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    angles = np.concatenate([r[..., None] * inv, c[..., None] * inv], axis=-1).astype(np.float32)
    angles = block_order(np.concatenate([angles, angles], axis=-1), merge).astype(np.float64)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


class QwenVision:
    """The Qwen3.5 vision tower and patch merger over a list of images, as transformers' Qwen3_5VisionModel computes
    them (its pooler_output, the rows the text model reads), the images' rows concatenated in order:

    - the patches through the patch embedding, a Conv3d whose kernel is its stride: a Linear over each patch's
      channels x 2 x 16 x 16 values, with a bias;
    - plus the learned 48 x 48 position table resampled to the image's patch grid, bilinear with align_corners=True,
      in merge-block order, computed in float32 from the table as the model holds it and rounded to the compute
      dtype (`position_embeddings`);
    - `depth` pre-norm blocks (LayerNorm eps 1e-6 with biases; a fused qkv with its bias; each head's queries and
      keys through the axial 2-D RoPE in float32, rounded back; full attention within each image only, as
      transformers splits its packed rows at cu_seqlens; an MLP through GELU's tanh form);
    - the merger: a LayerNorm per patch, each merge block's 4 patches as one row of 4 x hidden, a Linear, the exact
      GELU, and a Linear to the text model's width.

    There is no windowed attention and no deepstack: Qwen3.5's tower has neither (a config's
    deepstack_visual_indexes is ignored, as transformers ignores it).

    Weights come from `root` (found among ROOTS when None): the creators' names (pplx-decider's `visual.`,
    Clef-Flash's `model.visual.`, the Conv3d weight [hidden, channels, temporal, patch, patch]) or an MLX build's
    (`vision_tower.`, the weight channels-last, [hidden, temporal, patch, patch, channels]). Its blocks are layers
    of qkv, o, up and down matrices, so kernel choices and the precision check can see them.

    Against transformers 5.17.0's own Qwen3_5VisionModel with the real vision weights, fp32 on the M5 Pro's CPU, on
    the same pixel_values (preprocess gives them bit for bit): pplx-decider flash v1.2's rows for a 320 x 240 image
    (80 rows) and a 1,280 x 960 one (1,200 rows) differ by at most 1.8e-4 and 1.5e-3 in values up to 7.1 and 9.4
    (relative norm 1.6e-5 and 3.4e-5); Clef-Flash's by 4.5e-4 and 2.2e-3 in values up to 21 and 26 (7.6e-6 and
    2.5e-5). That is float32's own noise: against the same model run in float64, transformers' fp32 rows are 3.0e-5
    and 3.1e-5 away in relative norm on the larger image, these 4.1e-5 and 3.5e-5."""

    linears = ("qkv", "o", "up", "down")
    gated_mlp = None  # an MLP without a gate
    ops: Ops = MLX  # how its matmuls and attention run: set from the chip profile, as a backbone's
    POSITION_ROWS = 65_536  # resampled position tables and rotary angles kept, by grid, up to this many rows in all

    def __init__(self, config: VisionConfig, weights: Weights, dtype: mx.Dtype = mx.float32, root: str | None = None):
        self.config, self.dtype = config, dtype
        c = config
        if root is None:
            root = next((r for r in ROOTS if r + "patch_embed.proj.weight" in weights), None)
            if root is None:
                raise UnsupportedModel(f"{weights.source}: no Qwen3.5 vision tower under any of {ROOTS}")
        elif root + "patch_embed.proj.weight" not in weights:
            raise UnsupportedModel(f"{weights.source}: no Qwen3.5 vision tower under {root!r}")
        self.root = root

        def m(name: str, shape: tuple[int, int]) -> Matrix:
            found = weights.matrix(root + name, dtype)
            if tuple(found.shape) != shape:
                raise UnsupportedModel(
                    f"{weights.source}: {root + name} is {tuple(found.shape)}, the config says {shape}"
                )
            return found

        def w(name: str, n: int) -> mx.array:
            found = weights.get(root + name, dtype)
            if tuple(found.shape) != (n,):
                raise UnsupportedModel(f"{weights.source}: {root + name} is {tuple(found.shape)}, not ({n},)")
            return found

        d, f, p, t, ch = c.hidden, c.intermediate, c.patch, c.temporal, c.channels
        kernel = weights.get(root + "patch_embed.proj.weight", dtype)
        if tuple(kernel.shape) == (d, t, p, p, ch):  # an MLX build's channels-last Conv3d
            kernel = kernel.transpose(0, 4, 1, 2, 3)
        elif tuple(kernel.shape) != (d, ch, t, p, p):
            raise UnsupportedModel(
                f"{weights.source}: the patch embedding is {tuple(kernel.shape)}, not {(d, ch, t, p, p)} (or "
                f"channels-last {(d, t, p, p, ch)})"
            )
        self.patch = mx.contiguous(kernel.reshape(d, c.patch_dim))
        self.patch_bias = w("patch_embed.proj.bias", d)
        table = weights.get(root + "pos_embed.weight", dtype)
        if tuple(table.shape) != (c.positions, d):
            raise UnsupportedModel(
                f"{weights.source}: the position table is {tuple(table.shape)}, not {(c.positions, d)}"
            )
        # transformers multiplies the table, in the model's dtype, by float32 weights: the table as the model holds it
        self.table = np.array(table.astype(F32))
        self.layers: list[dict[str, Any]] = []
        for i in range(c.depth):
            b = f"blocks.{i}."
            layer: dict[str, Any] = {
                "qkv": m(b + "attn.qkv.weight", (3 * d, d)),
                "qkv_bias": w(b + "attn.qkv.bias", 3 * d),
                "o": m(b + "attn.proj.weight", (d, d)),
                "o_bias": w(b + "attn.proj.bias", d),
                "up": m(b + "mlp.linear_fc1.weight", (f, d)),
                "up_bias": w(b + "mlp.linear_fc1.bias", f),
                "down": m(b + "mlp.linear_fc2.weight", (d, f)),
                "down_bias": w(b + "mlp.linear_fc2.bias", d),
                "norm1": (w(b + "norm1.weight", d), w(b + "norm1.bias", d)),
                "norm2": (w(b + "norm2.weight", d), w(b + "norm2.bias", d)),
            }
            self.layers.append(finish_layer(layer, self.linears, None))
        merged = d * c.merge**2
        self.merger_norm = (w("merger.norm.weight", d), w("merger.norm.bias", d))
        self.merger_up = m("merger.linear_fc1.weight", (merged, merged))
        self.merger_up_bias = w("merger.linear_fc1.bias", merged)
        self.merger_down = m("merger.linear_fc2.weight", (c.out, merged))
        self.merger_down_bias = w("merger.linear_fc2.bias", c.out)
        mx.eval(
            arrays([self.patch, self.patch_bias, *self.merger_norm])
            + arrays([self.merger_up, self.merger_up_bias, self.merger_down, self.merger_down_bias])
        )
        self._grids: OrderedDict[tuple[int, int], tuple[mx.array, mx.array, mx.array]] = OrderedDict()

    @classmethod
    def load(
        cls, weights: Weights, config: Mapping[str, Any], dtype: mx.Dtype = mx.float32, root: str | None = None
    ) -> QwenVision:
        """The vision tower and merger of a Qwen3.5 checkpoint (its config.json and its weights, under `root`)."""
        return cls(VisionConfig.from_hf(config), weights, dtype, root)

    @property
    def hidden_size(self) -> int:
        """The width of the rows it returns: the text model's."""
        return self.config.out

    def grid(self, h: int, w: int) -> tuple[mx.array, mx.array, mx.array]:
        """For an h x w patch grid, in merge-block order: the resampled position table [h * w, hidden] in the
        compute dtype, and the rotary cos and sin [h * w, 1, head_dim] in float32. Kept, by grid, up to
        POSITION_ROWS rows in all."""
        found = self._grids.get((h, w))
        if found is not None:
            self._grids.move_to_end((h, w))
            return found
        cos, sin = rotary_angles(h, w, self.config.head_dim, self.config.theta, self.config.merge)
        found = (
            mx.array(position_embeddings(self.table, h, w, self.config.merge)).astype(self.dtype),
            mx.array(cos)[:, None],
            mx.array(sin)[:, None],
        )
        self._grids[(h, w)] = found
        while len(self._grids) > 1 and sum(a * b for a, b in self._grids) > self.POSITION_ROWS:
            self._grids.popitem(last=False)
        return found

    def __call__(self, images: Sequence[Patches]) -> mx.array:
        """The rows of every image, in order: [sum of image.rows(), out] in the compute dtype."""
        c, ops = self.config, self.ops
        if not images:
            return mx.zeros((0, c.out), self.dtype)
        segments, positions, cosines, sines = [], [], [], []
        for image in images:
            t, h, w = image.grid
            if image.pixels.shape != (t * h * w, c.patch_dim) or h % c.merge or w % c.merge or min(t, h, w) < 1:
                raise ValueError(f"an image of grid {image.grid} has patches {image.pixels.shape}")
            table, cos, sin = self.grid(h, w)
            for _ in range(t):  # each frame attends within itself, as transformers' cu_seqlens split them
                segments.append(h * w)
                positions.append(table)
                cosines.append(cos)
                sines.append(sin)

        def joined(parts: list[mx.array]) -> mx.array:
            return mx.concatenate(parts) if len(parts) > 1 else parts[0]

        pixels = np.concatenate([image.pixels for image in images]) if len(images) > 1 else images[0].pixels
        x = ops.mm_bias(mx.array(pixels).astype(self.dtype), self.patch, self.patch_bias) + joined(positions)
        cos, sin = joined(cosines), joined(sines)
        n, heads, hd = x.shape[0], c.heads, c.head_dim
        spans = list(itertools.pairwise(np.cumsum([0, *segments]).tolist()))
        for layer in self.layers:
            (w1, b1), (w2, b2) = layer["norm1"], layer["norm2"]
            y = layer_norm(x, w1, b1, c.eps)
            qkv = ops.mm_bias(y, layer["qkv"], layer["qkv_bias"]).reshape(n, 3, heads, hd)
            q, k = (self.rotate(qkv[:, j], cos, sin).transpose(1, 0, 2)[None] for j in range(2))
            v = qkv[:, 2].transpose(1, 0, 2)[None]
            outs = [ops.attention(q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], hd**-0.5, None) for s, e in spans]
            o = (outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2))[0]
            x = x + ops.mm_bias(o.transpose(1, 0, 2).reshape(n, c.hidden), layer["o"], layer["o_bias"])
            y = layer_norm(x, w2, b2, c.eps)
            y = activate(ops.mm_bias(y, layer["up"], layer["up_bias"]), c.activation)
            x = x + ops.mm_bias(y, layer["down"], layer["down_bias"])
        x = layer_norm(x, *self.merger_norm, c.eps).reshape(-1, c.hidden * c.merge**2)
        x = activate(ops.mm_bias(x, self.merger_up, self.merger_up_bias), "gelu")
        return ops.mm_bias(x, self.merger_down, self.merger_down_bias)

    @staticmethod
    def rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """rotate_half RoPE of [n, heads, head_dim] in float32 by per-row cos and sin [n, 1, head_dim], rounded back
        to x's dtype (apply_rotary_pos_emb_vision)."""
        y = x.astype(F32)
        half = y.shape[-1] // 2
        turned = mx.concatenate([-y[..., half:], y[..., :half]], axis=-1)
        return (y * cos + turned * sin).astype(x.dtype)


# -- a checkpoint's image and video paths ------------------------------------------------------------------------------

IMAGE_TOKEN = "<|image_pad|>"  # the image token Qwen's processor expands to an image's rows
VIDEO_TOKEN = "<|video_pad|>"  # the video token it expands to a video's frame pairs


class Tower(Protocol):
    """A vision tower: images' patches to their rows, the images' rows concatenated in order."""

    def __call__(self, images: Sequence[Patches], /) -> mx.array: ...


@dataclass
class QwenImages:
    """How one checkpoint reads images, as Qwen's processor and vision tower do: its processor's settings
    (`ImageConfig`, with the creator's pixel range), whether the creator turns an image by its EXIF orientation, the
    image token's id, and `loader`, which builds the vision tower on the first image (its weights stay unread until
    then). `matte`: the colour a transparent image is composited on, for a creator whose loader does that
    (media.load_rgb); None drops the alpha.

    Each image becomes one placeholder token per row (arch/common.media_id), so prompts with different images differ
    token for token; it is decoded with its EXIF turn or without it, as its creator loads it, and its rows counted by
    Qwen's resize rule under the creator's pixel range (`smart_resize`): (height / 32) x (width / 32). `encode` runs
    the vision tower and merger and gives each image's rows shaped by its grid, [frames, block rows, block columns,
    hidden], which the backbone places by (arch/qwen35's multimodal RoPE). The last images' rows are kept, so a
    re-sent image costs no tower pass."""

    config: ImageConfig
    exif: bool
    token: int
    loader: Callable[[], Tower]
    matte: tuple[int, int, int] | None = field(default=None, kw_only=True)
    budget: int | None = field(default=IMAGE_MAX_TOKENS, kw_only=True)
    _kept: Kept = field(default_factory=Kept, init=False, repr=False)  # the last images' rows, by digest
    _counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def attach(
        cls,
        model: ModelDir,
        ctx: LoadContext,
        backbone: Backbone,
        weights: Weights,
        tokenizer: Tokenizer,
        *,
        exif: bool,
        pixels: tuple[int, int] | None = None,
        matte: tuple[int, int, int] | None = None,
    ) -> QwenImages | None:
        """The checkpoint's image path, or None for a build without one (no processor_config.json or
        preprocessor_config.json, no vision_config, or no vision tower among its `weights`, or for a GGUF build in its
        mmproj file: a text-only conversion, or a GGUF file without its mmproj beside it). `pixels`, (floor, ceiling),
        replaces the processor's pixel range, as a creator's code that sets it does. UnsupportedModel when the
        tokenizer's image token is not config.json's."""
        config = model.config
        found = next((f for f in ("processor_config.json", "preprocessor_config.json") if model.has(f)), None)
        if found is None or not isinstance(config.get("vision_config"), dict):
            return None
        if not any(ctx.has_media(model, weights, root + "patch_embed.proj.weight") for root in ROOTS):
            return None
        image = ImageConfig.from_hf(model.json(found))
        if pixels is not None:
            image = image.pixels(*pixels)
        VisionConfig.from_hf(config).check(image)
        token = config_token(model, tokenizer, IMAGE_TOKEN, "image_token_id")
        return cls(image, exif, token, encoder_loader(QwenVision.load, model, ctx, backbone), matte=matte)

    def count_size(self, width: int, height: int) -> int:
        """The rows a width x height image becomes, from its header dimensions alone."""
        return image_tokens(width, height, self.config)

    def count(self, rgb: np.ndarray) -> int:
        """The rows a decoded image becomes."""
        return self.count_size(rgb.shape[1], rgb.shape[0])

    def prepare(self, rgb: np.ndarray) -> np.ndarray:
        """Resize a decoded image to the size the vision tower reads, so full-resolution camera frames are not kept
        in memory across all images of a request."""
        height, width = rgb.shape[:2]
        h, w = smart_resize(height, width, self.config.factor, self.config.min_pixels, self.config.max_pixels)
        if (h, w) == (height, width):
            return rgb
        resized: np.ndarray = resize_frames(rgb[None], h, w, RESAMPLE[self.config.resample])[0]
        return resized

    def patches(self, rgb: np.ndarray) -> Patches:
        """A decoded image as the tower takes it."""
        return preprocess(rgb, self.config)

    def rows(self, images: Sequence[Media]) -> tuple[list[tuple[int, ...]], dict[int, np.ndarray]]:
        """Each image's placeholders, in order, and the resized pixels decoded to count them (by digest, for `encode`).
        Checks the encoded vision-token budget from headers before decoding pixels, and resizes each decoded image
        immediately so camera-sized frames do not accumulate in RAM. An image seen before is counted from its bytes'
        digest without decoding it again."""
        counts: list[int] = []
        decode_needed: list[bool] = []
        for image in images:
            n = self._counts.get(image.sha256)
            if n is None:
                width, height = image_size(image)
                n = self.count_size(width, height)
                decode_needed.append(True)
            else:
                decode_needed.append(False)
            counts.append(n)
        if self.budget is not None:
            # Never refuse a single image that fits the model's own processor max_pixels (e.g. Clef's 16 MP single-page
            # scans up to 16,384 tokens), while keeping multi-image totals bounded (8,192 on d3/d1, 16,384 on 16 MP).
            one_image = self.config.max_pixels // (self.config.factor * self.config.factor)
            limit = max(self.budget, one_image)
            total = sum(counts)
            if total > limit:
                raise ContextTooLong(
                    f"images: {total:,} vision tokens across {len(images)} images, over the {limit:,} tokens "
                    "a request may spend on images; send fewer or smaller images"
                )
        out: list[tuple[int, ...]] = []
        pixels: dict[int, np.ndarray] = {}
        for image, n, needs_decode in zip(images, counts, decode_needed, strict=True):
            digest = media_digest(image.sha256)
            if needs_decode:
                if digest not in pixels:
                    rgb = load_rgb(image, exif=self.exif, matte=self.matte)
                    prepared = self.prepare(rgb)
                    n = self.count(prepared)
                    pixels[digest] = prepared
                if len(self._counts) >= 4096:
                    self._counts.clear()  # counts are cheap to redo; this only bounds the memory
                self._counts[image.sha256] = n
            out.append(tuple(media_id(digest, row) for row in range(n)))
        return out, pixels

    def expand(self, ids: Sequence[int], images: Sequence[tuple[int, ...]]) -> list[int]:
        """`ids` with each image token replaced by its image's placeholders, in order, as the processor expands
        them (arch/common.expand)."""
        return expand(ids, self.token, images)

    def encode(self, renderings: Sequence[Rendering]) -> dict[int, mx.array]:
        """Every image of the renderings' requests, by digest: its rows from the vision tower, [frames, block rows,
        block columns, hidden]; the images not kept from before through the tower in one call."""
        needed: dict[int, Any] = {}
        for rendering in renderings:
            for image in rendering.request.images:
                digest = media_digest(image.sha256)
                needed.setdefault(digest, rendering.pixels.get(digest, image))
        return encode_kept(self._kept, needed, self._through_tower)

    def _through_tower(self, new: dict[int, Any]) -> dict[int, dict[int, mx.array]]:
        """The rows of images `new` (their decoded pixels, or the media to decode) through the tower, chunked by a
        patch budget so many images do not hold all their vision activations in Metal memory at once."""
        patches = {
            digest: self.patches(
                source if isinstance(source, np.ndarray) else load_rgb(source, exif=self.exif, matte=self.matte)
            )
            for digest, source in new.items()
        }
        tower = self.loader()
        merge = self.config.merge
        out: dict[int, dict[int, mx.array]] = {}
        chunk: list[tuple[int, Patches]] = []
        chunk_patches = 0

        def flush() -> None:
            nonlocal chunk_patches
            if not chunk:
                return
            encoded = tower([found for _, found in chunk])
            at = 0
            built: list[mx.array] = []
            for digest, found in chunk:
                t, h, w = found.grid
                n = found.rows(merge)
                # a copy, so a kept image's rows never keep the others' alive
                rows = mx.contiguous(encoded[at : at + n].reshape(t, h // merge, w // merge, encoded.shape[-1]))
                out[digest] = {digest: rows}
                built.append(rows)
                at += n
            mx.eval(*built)
            chunk.clear()
            chunk_patches = 0

        for digest, found in patches.items():
            count = int(found.pixels.shape[0])
            if chunk and chunk_patches + count > 4_096:
                flush()
            chunk.append((digest, found))
            chunk_patches += count
        flush()
        return out


def pair_digest(sha256: str, pair: int) -> int:
    """What the placeholders of a video's frame pair `pair` name it by: each pair is a medium of its own."""
    return media_digest(hashlib.sha256(f"{sha256}/{pair}".encode()).hexdigest())


def frame_pair_times(clip: Clip, temporal: int) -> tuple[float, ...]:
    """Each frame pair's time in seconds, as Qwen3VLProcessor gives it (media/video.timestamps)."""
    return tuple(timestamps(clip.indices, clip.fps, temporal))


@dataclass
class QwenVideos:
    """How one checkpoint reads videos, as Qwen3VLProcessor does: its video processor's settings (`VideoConfig`, with
    the creator's pixel budget and per-frame cap), how the creator samples frames and decodes them (`decoder`, the
    creator's own decoder), the video token's id, `images`, whose vision tower reads the frames, the most rows a
    request's videos may take (`budget`, None: no limit), the video token's text (`pad`) and each frame pair's time
    (`times`, Qwen's unless a creator computes it otherwise).

    The chat template's video goes after the images, and the processor replaces its video token with every frame
    pair's `<t seconds><|vision_start|>` + its rows + `<|vision_end|>` (t the mean of the pair's frames' times, one
    decimal), so a video's frame pairs are separated by text. Each frame pair is a medium of its own (its
    placeholders named by `pair_digest`), shaped [1, block rows, block columns, hidden], since the text model places
    each pair as an image (Qwen3_5Model.get_rope_index splits a video's grid by frame pair)."""

    config: VideoConfig
    sampling: Sampling
    token: int
    images: QwenImages
    budget: int | None = None
    decoder: Callable[[Media, float, int, int, Shrink | None], Clip] = field(kw_only=True)
    pad: str = field(default=VIDEO_TOKEN, kw_only=True)
    times: Callable[[Clip, int], tuple[float, ...]] = field(default=frame_pair_times, kw_only=True)
    _kept: Kept = field(default_factory=Kept, init=False, repr=False)  # each video's frame pairs' rows, by sha256
    _seen: dict[str, Seen] = field(default_factory=dict, init=False, repr=False)

    def decode(self, video: Media) -> Clip:
        """A video's sampled frames, as the creator decodes them, each resized as it is decoded (`shrink`)."""
        s = self.sampling
        return self.decoder(video, s.rate, s.least, s.most, self.shrink)

    def shrink(self, frame: np.ndarray, count: int) -> np.ndarray:
        """One decoded frame [H, W, 3] of a clip of `count` frames, resized as `preprocess_video` resizes each frame
        of it: the same resize of the same frame, done as it is decoded, so a clip holds its frames at the size the
        tower reads, not the source's (768 4K frames are 19 GB)."""
        if count < self.config.image.temporal:  # too short for the processor, which says so (`video_size`)
            return frame
        h, w = video_size(count, frame.shape[0], frame.shape[1], self.config)
        if (h, w) == frame.shape[:2]:
            return frame
        resized: np.ndarray = resize_frames(frame[None], h, w, RESAMPLE[self.config.image.resample])[0]
        return resized

    def seen(self, videos: Sequence[Media]) -> tuple[list[Seen], dict[str, Clip]]:
        """Each video's frame pairs' times and its grid, and the clips decoded to find them (by sha256, for
        `encode`). A video seen before is not decoded again. InvalidRequest when the videos take more than `budget`
        rows."""
        out: list[Seen] = []
        clips: dict[str, Clip] = {}
        for video in videos:
            found = self._seen.get(video.sha256)
            if found is None:
                clip = clips.get(video.sha256) or self.decode(video)
                clips[video.sha256] = clip
                frames, height, width = clip.frames.shape[:3]
                grid = video_grid(frames, height, width, self.config)
                duration_s = round(clip.total / clip.fps, 1) if clip.fps > 0 else 0.0
                found = Seen(
                    self.times(clip, self.config.image.temporal),
                    grid,
                    frames=len(set(clip.indices)),
                    duration_s=duration_s,
                )
                if len(self._seen) >= 1024:
                    self._seen.clear()  # cheap to redo; this only bounds the memory
                self._seen[video.sha256] = found
            out.append(found)
        if self.budget is not None:
            merge = self.config.image.merge**2
            rows = sum(s.grid[0] * s.grid[1] * s.grid[2] // merge for s in out)
            if rows > self.budget:
                raise InvalidRequest(
                    f"the videos take {rows:,} input tokens, over the {self.budget:,} tokens a request may spend on "
                    "videos; send fewer or shorter videos"
                )
        return out, clips

    def summary(self, seen: Sequence[Seen]) -> tuple[dict[str, Any], ...]:
        """Per-video sampling metadata for responses: how many distinct frames were read across the clip's duration,
        and whether sampling hit this model's frame cap."""
        out: list[dict[str, Any]] = []
        for item in seen:
            capped = item.duration_s * self.sampling.rate > self.sampling.most
            entry: dict[str, Any] = {
                "frames_read": item.frames,
                "duration_s": item.duration_s,
                "capped": capped,
            }
            if item.frames > 0 and item.duration_s > 0:
                entry["seconds_per_frame"] = round(item.duration_s / item.frames, 1)
            out.append(entry)
        return tuple(out)

    def text(self, seen: Seen) -> str:
        """What a video's token becomes in the prompt's text: each frame pair's time and its one video token
        (expanded to the pair's rows after tokenizing)."""
        return "".join(f"<{t:.1f} seconds><|vision_start|>{self.pad}<|vision_end|>" for t in seen.times)

    def rows(self, video: Media, seen: Seen) -> list[tuple[int, ...]]:
        """Each frame pair's placeholders, in order."""
        t, h, w = seen.grid
        n = (h // self.config.image.merge) * (w // self.config.image.merge)
        return [tuple(media_id(pair_digest(video.sha256, i), r) for r in range(n)) for i in range(t)]

    def encode(self, renderings: Sequence[Rendering]) -> dict[int, mx.array]:
        """Every frame pair of the renderings' requests' videos, by its digest: its rows from the vision tower,
        [1, block rows, block columns, hidden]; the videos with a pair not kept from before through the tower in
        one call."""
        needed: dict[str, tuple[Media, Clip | None]] = {}
        for rendering in renderings:
            for video in rendering.request.videos:
                needed.setdefault(video.sha256, (video, rendering.clips.get(video.sha256)))
        return encode_kept(self._kept, needed, self._through_tower)

    def _through_tower(self, new: dict[str, tuple[Media, Clip | None]]) -> dict[str, dict[int, mx.array]]:
        """The frame pairs' rows of videos `new` through the tower in one call."""
        patches = {
            sha256: preprocess_video((clip or self.decode(video)).frames, self.config)
            for sha256, (video, clip) in new.items()
        }
        encoded = self.images.loader()(list(patches.values()))
        out: dict[str, dict[int, mx.array]] = {}
        at, merge = 0, self.config.image.merge
        for sha256, found in patches.items():
            t, h, w = found.grid
            n = found.rows(merge)
            rows = encoded[at : at + n].reshape(t, 1, h // merge, w // merge, encoded.shape[-1])
            # copies, so a kept video's pairs never keep the others' rows alive
            out[sha256] = {pair_digest(sha256, i): mx.contiguous(rows[i]) for i in range(t)}
            at += n
        return out
