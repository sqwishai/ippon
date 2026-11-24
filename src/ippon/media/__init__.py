"""Images and audio in requests (`decode`), torchvision's resizes in ippon's native library (`resize`), what every
media path shares (`paths`, `video`), and each model's own media pipelines (`qwen`, `lfm3`, `lfm3_audio`, and the
video decoders `d3_video` and `lfm3_video`), which import MLX or their decoder and are imported where they are used.

`decode` and `resize` need numpy and the native library only, so the schema can check media where MLX doesn't load.
"""

from __future__ import annotations

from .decode import (
    AUDIO_FORMATS,
    IMAGE_FORMATS,
    MAX_IMAGES,
    MAX_MEDIA_BYTES,
    Media,
    MediaKind,
    as_media,
    check_images,
    image_size,
    load_rgb,
    parse_media,
    sniff,
)
from .resize import resize_frames

__all__ = [
    "AUDIO_FORMATS",
    "IMAGE_FORMATS",
    "MAX_IMAGES",
    "MAX_MEDIA_BYTES",
    "Media",
    "MediaKind",
    "as_media",
    "check_images",
    "image_size",
    "load_rgb",
    "parse_media",
    "resize_frames",
    "sniff",
]
