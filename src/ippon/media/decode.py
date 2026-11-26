"""Images, audio clips and videos in requests: their bytes, checked once where they enter, and decoded when a model
reads them.

Over JSON a medium is a string: standard base64, as Ollama sends `images` (a Go []byte), or a data URI
(`data:image/png;base64,...`), as llama.cpp's server and Perplexity's API take. `parse_media` decodes it strictly
(the standard alphabet, padded, nothing else), caps its size, and names its format from its first bytes; it never
fetches a URL or reads a path, as llama.cpp's server refuses them. `as_media` takes what Python callers have: bytes,
a path, a PIL image, a numpy array.

A video is a clip's file (MP4, QuickTime, WebM or Matroska), named from its container's first bytes; its frames are
decoded by the model's own decoder when it reads it (media/video.py).

Pixels are decoded only when a model reads the image (`load_rgb`), by Pillow, the decoder the creators' own code uses,
so a JPEG decodes to the same pixels as theirs; it is imported then, so the CLI's checks run without it. A WAV clip's
samples are read from its chunks as stored (`wav`), and compressed audio is decoded by the native audio helper.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import io
import os
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..errors import InvalidRequest

if TYPE_CHECKING:
    import numpy as np

MediaKind = Literal["image", "audio", "video"]
MAX_MEDIA_BYTES = 1 << 30  # decoded bytes per image, clip or video sent over HTTP or MCP; local files have no cap
MAX_IMAGES = 12  # images in one request at most; encoded vision tokens are bounded per model in QwenImages
IMAGE_FORMATS = ("png", "jpeg", "webp", "gif", "bmp", "tiff", "avif")  # what a request may send
# a recording as a Mac or a phone saves it: decoded natively (native/src/audio.rs), a WAV of 16-bit PCM or float32 as
# Liquid's code reads one
AUDIO_FORMATS = ("wav", "mp3", "aac", "flac", "ogg", "aiff", "caf", "mp4", "quicktime", "webm", "matroska")
# the containers a request's video may be, for every model that reads videos: those both video decoders read (OpenCV's
# and PyAV's, media/d3_video.py and media/lfm3_video.py), which are vllm-sr d3's server's four
VIDEO_FORMATS = ("mp4", "quicktime", "webm", "matroska")
# the rate a caller's audio array is taken to be at (`as_media`), written into the WAV it becomes: the rate of the one
# model that hears audio (d1-Omni-2B, whose creator never converts an array)
ARRAY_RATE = 16_000
PILLOW_FORMATS = {
    "png": "PNG",
    "jpeg": "JPEG",
    "webp": "WEBP",
    "gif": "GIF",
    "bmp": "BMP",
    "tiff": "TIFF",
    "avif": "AVIF",
}
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_PPM = re.compile(rb"P6\n(\d+) (\d+)\n255\n")
WHAT = {"image": "an image", "audio": "an audio clip", "video": "a video"}
_NAMES = {
    "png": "PNG",
    "jpeg": "JPEG",
    "webp": "WebP",
    "gif": "GIF",
    "bmp": "BMP",
    "tiff": "TIFF",
    "avif": "AVIF",
    "wav": "WAV",
}
_NAMES |= {"mp4": "MP4", "quicktime": "QuickTime", "webm": "WebM", "matroska": "Matroska"}
_NAMES |= {"mp3": "MP3", "aac": "AAC", "flac": "FLAC", "ogg": "Ogg", "aiff": "AIFF", "caf": "CAF"}
AUDIO_ONLY = ("wav", "mp3", "aac", "flac", "ogg", "aiff", "caf")  # never an image or a video
AUDIO_HELP = "WAV, MP3, M4A or AAC, FLAC, Ogg Vorbis, AIFF or CAF"
# the major brands of HEIF stills (an iPhone's photos), whose ftyp box is the one MP4 starts with too
_HEIF_BRANDS = (b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1")
_QUICKTIME_ATOMS = (b"moov", b"mdat", b"wide", b"pnot")  # a QuickTime file older than the ftyp box starts with one


@dataclass(frozen=True)
class Media:
    """One image, audio clip or video as sent: its bytes, their sha256 (what caches of its embeddings are keyed by)
    and its format, named from its first bytes. `format` is one of IMAGE_FORMATS, AUDIO_FORMATS or VIDEO_FORMATS, or
    "ppm" for pixels a Python caller passed as an array (binary PPM, which needs no decoder)."""

    kind: MediaKind
    data: bytes = field(repr=False)
    sha256: str
    format: str

    @classmethod
    def of(cls, data: bytes, kind: MediaKind, name: str | None = None, limit: int | None = MAX_MEDIA_BYTES) -> Media:
        """`data` as a medium of `kind`: InvalidRequest when it is empty, over `limit` bytes (None: no limit), or not
        a format of that kind."""
        where = name or kind
        if not data:
            raise InvalidRequest(f"{where}: empty")
        if limit is not None and len(data) > limit:
            raise InvalidRequest(
                f"{where}: {len(data):,} bytes, more than the {limit:,} a medium sent in a request may be: send a "
                "smaller one"
            )
        found = sniff(data)
        accepted = {"image": (*IMAGE_FORMATS, "ppm"), "audio": AUDIO_FORMATS, "video": VIDEO_FORMATS}[kind]
        if found not in accepted:
            if kind == "video":
                what = f"a {_NAMES[found]} file, not a video" if found in _NAMES else "not a video ippon reads"
                raise InvalidRequest(f"{where}: {what} (MP4, QuickTime, WebM or Matroska)")
            if found == "heic":
                raise InvalidRequest(
                    f"{where}: a HEIC photo, which ippon doesn't read: send it as JPEG, as "
                    "`sips -s format jpeg photo.heic --out photo.jpg` converts it on a Mac"
                )
            if kind == "image" and data.startswith(b"%PDF-"):
                raise InvalidRequest(
                    f"{where}: a PDF, which ippon doesn't read: send each page as an image, as "
                    "`sips -s format png scan.pdf --out page.png` turns its first page into one on a Mac"
                )
            if kind == "image":
                what = (
                    f"a {_NAMES[found]} clip, not an image"
                    if found in AUDIO_ONLY
                    else "not an image in a format ippon reads"
                )
                raise InvalidRequest(f"{where}: {what} (PNG, JPEG, WebP, GIF, BMP, TIFF or AVIF)")
            what = f"a {_NAMES.get(str(found), str(found))} image, not a clip" if found else "not an audio file"
            raise InvalidRequest(f"{where}: {what} (audio is {AUDIO_HELP})")
        return cls(kind, data, hashlib.sha256(data).hexdigest(), str(found))


def sniff(data: bytes) -> str | None:
    """The format whose magic bytes `data` starts with, None when it is none ippon reads."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data.startswith(b"BM"):
        return "bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if _PPM.match(data[:32]):
        return "ppm"
    if data.startswith(b"ID3") or (data[:1] == b"\xff" and len(data) > 1 and data[1] & 0xF6 == 0xF0):
        return "mp3" if data.startswith(b"ID3") else "aac"  # an ID3 tag, or AAC's ADTS sync word
    if data[:1] == b"\xff" and len(data) > 1 and data[1] & 0xE0 == 0xE0:
        return "mp3"  # an MPEG audio frame's sync word
    if data.startswith(b"fLaC"):
        return "flac"
    if data.startswith(b"OggS"):
        return "ogg"
    if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if data.startswith(b"caff"):
        return "caf"
    if data[4:8] == b"ftyp":  # ISO base media: MP4, QuickTime, an AVIF or a HEIF still, by its major brand
        # an AVIF may name the generic still image brand (mif1) first and itself among the compatible brands after
        compatible = data[16 : min(int.from_bytes(data[:4], "big"), 64)]
        if data[8:12] in (b"avif", b"avis") or any(
            compatible[i : i + 4] in (b"avif", b"avis") for i in range(0, len(compatible) - 3, 4)
        ):
            return "avif"
        if data[8:12] in _HEIF_BRANDS:
            return "heic"
        return "quicktime" if data[8:12] == b"qt  " else "mp4"
    if data[4:8] in _QUICKTIME_ATOMS:
        return "quicktime"
    if data.startswith(b"\x1a\x45\xdf\xa3"):  # EBML: WebM or Matroska by the DocType in its header
        return "webm" if b"\x42\x82\x84webm" in data[:64] else "matroska"
    return None


def parse_media(value: object, kind: MediaKind, name: str | None = None) -> Media:
    """A medium from a request's JSON value: a standard base64 string or a `data:<type>/...;base64,` URI of `kind`
    ("image", "audio" or "video"). `name` (e.g. "images[0]") starts its error messages. InvalidRequest for anything
    else: another JSON type, a URL, a path, base64 that is not strict, a data URI of another type or not in base64,
    more than MAX_MEDIA_BYTES, or a format ippon doesn't read."""
    where = name or kind
    if not isinstance(value, str):
        raise InvalidRequest(f"{where}: {WHAT[kind]} is a base64 string or a data: URI")
    payload = value
    if value[:5].lower() == "data:":
        header, comma, payload = value[5:].partition(",")
        mime, *params = (p.strip().lower() for p in header.split(";"))
        if not comma or "base64" not in params:
            raise InvalidRequest(f"{where}: only base64 data URIs are read (data:{kind}/...;base64,...)")
        if not mime.startswith(kind + "/"):
            raise InvalidRequest(f"{where}: a data URI of type {mime or 'none'!r} where {WHAT[kind]} goes")
    elif _URL.match(value):
        raise InvalidRequest(f"{where}: URLs are not fetched; send the {kind} itself, in base64 or as a data: URI")
    if not payload:
        raise InvalidRequest(f"{where}: empty")
    if len(payload) > 4 * -(-MAX_MEDIA_BYTES // 3):  # checked before decoding, so a huge string costs nothing
        raise InvalidRequest(f"{where}: more than the {MAX_MEDIA_BYTES >> 20} MiB a medium sent in a request may be")
    try:  # validate=True refuses any character outside the standard alphabet, line breaks and bad padding included
        data = base64.b64decode(payload, validate=True) if len(payload) % 4 == 0 else None
    except binascii.Error:
        data = None
    if data is None:
        if payload is value and (value.startswith(("/", "~", ".", "\\")) or re.match(r"[A-Za-z]:[\\/]", value)):
            raise InvalidRequest(
                f"{where}: paths are not read from requests; send the {kind} itself, in base64 or as a data: URI"
            )
        raise InvalidRequest(f"{where}: not valid base64 (the standard alphabet, padded, without line breaks)")
    media = Media.of(data, kind, where)
    if media.format == "ppm":  # as_media's form for arrays, not one a request sends
        raise InvalidRequest(f"{where}: not an image in a format ippon reads (PNG, JPEG, WebP, GIF, BMP, TIFF or AVIF)")
    return media


def as_media(obj: Any, kind: MediaKind) -> Media:
    """A medium from what a Python caller has. An image: bytes of a file, a path to one, a data: URI, a PIL image
    (as the creators' code takes one: its pixels as they are, turned by its EXIF orientation where the
    creator's code turns a picture) or an H x W x 3 uint8 array.
    A clip: bytes or a path of a WAV file, or a 1-D numpy array of 16 kHz mono samples, int16 or float (the creator's
    own input; floats are kept as float32, as its code reads them). URLs are never fetched. Nothing here is capped
    by size: the caller's own files and arrays are as large as the model can read (a long video's frames are
    sampled).
    A video: bytes or a path of its file."""
    if isinstance(obj, Media):
        if obj.kind != kind:
            raise InvalidRequest(f"{WHAT[obj.kind]} where {WHAT[kind]} goes")
        return obj
    if isinstance(obj, bytes | bytearray | memoryview):
        return Media.of(bytes(obj), kind, limit=None)
    if isinstance(obj, str) and (obj[:5].lower() == "data:" or _URL.match(obj)):
        return parse_media(obj, kind)
    if isinstance(obj, str | os.PathLike):
        path = Path(obj).expanduser()
        if not path.is_file():
            raise InvalidRequest(f"{path}: no such file")
        return Media.of(path.read_bytes(), kind, str(path), limit=None)
    if kind == "image" and type(obj).__module__.startswith("PIL.") and hasattr(obj, "convert"):
        if obj.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I;16"):  # none PNG holds: as RGB, as before
            import numpy as np

            return Media.of(_ppm(np.asarray(obj.convert("RGB"))), kind, limit=None)
        png = io.BytesIO()  # its EXIF kept, so a model whose creator turns a picture by it (d1, h2o) turns this one
        obj.save(png, "PNG", exif=obj.getexif(), compress_level=1)
        return Media.of(png.getvalue(), kind, limit=None)
    if kind != "video":
        import numpy as np

        if isinstance(obj, np.ndarray):
            return Media.of(_ppm(obj) if kind == "image" else _wav(obj), kind, limit=None)
    raise InvalidRequest(f"{WHAT[kind]} can't be a {type(obj).__name__}")


def _ppm(pixels: np.ndarray) -> bytes:
    import numpy as np

    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or 0 in pixels.shape:
        raise InvalidRequest(f"an image array is H x W x 3 uint8 RGB, not {pixels.dtype} {pixels.shape}")
    height, width = pixels.shape[:2]
    return b"P6\n%d %d\n255\n" % (width, height) + np.ascontiguousarray(pixels).tobytes()


def _wav(samples: np.ndarray) -> bytes:
    """16 kHz mono WAV of `samples`: PCM16 for int16, IEEE float32 (format 3) for floats."""
    import numpy as np

    if samples.ndim != 1 or samples.size == 0:
        raise InvalidRequest(f"audio is mono: a 1-D array of 16 kHz samples, not {samples.dtype} {samples.shape}")
    if samples.dtype == np.int16:
        code, width, data = 1, 2, samples.astype("<i2").tobytes()
    elif np.issubdtype(samples.dtype, np.floating):
        code, width, data = 3, 4, samples.astype("<f4").tobytes()
    else:
        raise InvalidRequest(f"audio samples are int16 or floats, not {samples.dtype}")
    fmt = struct.pack("<HHIIHH", code, 1, ARRAY_RATE, ARRAY_RATE * width, width, 8 * width)
    body = b"WAVE" + _chunk(b"fmt ", fmt) + _chunk(b"data", data)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _chunk(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack("<I", len(body)) + body + b"\x00" * (len(body) % 2)


_UNKNOWN_SIZE = 0xFFFFFFFF  # a streaming writer's size for a chunk whose length it could not go back and fill in


def _chunks(data: bytes) -> dict[bytes, bytes]:
    """A RIFF/WAVE file's chunks by tag (the last of a tag kept), as `_chunk` writes them: InvalidRequest when it is
    not one or a chunk is cut short."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise InvalidRequest("not a WAV clip (RIFF/WAVE)")
    chunks, at = {}, 12
    while at + 8 <= len(data):
        tag, size = data[at : at + 4], struct.unpack_from("<I", data, at + 4)[0]
        start = at + 8
        if size == _UNKNOWN_SIZE:  # written to a pipe (FFmpeg's `-f wav pipe:1`): the chunk runs to the file's end
            size = len(data) - start
        if start + size > len(data):
            raise InvalidRequest(f"the WAV clip's {tag.decode('latin-1')!r} chunk is cut short")
        chunks[tag] = data[start : start + size]
        at = start + size + (size & 1)
    return chunks


_EXTENSIBLE, _PCM, _FLOAT = 0xFFFE, 1, 3
_GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"  # KSDATAFORMAT_SUBTYPE_* after the format code


def wav_sample(media: Media) -> Literal["pcm16", "float32"] | None:
    """A WAV clip's sample format as `wav` reads it, PCM16 or IEEE float32 (either in WAVE_FORMAT_EXTENSIBLE too),
    or None for another one (24-bit PCM, 64-bit float), which only a decoder of its own reads. InvalidRequest for a
    file that is not a WAV clip or has no format chunk."""
    _, _, bits, code = _format(media)
    return _sample(code, bits)


def _sample(code: int, bits: int) -> Literal["pcm16", "float32"] | None:
    return "pcm16" if (code, bits) == (_PCM, 16) else "float32" if (code, bits) == (_FLOAT, 32) else None


def _format(media: Media) -> tuple[dict[bytes, bytes], int, int, int]:
    """A WAV clip's chunks, channels, bits and format code (the subformat's for WAVE_FORMAT_EXTENSIBLE)."""
    if media.kind != "audio":
        raise InvalidRequest(f"{WHAT[media.kind]} is not an audio clip")
    chunks = _chunks(media.data)
    fmt = chunks.get(b"fmt ")
    if fmt is None or len(fmt) < 16 or b"data" not in chunks:
        raise InvalidRequest("the WAV clip has no format or no data chunk")
    code, channels, _, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
    if code == _EXTENSIBLE and len(fmt) >= 40 and fmt[26:40] == _GUID_TAIL:
        code = struct.unpack_from("<H", fmt, 24)[0]  # the subformat GUID starts with the format code
    return chunks, channels, bits, code


def wav(media: Media) -> tuple[np.ndarray, int, str]:
    """A WAV clip as it is stored: its samples [frames, channels] (int16 for PCM16, float32 for IEEE float32, either
    in WAVE_FORMAT_EXTENSIBLE too), its rate and its sample format ("pcm16", "float32"). InvalidRequest for another
    sample format (`wav_sample`), a chunk cut short, no samples, or a float sample that is not finite."""
    import numpy as np

    chunks, channels, bits, code = _format(media)
    rate = struct.unpack_from("<I", chunks[b"fmt "], 4)[0]
    body = chunks[b"data"]
    sample = _sample(code, bits)
    dtype: np.dtype[Any]
    if sample == "pcm16":
        dtype = np.dtype("<i2")
    elif sample == "float32":
        dtype = np.dtype("<f4")
    else:
        what = "PCM" if code == _PCM else "float" if code == _FLOAT else f"format {code:#x}"
        raise InvalidRequest(f"audio is PCM16 or float32 WAV, not {bits}-bit {what}")
    if channels < 1 or not body or len(body) % (dtype.itemsize * channels):
        raise InvalidRequest("the WAV clip has no samples" if not body else "the WAV clip's data ends mid-sample")
    samples = np.frombuffer(body, dtype)
    if dtype.kind == "f" and not np.isfinite(samples).all():
        raise InvalidRequest("the WAV clip has samples that are not finite numbers")
    out: np.ndarray = samples.astype(dtype.newbyteorder("=")).reshape(-1, channels)
    return out, rate, sample


def pillow() -> tuple[Any, Any]:
    """Pillow's Image and ImageOps modules, imported when a model first reads an image."""
    return importlib.import_module("PIL.Image"), importlib.import_module("PIL.ImageOps")


def image_size(media: Media) -> tuple[int, int]:
    """An image's (width, height) from its header, without decoding its pixels. InvalidRequest when Pillow cannot
    read the header or the image passes Pillow's decompression-bomb bound."""
    if media.kind != "image":
        raise InvalidRequest(f"{WHAT[media.kind]} is not an image")
    if media.format == "ppm":
        match = _PPM.match(media.data)
        assert match is not None, "a PPM medium is made by as_media"
        return int(match[1]), int(match[2])
    image_module, _ = pillow()
    try:
        with image_module.open(io.BytesIO(media.data), formats=[PILLOW_FORMATS[media.format]]) as image:
            width, height = image.size
            return int(width), int(height)
    except (OSError, ValueError, SyntaxError, RuntimeError, image_module.DecompressionBombError) as error:
        raise InvalidRequest(f"the {_NAMES[media.format]} image could not be decoded: {error}") from None


def check_images(images: Sequence[Media]) -> None:
    """Refuses a request with more than MAX_IMAGES images or an unreadable/bomb image header, without imposing a
    source-pixel sum cap (ordinary multi-photo phone sets downscale in the vision processor, where encoded vision
    tokens are budgeted instead)."""
    if not images:
        return
    if len(images) > MAX_IMAGES:
        raise InvalidRequest(f"images: {len(images)} images in one request; at most {MAX_IMAGES}")
    for image in images:
        image_size(image)


def load_rgb(media: Media, exif: bool, matte: tuple[int, int, int] | None = None) -> np.ndarray:
    """The image's pixels, H x W x 3 uint8 RGB, decoded as the model's creator decodes it: with `exif`, turned by
    its EXIF orientation first (ImageOps.exif_transpose), as transformers' load_image does for d1-omni and Clef;
    then convert("RGB"), as most creators do (alpha dropped, palettes and grayscale expanded, the first frame of an
    animation). With `matte`, an image other than RGB with transparency (RGBA, LA, PA, or a transparent colour) is
    composited on that colour first, as vLLM's image loading does (vllm/multimodal/image.py, convert_image_mode).
    InvalidRequest when Pillow can't decode it."""
    if media.kind != "image":
        raise InvalidRequest(f"{WHAT[media.kind]} is not an image")
    import numpy as np

    if media.format == "ppm":
        m = _PPM.match(media.data)
        assert m is not None, "a PPM medium is made by as_media"
        width, height = int(m[1]), int(m[2])
        raw = np.frombuffer(media.data, np.uint8, offset=m.end()).reshape(height, width, 3)
        copied: np.ndarray = raw.copy()
        return copied
    image_module, ops = pillow()
    try:
        with image_module.open(io.BytesIO(media.data), formats=[PILLOW_FORMATS[media.format]]) as image:
            turned = ops.exif_transpose(image) if exif else image
            clear = turned.mode in ("RGBA", "LA", "PA") or "transparency" in turned.info
            if matte is not None and turned.mode != "RGB" and clear:
                rgba = turned.convert("RGBA")
                turned = image_module.new("RGB", rgba.size, matte)
                turned.paste(rgba, mask=rgba.split()[3])
            pixels: np.ndarray = np.array(turned.convert("RGB"))
            return pixels
    except (OSError, ValueError, SyntaxError, RuntimeError, image_module.DecompressionBombError) as error:
        raise InvalidRequest(f"the {_NAMES[media.format]} image could not be decoded: {error}") from None
