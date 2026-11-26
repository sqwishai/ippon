"""media/decode.py: media in requests (base64, data URIs, what is refused and how), media from Python values, format
sniffing on real files, and Pillow decoding against what transformers' load_image and plain Pillow give in the
creators' venv (tests/data/media/make_decode_fixtures.py)."""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers import b64
from ippon import native
from ippon.errors import InvalidRequest
from ippon.media import Media, as_media, decode, load_rgb, parse_media

DATA = Path(__file__).resolve().parent / "data/media"
DECODED = json.loads((DATA / "decode.json").read_text())
PNG = (DATA / "images/plain.png").read_bytes()
JPEG = (DATA / "images/plain.jpg").read_bytes()


# what follows the format code in a WAVE_FORMAT_EXTENSIBLE header's subformat GUID (KSDATAFORMAT_SUBTYPE_*)
GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def wav(samples: np.ndarray) -> bytes:
    return as_media(samples, "audio").data


def test_base64_and_data_uris_give_the_same_medium() -> None:
    plain = parse_media(b64(PNG), "image", "images[0]")
    uri = parse_media("data:image/png;base64," + b64(PNG), "image")
    other_type = parse_media("data:image/jpeg;base64," + b64(PNG), "image")  # the bytes decide the format
    assert plain == uri == other_type
    assert plain.format == "png"
    assert plain.kind == "image"
    assert plain.data == PNG
    assert plain.sha256 == hashlib.sha256(PNG).hexdigest()
    assert "data=" not in repr(plain)  # bytes stay out of logs


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (123, "a base64 string or a data: URI"),
        ("", "empty"),
        ("file:///tmp/cat.png", "URLs are not fetched"),
        (str(Path("/") / "Users" / "someone" / "cat.png"), "paths are not read"),
        ("C:\\photos\\cat.png", "paths are not read"),
        (b64(PNG)[:-2], "not valid base64"),  # padding cut
        (b64(PNG)[:40] + "\n" + b64(PNG)[40:], "not valid base64"),  # line breaks
        (base64.urlsafe_b64encode(b"\xfb\xff" * 30).decode(), "not valid base64"),  # the URL-safe alphabet
        ("data:image/png," + b64(PNG), "only base64 data URIs"),
        ("data:image/png;base64" + b64(PNG), "only base64 data URIs"),
        ("data:audio/wav;base64," + b64(PNG), "of type 'audio/wav' where an image goes"),
        (b64(b"GIF86a not quite"), "not an image in a format ippon reads"),
        (b64(b"P6\n1 1\n255\n\x00\x00\x00"), "not an image in a format ippon reads"),  # arrays' form only
    ],
)
def test_what_is_not_an_image_is_refused(value: object, message: str) -> None:
    with pytest.raises(InvalidRequest, match=message) as error:
        parse_media(value, "image", "images[1]")
    assert str(error.value).startswith("images[1]: ")


def test_media_sent_over_the_wire_are_capped_and_the_callers_own_files_are_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(decode, "MAX_MEDIA_BYTES", len(PNG) - 1)
    with pytest.raises(InvalidRequest, match="more than the"):
        parse_media(b64(PNG), "image")  # refused on its length, before decoding
    (tmp_path / "plain.png").write_bytes(PNG)
    assert as_media(tmp_path / "plain.png", "image").format == "png"
    assert as_media(PNG, "image").format == "png"


def test_audio() -> None:
    samples = (np.sin(np.arange(1600) / 5) * 20000).astype(np.int16)
    clip = parse_media(b64(wav(samples)), "audio", "audio")
    assert clip.kind == "audio"
    assert clip.format == "wav"
    assert as_media(samples, "audio") == clip
    with pytest.raises(InvalidRequest, match="a PNG image, not a clip"):
        parse_media(b64(PNG), "audio")
    with pytest.raises(InvalidRequest, match="a WAV clip, not an image"):
        parse_media(b64(clip.data), "image")
    with pytest.raises(InvalidRequest, match="mono"):
        as_media(np.zeros((2, 100), np.int16), "audio")
    with pytest.raises(InvalidRequest, match="int16 or floats"):
        as_media(np.zeros(100, np.int32), "audio")
    with pytest.raises(InvalidRequest, match="an audio clip where an image goes"):
        as_media(clip, "image")


def wav_of(code: int, channels: int, rate: int, bits: int, data: bytes, extra: bytes = b"") -> bytes:
    """A WAV file with these fmt fields (`extra` after them: an extensible header's) and an odd chunk before data."""
    width = channels * bits // 8
    fmt = struct.pack("<HHIIHH", code, channels, rate, rate * width, width, bits) + extra
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"LIST" + struct.pack("<I", 3) + b"abc\x00"
    return b"RIFF" + struct.pack("<I", len(body) + 8 + len(data)) + body + b"data" + struct.pack("<I", len(data)) + data


def test_wav_clips_read_as_the_samples_sent_and_the_rest_is_refused() -> None:
    """PCM16 and float32 (plain or extensible) come back as the int16 and float32 samples written, past other chunks,
    with their channels and rate as stored; another sample format is refused, as is anything malformed."""
    ints = (np.sin(np.arange(1600) / 5) * 20000).astype(np.int16)
    floats = (ints / 32768).astype(np.float32)
    for samples in (ints, floats):
        got, rate, _ = decode.wav(as_media(samples, "audio"))
        assert (got.dtype, rate) == (samples.dtype, decode.ARRAY_RATE)
        np.testing.assert_array_equal(got[:, 0], samples)
    guid = struct.pack("<HHI", 22, 16, 4) + struct.pack("<H", 1) + GUID_TAIL  # extensible, PCM subformat
    extensible = wav_of(0xFFFE, 1, 16000, 16, ints.tobytes(), guid)
    np.testing.assert_array_equal(decode.wav(Media.of(extensible, "audio"))[0][:, 0], ints)
    stereo, rate, _ = decode.wav(Media.of(wav_of(1, 2, 44100, 16, ints.tobytes()), "audio"))
    assert (stereo.shape, rate) == ((800, 2), 44100)
    streamed = bytearray(wav_of(1, 1, 16000, 16, ints.tobytes()))  # as FFmpeg writes to a pipe: sizes left unknown
    struct.pack_into("<I", streamed, 4, 0xFFFFFFFF)
    struct.pack_into("<I", streamed, len(streamed) - ints.nbytes - 4, 0xFFFFFFFF)
    np.testing.assert_array_equal(decode.wav(Media.of(bytes(streamed), "audio"))[0][:, 0], ints)
    nan = floats.copy()
    nan[5] = np.nan
    refused = {
        "not 24-bit PCM": wav_of(1, 1, 16000, 24, b"\x00" * 30),
        "not 64-bit float": wav_of(3, 1, 16000, 64, b"\x00" * 32),
        "not finite": wav_of(3, 1, 16000, 32, nan.tobytes()),
        "no samples": wav_of(1, 1, 16000, 16, b""),
        "ends mid-sample": wav_of(1, 1, 16000, 16, b"\x00" * 3),
        "cut short": wav_of(1, 1, 16000, 16, ints.tobytes())[:-10],
    }
    for message, data in refused.items():
        with pytest.raises(InvalidRequest, match=message):
            decode.wav(Media.of(data, "audio"))


def test_python_values(tmp_path: Path) -> None:
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    m = as_media(pixels, "image")
    assert m.format == "ppm"
    np.testing.assert_array_equal(load_rgb(m, exif=True), pixels)  # no decoder: arrays need no Pillow
    assert as_media(PNG, "image") == as_media(bytearray(PNG), "image") == as_media(DATA / "images/plain.png", "image")
    assert as_media(str(DATA / "images/plain.png"), "image").format == "png"
    assert as_media("data:image/jpeg;base64," + b64(JPEG), "image").format == "jpeg"
    assert as_media(m, "image") is m
    with pytest.raises(InvalidRequest, match="no such file"):
        as_media(tmp_path / "missing.png", "image")
    with pytest.raises(InvalidRequest, match="URLs are not fetched"):
        as_media("https://example.com/cat.png", "image")
    with pytest.raises(InvalidRequest, match="H x W x 3 uint8"):
        as_media(np.zeros((4, 4), np.uint8), "image")
    with pytest.raises(InvalidRequest, match="H x W x 3 uint8"):
        as_media(np.zeros((4, 4, 3), np.float32), "image")
    with pytest.raises(InvalidRequest, match="can't be a list"):
        as_media([1, 2], "image")
    with pytest.raises(InvalidRequest, match="not an audio file"):
        as_media(b"\x00" * 64, "audio")


def test_pillow_decodes_as_the_makers_decoders_do() -> None:
    """Every fixture image, turned by its EXIF orientation (transformers' load_image, d1-omni's and Clef's decode) and
    not (pplx's), against the pixels Pillow 12.3.0 gave in the creators' venv: palettes, grayscale, 16-bit, RGBA and
    CMYK through convert("RGB"), the first frame of a GIF. Each is opened only as the format named from its first
    bytes, so a misnamed format fails here too."""
    pil = importlib.import_module("PIL.Image")

    for name, want in DECODED["images"].items():
        m = as_media(DATA / "images" / name, "image")
        for key, exif in (("exif", True), ("plain", False)):
            pixels = load_rgb(m, exif=exif)
            assert pixels.dtype == np.uint8, name
            assert list(pixels.shape) == want[key]["shape"], (name, key)
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == want[key]["sha256"], (name, key, pil.__version__)
    with pil.open(DATA / "images/exif6.jpg") as image:  # a PIL image keeps its EXIF, as load_image turns one too
        given = as_media(image, "image")
    for key, exif in (("exif", True), ("plain", False)):
        pixels = load_rgb(given, exif=exif)
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == DECODED["images"]["exif6.jpg"][key]["sha256"], key
    broken = Media.of(PNG[:60], "image")
    with pytest.raises(InvalidRequest, match="could not be decoded"):
        load_rgb(broken, exif=True)
    # Malformed AVIF raises RuntimeError inside Pillow's AVIF plugin; load_rgb turns it into InvalidRequest (400).
    broken_avif = Media.of(b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1\x00\x00\x00\x00bad!", "image")
    with pytest.raises(InvalidRequest, match="AVIF image could not be decoded"):
        load_rgb(broken_avif, exif=True)
    # an AVIF that names the generic still image brand first (mif1) and AVIF among its compatible brands is an AVIF
    mif1_first = Media.of(b"\x00\x00\x00\x1cftypmif1\x00\x00\x00\x00mif1avif\x00\x00\x00\x00bad!", "image")
    with pytest.raises(InvalidRequest, match="AVIF image could not be decoded"):
        load_rgb(mif1_first, exif=True)
    # Image count is enforced at the boundary; ordinary multi-photo phone sets (e.g. 2 x 24 MP or 4 x 12 MP) are
    # accepted by check_images and budgeted by encoded vision tokens inside QwenImages instead of raw source pixels.
    small = as_media(PNG, "image")
    with pytest.raises(InvalidRequest, match="at most"):
        decode.check_images([small] * (decode.MAX_IMAGES + 1))
    buf = io.BytesIO()
    pil.new("L", (4000, 3000)).save(buf, format="PNG")
    phone = Media.of(buf.getvalue(), "image")
    decode.check_images([phone] * 4)  # 4 x 12 MP = 48 MP source pixels: must not be refused at the header

    from ippon.errors import ContextTooLong
    from ippon.media.qwen import IMAGE_MAX_TOKENS, ImageConfig, QwenImages

    def _never_tower() -> Any:
        raise AssertionError("tower should not load during rows()")

    # On a 1.64 MP model (d3/d1), 4 x 12 MP phone photos fit (6,400 tokens <= 8,192) and are resized immediately on
    # decode so full 12 MP frames are not held in memory, while 12 x 4 MP (19,200 tokens) is refused from headers.
    d3_cfg = ImageConfig(16, 2, 2, 65_536, 1_638_400, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1 / 255)
    d3_images = QwenImages(d3_cfg, exif=False, token=1, loader=_never_tower)
    rows, decoded_pixels = d3_images.rows([phone])
    assert len(rows[0]) <= 1_600
    prepared = next(iter(decoded_pixels.values()))
    assert prepared.shape[0] * prepared.shape[1] <= d3_cfg.max_pixels
    buf_4mp = io.BytesIO()
    pil.new("L", (2048, 2048)).save(buf_4mp, format="PNG")
    four_mp = Media.of(buf_4mp.getvalue(), "image")
    with pytest.raises(ContextTooLong, match=f"{IMAGE_MAX_TOKENS:,} tokens"):
        d3_images.rows([four_mp] * 12)
    # On a compact 262k-pixel model (pplx), 12 x 4 MP encodes to 3,072 tokens (<= 8,192) and passes cleanly.
    compact_cfg = ImageConfig(16, 2, 2, 65_536, 262_144, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1 / 255)
    compact_images = QwenImages(compact_cfg, exif=False, token=1, loader=_never_tower)
    compact_rows, _ = compact_images.rows([four_mp] * 12)
    assert sum(len(r) for r in compact_rows) <= IMAGE_MAX_TOKENS


@pytest.mark.parametrize(
    ("name", "format", "lossless"),
    [
        ("chirp.flac", "flac", True),
        ("chirp.aiff", "aiff", True),
        ("chirp.caf", "caf", True),  # 24-bit samples
        ("chirp.m4a", "mp4", False),  # afconvert's AAC, its priming in an iTunSMPB tag
        ("chirp-ffmpeg.m4a", "mp4", False),  # FFmpeg's, in an edit list
        ("chirp.mp3", "mp3", False),  # LAME's, in its Xing header
        ("chirp-opus.ogg", "ogg", False),  # Opus, a voice note's codec: decoded by FFmpeg, not natively
    ],
)
def test_a_recording_in_any_common_format_is_heard_as_its_wav(name: str, format: str, lossless: bool) -> None:
    """Each file decodes to the WAV's samples: the same length (an encoder's priming and padding trimmed, as FFmpeg
    trims them), equal for a lossless format and within the codec's loss for AAC and MP3 (data/media/
    make_audio_fixtures.py)."""
    from ippon.media.lfm3_audio import samples

    want = samples(as_media(DATA / "audio/chirp.wav", "audio"))
    clip = as_media(DATA / "audio" / name, "audio")
    got = samples(clip)
    assert clip.format == format
    assert got.shape == want.shape
    if lossless:
        np.testing.assert_allclose(got, want, atol=1e-4)
    else:
        assert np.corrcoef(got, want)[0, 1] > 0.99


def test_an_ogg_recording_of_two_streams_is_heard_whole() -> None:
    """A chained Ogg file (chirp.ogg twice, end to end, and chirp-opus.ogg twice with the same serial) decodes to both
    streams' samples, the one file's twice over, and the chirp is heard in each."""
    from ippon.media.lfm3_audio import samples

    once, rate = native.decode_audio((DATA / "audio/chirp.ogg").read_bytes(), 3600)
    twice, _ = native.decode_audio((DATA / "audio/chirp-chained.ogg").read_bytes(), 3600)
    want, _, _ = decode.wav(as_media(DATA / "audio/chirp.wav", "audio"))
    assert (rate, twice.shape) == (22050, (2 * len(once), 2))
    for part in (twice[: len(want)], twice[len(once) : len(once) + len(want)]):
        assert np.corrcoef(part[:, 0], want[:, 0])[0, 1] > 0.99
    opus = (DATA / "audio/chirp-opus.ogg").read_bytes()
    opus_once = samples(Media.of(opus, "audio"))
    opus_twice = samples(Media.of(opus + opus, "audio"))
    assert len(opus_twice) == 2 * len(opus_once)
    for part in (opus_twice[: len(opus_once)], opus_twice[len(opus_once) :]):
        assert np.corrcoef(part, opus_once)[0, 1] > 0.99


def test_hostile_recordings_and_videos_are_refused_or_read_at_once() -> None:
    """A WAV whose sample rate is 0 or past 384 kHz is refused, a prime rate inside the bound (383,987 Hz) resamples
    in milliseconds instead of stalling for tens of seconds, a FLAC declaring 384 kHz x 8 ch x 10 min is refused before
    decoding instead of expanding into gigabytes, and an MKV of one second whose header claims 20 hours is read in
    moments (d3's reader kept grabbing to the claimed end)."""
    import struct
    import time

    from ippon.media import d3_video
    from ippon.media.lfm3_audio import samples

    def wav(rate: int) -> bytes:
        fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
        body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", 32) + bytes(32)
        return b"RIFF" + struct.pack("<I", len(body)) + body

    def flac_header(rate: int, channels: int, total_samples: int) -> bytes:
        # 42-byte FLAC magic + last-metadata STREAMINFO block (block sizes 4096, 16-bit PCM)
        bits = (rate << 44) | ((channels - 1) << 41) | (15 << 36) | (total_samples & ((1 << 36) - 1))
        streaminfo = struct.pack(">HH", 4096, 4096) + bytes(6) + bits.to_bytes(8, "big") + bytes(16)
        return b"fLaC" + bytes([0x80, 0, 0, 34]) + streaminfo

    for rate in (0, 1_000_003):
        with pytest.raises(InvalidRequest, match="sample rate"):
            samples(Media.of(wav(rate), "audio", "clip"))
    started = time.perf_counter()
    prime = samples(Media.of(wav(383_987), "audio", "clip"))
    assert time.perf_counter() - started < 0.5
    assert len(prime) >= 1
    # A 43-minute 48 kHz stereo recording (123.8M frames, 247.7M raw interleaved samples) must pass the frame budget
    # (header-only FLAC fails only because it has no packets, not from the sample/frame cap), and on-the-fly mono
    # mixdown in Rust must match numpy's stereo mean.
    with pytest.raises(InvalidRequest, match="holds no audio"):
        samples(Media.of(flac_header(48_000, 2, 48_000 * 60 * 43), "audio", "clip"))
    stereo_raw, stereo_rate = native.decode_audio((DATA / "audio/chirp.ogg").read_bytes(), 3600)
    mono_raw, mono_rate = native.decode_audio((DATA / "audio/chirp.ogg").read_bytes(), 3600, mono=True)
    assert (mono_rate, mono_raw.shape) == (stereo_rate, (len(stereo_raw), 1))
    np.testing.assert_allclose(mono_raw[:, 0], stereo_raw.mean(axis=1, dtype=np.float32), atol=1e-6)
    started = time.perf_counter()
    with pytest.raises(InvalidRequest, match="more than"):
        samples(Media.of(flac_header(384_000, 8, 384_000 * 600), "audio", "clip"))
    assert time.perf_counter() - started < 0.5
    started = time.perf_counter()
    clip = d3_video.load_frames(as_media(DATA / "videos/claims-20-hours.mkv", "video"))
    assert time.perf_counter() - started < 5
    assert len(clip.frames) >= 2
