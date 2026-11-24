"""d1-Omni-2B's audio clips, as Liquid's code reads them (media.py, feature_extraction_lfm3_omni.py and
modeling_lfm3_omni.py in its checkpoint):

- the clip as 16 kHz mono samples (`samples`): a 16-bit PCM WAV's samples over 32,768, a float32 WAV's as they are
  (the waveform a caller hands the creator's API), the channels averaged, and another rate resampled as Liquid's own
  resampler does (torchaudio's default: a Hann-windowed sinc over 6 zero crossings, cut at 0.99 of the lower Nyquist);
- the log-mel front end (`log_mel`): the clip right-padded to whole frames, 400-sample frames every 160 under a
  periodic Hann window with no centring, their power spectra through 128 Slaney mel filters from 0 to 8 kHz, log10
  floored at -12, then (x + 4) / 4;
- the downsampler: three stride-2 3 x 3 convolutions over the (mel, time) plane, the first from 1 to 256 channels and
  the others depthwise, each but the first followed by a 1 x 1 convolution, each by SiLU, then each time step's 256
  channels x 16 mel bins projected to 768 (here in float32);
- a causal LFM2 over those rows (arch/lfm2.py), its attention within 32 positions;
- the projector: a LayerNorm, a Linear, the exact GELU and a Linear to the text model's width.

A clip becomes ceil(frames / 8) rows, about 12.5 a second.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import mlx.core as mx
import numpy as np

from .. import native
from ..arch import lfm2
from ..arch.common import MEDIA_ROW_BITS
from ..arch.ops import MLX, Ops, activate, layer_norm
from ..errors import InvalidRequest, UnsupportedModel
from ..quant import arrays
from ..weights import Weights
from .decode import Media, wav, wav_sample
from .lfm3_video import pyav

F32 = mx.float32
ENCODER, PROJECTOR = "model.audio_encoder.", "model.audio_projector."
N_FFT, HOP, MELS, STRIDE = 400, 160, 128, 8
SAMPLE_RATE = 16_000  # the front end's rate (processor_config.json's sampling_rate), which a clip is resampled to
# the longest clip one placeholder span addresses (about 43 minutes): refused while decoding, before its samples are
MAX_SECONDS = (1 << MEDIA_ROW_BITS) * STRIDE * HOP / SAMPLE_RATE
RATES = (1, 384_000)  # the sample rates a clip may have: one past them is not audio, or divides by 0
# Decoded mono frames permitted before resampling: 135M (~540 MiB as f32), matching audio.rs — covers the full
# ~43-minute limit at 48 kHz for stereo or multi-channel recordings while still refusing high-rate silence bombs.
MAX_DECODED_FRAMES = 135_000_000
_FILTER_FLOATS = 1 << 24  # the most of the resampling filter made at once: 64 MB of float32
# Above this [new, taps] polyphase table size (prime/coprime rates like 383,987 Hz where gcd(rate, 16000) == 1),
# evaluate the same Hann-windowed sinc directly per output sample so resampling is O(out_len * support).
_POLYPHASE_LIMIT = 2_000_000


def _resampled_direct(wave: np.ndarray, rate: int, target: int, zeros: int, rolloff: float) -> np.ndarray:
    """Direct per-output Hann-windowed sinc resampling for coprime/prime sample rates whose polyphase table would be
    huge; mathematically identical to `resampled` within float32 rounding."""
    gcd = math.gcd(rate, target)
    old, new = rate // gcd, target // gcd
    cut = min(old, new) * rolloff
    scale = cut / old
    half = math.ceil(zeros / scale)
    n_in = len(wave)
    out_len = (n_in * target + rate - 1) // rate
    out = np.empty(out_len, dtype=np.float32)
    step = rate / target
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    chunk = max(1, min(4096, 2_000_000 // max(1, len(offsets))))
    for start in range(0, out_len, chunk):
        indices = np.arange(start, min(start + chunk, out_len), dtype=np.int64)
        positions = indices * step
        centers = np.floor(positions).astype(np.int64)
        source = centers[:, None] + offsets[None, :]
        valid = (source >= 0) & (source < n_in)
        taps = np.clip((source - positions[:, None]) * scale, -zeros, zeros)
        window = np.cos(taps * (math.pi / (2.0 * zeros))) ** 2
        weights = np.sinc(taps) * window * scale
        values = wave[np.clip(source, 0, n_in - 1)] * valid
        out[start : start + len(indices)] = (values * weights).sum(axis=1).astype(np.float32)
    return out


def resampled(
    wave: np.ndarray, rate: int, target: int = SAMPLE_RATE, zeros: int = 6, rolloff: float = 0.99
) -> np.ndarray:
    """A mono float32 waveform at `rate` resampled to `target`, as Liquid's media.resampled (torchaudio's
    functional.resample by default): one Hann-windowed sinc filter per output phase, in float32."""
    if rate == target or len(wave) == 0:
        return wave
    gcd = math.gcd(rate, target)
    old, new = rate // gcd, target // gcd
    cut = min(old, new) * rolloff
    width = math.ceil(zeros * old / cut)
    if new * (2 * width + old) > _POLYPHASE_LIMIT:
        return _resampled_direct(wave, rate, target, zeros, rolloff)
    taps = np.arange(-width, width + old, dtype=np.float32)[None] / np.float32(old)
    x = np.pad(wave.astype(np.float32), (width, width + old))
    frames = np.lib.stride_tricks.sliding_window_view(x, taps.shape[1])[::old]
    out = np.empty((len(frames), new), np.float32)
    step = max(1, _FILTER_FLOATS // taps.shape[1])
    for first in range(0, new, step):
        t = np.arange(-first, -min(first + step, new), -1, dtype=np.float32)[:, None] / np.float32(new)
        t = ((t + taps) * np.float32(cut)).clip(-zeros, zeros)
        window = np.cos(t * np.float32(math.pi / zeros / 2)) ** 2
        t = t * np.float32(math.pi)
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc = np.where(t == 0, np.float32(1), np.sin(t) / t)
        kernel = (sinc * (window * np.float32(cut / old))).astype(np.float32)  # [phases, taps]
        out[:, first : first + len(kernel)] = frames @ kernel.T
    return out.reshape(-1)[: math.ceil(new * len(wave) / old)]


def _ogg_links(data: bytes) -> list[bytes]:
    """Splits a chained Ogg stream into sequential links at RFC 3533 BOS boundaries (a BOS page after any non-BOS page
    starts a new link; concurrent multiplexed streams keep their consecutive BOS pages together)."""
    starts: list[int] = []
    pos, total = 0, len(data)
    seen_body = False
    while pos + 27 <= total and data[pos : pos + 4] == b"OggS":
        header_type = data[pos + 5]
        segments = data[pos + 26]
        table_end = pos + 27 + segments
        if table_end > total:
            return [data]
        page_end = table_end + sum(data[pos + 27 : table_end])
        if page_end > total:
            return [data]
        is_bos = bool(header_type & 0x02)
        if is_bos and (not starts or seen_body):
            starts.append(pos)
            seen_body = False
        elif not is_bos:
            seen_body = True
        pos = page_end
    if len(starts) <= 1 or pos != total:
        return [data]
    return [data[starts[i] : (starts[i + 1] if i + 1 < len(starts) else total)] for i in range(len(starts))]


def _decode_one(clip: Media, max_seconds: float, max_frames: int) -> tuple[np.ndarray, int]:
    """Decodes one non-chained audio clip to 16 kHz mono float32 samples and returns (wave, decoded_frames)."""
    data, rate, sample = None, 0, ""
    decoded_frames = 0
    if clip.format == "wav" and wav_sample(clip) is not None:  # a WAV of another sample format: the native decoder's
        data, rate, sample = wav(clip)
        decoded_frames = len(data)
    if data is None:
        try:
            data, rate = native.decode_audio(clip.data, max_seconds, mono=True)
            decoded_frames = len(data)
        except ValueError as why:
            message = str(why)
            # Budget/rate refusals from the native decoder must not fall through to FFmpeg and decode a bomb twice.
            if any(
                tag in message
                for tag in (
                    "longer than",
                    "more than",
                    "sample rate",
                    "channel count",
                    "changes from",
                    "holds no audio",
                )
            ):
                raise InvalidRequest(f"audio: {message}") from None
            data, rate, decoded_frames = ffmpeg_audio(clip.data, message, max_seconds, max_frames)
    elif sample == "pcm16":
        data = data.astype(np.float32) / np.float32(32768)
    if not RATES[0] <= rate <= RATES[1]:
        raise InvalidRequest(f"audio: a sample rate of {rate:,} Hz; a clip is {RATES[0]} Hz to {RATES[1]:,} Hz")
    if decoded_frames > max_frames:
        raise InvalidRequest(f"audio: the recording decodes to more than {MAX_DECODED_FRAMES // 1_000_000}M frames")
    if len(data) > max_seconds * rate:
        raise InvalidRequest(f"audio: the recording is longer than the {MAX_SECONDS // 60:.0f} minutes a clip may be")
    mono = data.mean(axis=1, dtype=np.float32) if data.shape[1] > 1 else data[:, 0]
    return resampled(mono, rate), decoded_frames


def samples(clip: Media) -> np.ndarray:
    """A clip as the 16 kHz mono float32 samples the front end reads (see the module docstring). A 16-bit PCM or
    float32 WAV is read as Liquid's code reads it; any other recording (MP3, M4A, FLAC, Ogg Vorbis, AIFF, CAF, a WAV
    of 24-bit samples) is decoded natively to mono float samples at its own rate, then resampled the same way.
    Chained Ogg recordings are decoded link by link so same-serial Opus, mixed-codec, or rate-changing streams are
    heard whole. InvalidRequest for a file that can't be decoded."""
    if clip.format == "ogg":
        links = _ogg_links(clip.data)
        if len(links) > 1:
            parts: list[np.ndarray] = []
            remaining_seconds = MAX_SECONDS
            remaining_frames = MAX_DECODED_FRAMES
            for link in links:
                part, used = _decode_one(Media.of(link, "audio"), remaining_seconds, remaining_frames)
                parts.append(part)
                remaining_seconds -= len(part) / SAMPLE_RATE
                remaining_frames -= used
                if remaining_seconds < 0:
                    raise InvalidRequest(
                        f"audio: the recording is longer than the {MAX_SECONDS // 60:.0f} minutes a clip may be"
                    )
                if remaining_frames < 0:
                    raise InvalidRequest(
                        f"audio: the recording decodes to more than {MAX_DECODED_FRAMES // 1_000_000}M frames"
                    )
            return np.concatenate(parts)
    wave, _ = _decode_one(clip, MAX_SECONDS, MAX_DECODED_FRAMES)
    return wave


def ffmpeg_audio(
    data: bytes, why: str, max_seconds: float = MAX_SECONDS, max_frames: int = MAX_DECODED_FRAMES
) -> tuple[np.ndarray, int, int]:
    """A clip the native decoder can't read (Opus, in an Ogg voice note or a browser's WebM recording), decoded by
    FFmpeg through PyAV, as transformers' `load_audio` decodes it: mono float32 [frames, 1] at its own rate, plus the
    mono frame count read. InvalidRequest saying `why` when FFmpeg can't read it either."""
    av = pyav()
    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.audio:
                raise InvalidRequest(f"audio: {why}")
            stream = container.streams.audio[0]
            rate = int(stream.rate or 0)
            if not RATES[0] <= rate <= RATES[1]:
                raise InvalidRequest(f"audio: a sample rate of {rate:,} Hz; a clip is {RATES[0]} Hz to {RATES[1]:,} Hz")
            channels = int(stream.channels or 1)
            if stream.duration is not None and stream.time_base is not None:
                duration_s = float(stream.duration * stream.time_base)
                if duration_s > max_seconds:
                    raise InvalidRequest(
                        f"audio: the recording is longer than the {MAX_SECONDS // 60:.0f} minutes a clip may be"
                    )
                if duration_s * rate > max_frames:
                    raise InvalidRequest(
                        f"audio: the recording decodes to more than {MAX_DECODED_FRAMES // 1_000_000}M frames "
                        f"({rate:,} Hz x {channels} ch)"
                    )
            mono = av.AudioResampler(format="flt", layout="mono", rate=rate)
            parts: list[np.ndarray] = []
            most_frames = max_seconds * rate
            total_frames = 0
            for frame in container.decode(stream):
                for chunk in mono.resample(frame):
                    flat = chunk.to_ndarray().reshape(-1)
                    total_frames += len(flat)
                    if total_frames > max_frames:
                        raise InvalidRequest(
                            f"audio: the recording decodes to more than {MAX_DECODED_FRAMES // 1_000_000}M frames "
                            f"({rate:,} Hz x {channels} ch)"
                        )
                    if total_frames > most_frames:  # refused before the rest is decoded
                        raise InvalidRequest(
                            f"audio: the recording is longer than the {MAX_SECONDS // 60:.0f} minutes a clip may be"
                        )
                    parts.append(flat)
            for chunk in mono.resample(None):
                flat = chunk.to_ndarray().reshape(-1)
                total_frames += len(flat)
                parts.append(flat)
    except av.FFmpegError:
        raise InvalidRequest(f"audio: {why}") from None
    if not parts or total_frames == 0:
        raise InvalidRequest("audio: the file holds no audio samples")
    return np.concatenate(parts).astype(np.float32)[:, None], rate, total_frames


def frames(n: int) -> int:
    """The spectrogram frames of `n` samples: whole frames after the right padding."""
    padded = N_FFT if n < N_FFT else n + (-(n - N_FFT) % HOP)
    return 1 + (padded - N_FFT) // HOP


def clip_rows(n: int) -> int:
    """The rows a clip of `n` 16 kHz samples becomes."""
    return -(frames(n) // -STRIDE)


def mel_filters(bins: int = N_FFT // 2 + 1, mels: int = MELS, rate: int = SAMPLE_RATE) -> np.ndarray:
    """transformers' mel_filter_bank(bins, mels, 0, rate / 2, rate, norm="slaney", mel_scale="slaney"), [bins, mels]
    float64."""

    def to_mel(f: np.ndarray) -> np.ndarray:
        out = 3.0 * f / 200.0
        high = f >= 1000.0
        out[high] = 15.0 + np.log(f[high] / 1000.0) * (27.0 / np.log(6.4))
        return out

    def to_hertz(m: np.ndarray) -> np.ndarray:
        out = 200.0 * m / 3.0
        high = m >= 15.0
        out[high] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (m[high] - 15.0))
        return out

    edges = to_mel(np.array([0.0, rate / 2]))
    points = to_hertz(np.linspace(edges[0], edges[1], mels + 2))
    fft = np.linspace(0, rate // 2, bins)
    slopes = points[None] - fft[:, None]
    gaps = np.diff(points)
    triangles = np.maximum(0.0, np.minimum(-slopes[:, :-2] / gaps[:-1], slopes[:, 2:] / gaps[1:]))
    return np.asarray(triangles * (2.0 / (points[2 : mels + 2] - points[:mels]))[None])


FILTERS = mel_filters().astype(np.float32)
# processor_config.json's feature extractor as this front end computes it (`check_front_end`)
FRONT_END = {
    "feature_size": MELS,
    "sampling_rate": SAMPLE_RATE,
    "n_fft": N_FFT,
    "hop_length": HOP,
    "win_length": N_FFT,
    "log_floor": -12.0,
    "normalization_offset": 4.0,
    "normalization_divisor": 4.0,
    "encoder_stride": STRIDE,
}


def check_front_end(processor: Mapping[str, Any]) -> None:
    """UnsupportedModel when the checkpoint's feature extractor is not the one `log_mel` computes."""
    found = processor.get("feature_extractor") or {}
    if found.get("feature_extractor_type") != "Lfm3OmniFeatureExtractor" or any(
        found.get(k, v) != v for k, v in FRONT_END.items()
    ):
        raise UnsupportedModel("a d1-Omni feature extractor this implementation does not cover")


def log_mel(wave: np.ndarray) -> np.ndarray:
    """The front end's features [128, frames] float32 of 16 kHz samples."""
    n = len(wave)
    count = frames(n)
    x = np.pad(wave.astype(np.float32), (0, (count - 1) * HOP + N_FFT - n))
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic
    windows = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:count] * hann
    spectrum = np.fft.rfft(windows.astype(np.float64), axis=-1)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    with np.errstate(divide="ignore"):
        mel = np.maximum(np.log10(power @ FILTERS), np.float32(-12.0))
    return np.asarray(((mel + 4.0) / 4.0).T, dtype=np.float32)


class Lfm3Audio:
    """d1-Omni-2B's speech encoder and projector (see the module docstring): a clip's samples to its rows."""

    ops: Ops = MLX  # how its matmuls run: set from the chip profile, as a backbone's

    def __init__(self, config: Mapping[str, Any], weights: Weights, dtype: mx.Dtype = mx.float32):
        a = config["audio_encoder_config"]
        if (a.get("num_mel_bins"), a.get("conv_channels")) != (MELS, 256) or config.get("projector_bias") is False:
            raise UnsupportedModel("a d1-Omni speech encoder this implementation does not cover")
        if ENCODER + "downsample.conv_in.weight" not in weights:
            raise UnsupportedModel(f"{weights.source}: no speech encoder under {ENCODER!r}")
        self.dtype = dtype
        d = ENCODER + "downsample."

        def conv(name: str) -> tuple[mx.array, mx.array]:  # a torch Conv2d's weight as MLX's [out, kh, kw, in]
            return weights.get(d + name + ".weight", F32).transpose(0, 2, 3, 1), weights.get(d + name + ".bias", F32)

        def linear(name: str) -> tuple[mx.array, mx.array]:  # a 1 x 1 convolution as a matrix [in, out]
            w = weights.get(d + name + ".weight", F32)
            return w.reshape(w.shape[0], -1).T, weights.get(d + name + ".bias", F32)

        self.conv_in, self.conv_mid, self.conv_end = conv("conv_in"), conv("conv_mid"), conv("conv_end")
        self.lin_mid, self.lin_end, self.lin_out = linear("lin_mid"), linear("lin_end"), linear("lin_out")
        text = dict(a["text_config"])
        window = int(a["sliding_window"]) - 1  # transformers' window of 32 positions: 31 keys before a query
        self.encoder = lfm2.LFM2(replace(lfm2.Config.from_hf(text), window=window), weights, dtype, ENCODER + "lfm.")
        p = PROJECTOR
        self.norm = (weights.get(p + "layer_norm.weight", dtype), weights.get(p + "layer_norm.bias", dtype))
        self.up = (weights.matrix(p + "linear_1.weight", dtype), weights.get(p + "linear_1.bias", dtype))
        self.down = (weights.matrix(p + "linear_2.weight", dtype), weights.get(p + "linear_2.bias", dtype))
        mx.eval(arrays([*self.conv_in, *self.conv_mid, *self.conv_end, *self.lin_mid, *self.lin_end, *self.lin_out]))
        mx.eval(arrays([*self.norm, *self.up, *self.down]))

    @classmethod
    def load(cls, weights: Weights, config: Mapping[str, Any], dtype: mx.Dtype = mx.float32) -> Lfm3Audio:
        return cls(config, weights, dtype)

    def downsample(self, features: np.ndarray) -> mx.array:
        """[128, frames] features to [ceil(frames / 8), 768] rows, in float32."""
        x = mx.array(features)[None, :, :, None]  # [1, mel, time, 1]: MLX's channels last
        x = mx.conv2d(x, self.conv_in[0], stride=2, padding=1) + self.conv_in[1]
        x = x * mx.sigmoid(x)
        for (taps, bias), (w, b) in ((self.conv_mid, self.lin_mid), (self.conv_end, self.lin_end)):
            x = mx.conv2d(x, taps, stride=2, padding=1, groups=x.shape[-1]) + bias
            x = x @ w + b
            x = x * mx.sigmoid(x)
        _, mel, time, channels = x.shape
        flat = x[0].transpose(1, 2, 0).reshape(time, channels * mel)  # torch's flatten of [channels, mel]
        return flat @ self.lin_out[0] + self.lin_out[1]

    def __call__(self, wave: np.ndarray) -> mx.array:
        """A clip's rows [ceil(frames / 8), out] in the compute dtype, from its 16 kHz samples."""
        self.encoder.ops = self.ops
        x = self.encoder.read_embeddings(self.downsample(log_mel(wave)).astype(self.dtype))
        x = layer_norm(x, *self.norm, 1e-5)  # the projector's LayerNorm: torch's default epsilon
        x = activate(self.ops.mm_bias(x, *self.up), "gelu")
        return self.ops.mm_bias(x, *self.down)
