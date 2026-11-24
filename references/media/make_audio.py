"""The clips in audio/: speech excerpts from public-domain and CC0 recordings on Wikimedia Commons, and two synthetic
clips, as 16 kHz mono WAV, and one excerpt at 22,050 Hz in stereo (LICENSES.md has each source and its licence):

    python3 -I make_audio.py DOWNLOADS

DOWNLOADS is a directory for the recordings as Commons serves them: a missing one is fetched there, and each is
checked by its sha256 before use. Each is decoded, mixed down to mono and resampled to 16 kHz by ffmpeg with its
default resampler (`ffmpeg -i FILE -ac 1 -ar 16000 -f f32le -`; 9.0.1 made the committed files), then cut here at
whole samples and written as PCM16 (x * 32,768, rounded half to even, clipped) or as IEEE float32. The stereo clip,
for a model that resamples and mixes down (d1-Omni-2B), is the 5 s bells excerpt taken to 22,050 Hz by linear
interpolation, its right channel at half the left's level. Needs numpy and ffmpeg. The files committed are the ones
the references were made on, named with their sha256 in ../requests/media-d1-audio.json and ../d1/requests-audio.json.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "audio"
RATE = 16_000
COMMONS = "https://upload.wikimedia.org/wikipedia/commons/"
# name: (Commons path, sha256 of the file as served)
SOURCES = {
    "armstrong": (
        "d/dd/Armstrong_Small_Step.ogg",
        "ad48575dea687c5109bcaf1675ee39008bd6f0a3a5f3457daab986d16c9d56ba",
    ),
    "kennedy": (
        "5/50/Jfk_rice_university_we_choose_to_go_to_the_moon.ogg",
        "7486fba805c20bde4d43750bd9c34aee86ab7acfd3854a1d0f3d64f62ae68b27",
    ),
    "apollo13": (
        "6/6f/Apollo13-wehaveaproblem.ogg",
        "fd0e3d47739d553bb0cf53fda782ddc3b410f1ae2b04323fb519a0d75a62308c",
    ),
    "keats": (
        "b/be/On-First-Looking-Into-Chapman%E2%80%99s-Homer-LibriVox-Vorbis.ogg",
        "0b930a4c9a63ff009debb0ae818640cbb7d5d767eee5da8fc865bd4eae1d00e4",
    ),
    "bells": (
        "6/6a/LibriVox_-_The_Bells_-_Alan_Davis-Drake.ogg",
        "2844ba1ee29ceac48cb8095117db5773e7e441b5ac4dc0d7ba0c1c7c28455931",
    ),
}
# file: (source, first sample, samples (None: to the end), sample format)
EXCERPTS = {
    "armstrong-small-step.wav": ("armstrong", 0, None, "int16"),  # 24.1 s, the whole recording
    "kennedy-rice-30s.wav": ("kennedy", 34 * RATE, 30 * RATE, "int16"),  # exactly 30 s
    "apollo13-problem-15s.wav": ("apollo13", 0, 15 * RATE, "float32"),
    "keats-chapman-40s.wav": ("keats", 13 * RATE, 40 * RATE, "int16"),  # cut to 30 s by the model
    "poe-bells-5s.wav": ("bells", 4 * RATE, 5 * RATE, "int16"),
    "poe-bells-0.3s.wav": ("bells", 20 * RATE, 4800, "float32"),  # padded to 0.5 s by the model
}


STEREO, STEREO_RATE = "poe-bells-5s-22k-stereo.wav", 22_050


def wav(samples: np.ndarray, rate: int = RATE) -> bytes:
    """WAV of `samples` ([n], or [n, channels] interleaved): PCM16 for int16, IEEE float32 (format 3) for float32; a
    fmt and a data chunk only."""
    code, width = (1, 2) if samples.dtype == np.int16 else (3, 4)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    data = samples.astype("<i2" if code == 1 else "<f4").tobytes()
    fmt = struct.pack("<HHIIHH", code, channels, rate, rate * width * channels, width * channels, 8 * width)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def source(name: str, downloads: Path) -> np.ndarray:
    """A recording at 16 kHz mono, float32, as ffmpeg decodes and resamples it."""
    path, digest = SOURCES[name]
    file = downloads / Path(path).name
    if not file.is_file():
        request = urllib.request.Request(COMMONS + path, headers={"User-Agent": "ippon-references/1.0"})
        with urllib.request.urlopen(request) as response:
            file.write_bytes(response.read())
    if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"{file}: not the recording these clips were cut from")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(file), "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(command, check=True, capture_output=True).stdout, "<f4")


def as_format(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "float32":
        return x.astype(np.float32)
    return np.clip(np.round(x.astype(np.float64) * 32768), -32768, 32767).astype(np.int16)


def main() -> None:
    downloads = Path(sys.argv[1]).expanduser()
    downloads.mkdir(parents=True, exist_ok=True)
    HERE.mkdir(exist_ok=True)
    decoded: dict[str, np.ndarray] = {}
    clips: dict[str, np.ndarray] = {}
    for name, (recording, start, count, kind) in EXCERPTS.items():
        if recording not in decoded:
            decoded[recording] = source(recording, downloads)
        x = decoded[recording][start : None if count is None else start + count]
        assert count is None or len(x) == count, (name, len(x))
        clips[name] = as_format(x, kind)
    bells = decoded["bells"][4 * RATE : 9 * RATE].astype(np.float64)
    left = np.interp(np.arange(5 * STEREO_RATE) * (RATE / STEREO_RATE), np.arange(len(bells)), bells)
    clips[STEREO] = as_format(np.stack([left, 0.5 * left], axis=1), "int16")
    clips["silence-1s.wav"] = np.zeros(RATE, np.int16)
    t = np.arange(2 * RATE) / RATE
    clips["tone-440hz-2s.wav"] = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for name, samples in clips.items():
        rate = STEREO_RATE if name == STEREO else RATE
        data = wav(samples, rate)
        (HERE / name).write_bytes(data)
        print(
            f"{name}: {len(samples) / rate:.3f} s {samples.dtype}, {len(data):,} bytes, sha256 "
            f"{hashlib.sha256(data).hexdigest()}"
        )


if __name__ == "__main__":
    main()
