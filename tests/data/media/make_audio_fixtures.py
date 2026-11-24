# mypy: ignore-errors
"""Reference script, run once with macOS's afconvert and Homebrew's FFmpeg 8 and LAME: a half-second chirp at 22,050 Hz
in stereo, saved as the recordings a Mac or a phone makes, which tests/test_media_decode.py decodes and holds to the
chirp itself (`chirp`, which the test computes the same way).

    python -I make_audio_fixtures.py

Writes audio/chirp.{wav,m4a,mp3,flac,aiff,caf,ogg}, audio/chirp-ffmpeg.m4a, audio/chirp-opus.ogg and
audio/chirp-chained.ogg next to this file: the WAV 16-bit PCM; chirp.m4a AAC from afconvert (its gapless priming in
an iTunSMPB tag), chirp-ffmpeg.m4a AAC from FFmpeg (in an edit list), chirp.mp3 from LAME (in its Xing header), the
lossless three as afconvert writes them, chirp.ogg Vorbis from FFmpeg 9's own encoder, chirp-chained.ogg that file
twice, end to end (a chained Ogg recording, two streams in one file), and chirp-opus.ogg Opus from FFmpeg's libopus
at 48 kHz, a phone's voice-note codec, which the native decoder can't read and FFmpeg decodes.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "audio"
RATE = 22_050


def chirp() -> np.ndarray:
    """[frames, 2] float: a 200 to 1,100 Hz sweep on the left, the same a quarter cycle later on the right."""
    t = np.arange(RATE // 2) / RATE
    phase = 2 * np.pi * (200 * t + 900 * t**2)
    return 0.5 * np.stack([np.sin(phase), np.cos(phase)], axis=1)


def main() -> None:
    HERE.mkdir(exist_ok=True)
    source = HERE / "chirp.wav"
    with wave.open(str(source), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(np.round(chirp() * 32767).astype("<i2").tobytes())
    run = lambda *command: subprocess.run(command, check=True)  # noqa: E731
    run("afconvert", "-f", "m4af", "-d", "aac", source, HERE / "chirp.m4a")
    run("afconvert", "-f", "flac", "-d", "flac", source, HERE / "chirp.flac")
    run("afconvert", "-f", "AIFF", "-d", "BEI16", source, HERE / "chirp.aiff")
    run("afconvert", "-f", "caff", "-d", "LEI24", source, HERE / "chirp.caf")
    run("lame", "--quiet", source, HERE / "chirp.mp3")
    run("ffmpeg", "-loglevel", "error", "-y", "-i", source, "-c:a", "aac", HERE / "chirp-ffmpeg.m4a")
    run(
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        "-c:a",
        "vorbis",
        "-strict",
        "experimental",
        HERE / "chirp.ogg",
    )
    run("ffmpeg", "-loglevel", "error", "-y", "-i", source, "-c:a", "libopus", "-ar", "48000", HERE / "chirp-opus.ogg")
    (HERE / "chirp-chained.ogg").write_bytes((HERE / "chirp.ogg").read_bytes() * 2)


if __name__ == "__main__":
    main()
