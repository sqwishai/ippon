"""d1-2B on LFM3: its rows against Liquid's own code (references/d1), with special-token text, no state, codes past Z,
letter labels, instructions and descriptions that are other JSON, and more questions than a row holds: the tokens,
each token's question under set attention, each question's answer slot and label tokens, and the usage; its pictures'
and d1-Omni-2B's clips' spans and rows; its median score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adapter_fixtures import question as q
from adapter_fixtures import renderings, snapshot
from ippon.adapters.d1 import Clips, D1Format, Pictures, videos
from ippon.arch.common import media_digest, media_of
from ippon.errors import InvalidRequest
from ippon.media.decode import as_media
from ippon.media.lfm3_video import load_frames
from ippon.media.qwen import ImageConfig, VideoConfig, preprocess_video
from ippon.media.video import sample
from ippon.model_dir import ModelDir
from ippon.questions import with_media
from ippon.schema import parse_request
from ippon.tokenizer import Tokenizer
from ippon.verify import load_reference, media_files, reference_request

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/d1/d1-2B.json.gz"
JSON_TEXT = ROOT / "references/d1/d1-2B-json.json.gz"  # Liquid's current prompt.py, which takes other JSON
PICTURES = ROOT / "references/d1/d1-2B-media.json.gz"
CLIPS = ROOT / "references/d1/d1-omni-2B-audio.json.gz"
IMAGE_TOKEN, AUDIO_TOKEN = 125030, 125031


def d1_format() -> D1Format:
    model = load_reference(REFERENCE)["model"]
    d = snapshot(model["repo"], model["revision"], "tokenizer.json", "tokenizer_config.json")
    tok = Tokenizer.from_file(d / "tokenizer.json", json.loads((d / "tokenizer_config.json").read_text()))
    return D1Format(tok, "<|startoftext|>", None, 1.3374)  # type: ignore[arg-type]


def test_rows_question_groups_slots_and_labels_are_the_makers() -> None:
    fmt = d1_format()
    found = load_reference(REFERENCE)["requests"] + load_reference(JSON_TEXT)["requests"]
    for case in (c for c in found if "error" in c):  # whitespace for instructions, none at all, a media object
        try:
            fmt.render(parse_request(case["request"]))
        except InvalidRequest:
            continue
        raise AssertionError(f"{case['name']}: the creator refuses it")
    for case, rendering in renderings(fmt, [c for c in found if "error" not in c], 28):
        assert [list(p.groups) for p in rendering.prompts] == case["segments"], case["name"]
        assert [[r.position for r in p.readouts] for p in rendering.prompts] == case["readouts"], case["name"]
        labels = rendering.meta["labels"]
        assert {k: [list(g) for g in labels[k]] for k in case["groups"]} == case["groups"], case["name"]


def test_pictures_head_the_state_as_the_maker_writes_them() -> None:
    """Each picture's span and rows, in order, against Liquid's code: its EXIF turn, d1's cap on the 12 MP ones,
    Qwen2-VL's resize to multiples of 32, the same picture twice; the state's tokens and the usage (no tower runs)."""
    model = load_reference(REFERENCE)["model"]
    d = snapshot(model["repo"], model["revision"], "processor_config.json")
    fmt = d1_format()
    image = ImageConfig.from_hf(json.loads((d / "processor_config.json").read_text()))
    fmt.pictures = Pictures(image, True, IMAGE_TOKEN, lambda: pytest.fail("rendering runs no tower"))
    found = load_reference(PICTURES)["requests"]
    for case in found:
        files = [(ROOT / "references/media/images" / name).read_bytes() for name in case["request"]["images"]]
        rendering = fmt.render(with_media(parse_request(dict(case["request"], images=None)), files, None))
        assert [[t if t >= 0 else IMAGE_TOKEN for t in p.tokens] for p in rendering.prompts] == case["prompts"]
        assert [list(p.groups) for p in rendering.prompts] == case["segments"], case["name"]
        assert fmt.usage(rendering)["input_tokens"] == case["input_tokens"], case["name"]


def test_clips_are_heard_where_the_maker_places_them() -> None:
    """d1-Omni-2B's clip spans and rows (one per 8 spectrogram frames of any length, a float32 clip as a waveform, a
    stereo clip at another rate): the state's at its head, a spoken question's where its text would be and among its
    question's tokens, against Liquid's code; the usage (no encoder runs)."""
    found = load_reference(CLIPS)
    d = snapshot(found["model"]["repo"], found["model"]["revision"], "tokenizer.json", "tokenizer_config.json")
    tok = Tokenizer.from_file(d / "tokenizer.json", json.loads((d / "tokenizer_config.json").read_text()))
    fmt = D1Format(tok, "<|startoftext|>", None, 3.383)  # type: ignore[arg-type]
    fmt.clips = Clips(AUDIO_TOKEN, lambda: pytest.fail("rendering runs no encoder"))
    files = media_files(CLIPS, found)
    for case in found["requests"]:
        rendering = fmt.render(reference_request(case, files(case)))
        assert [[t if t >= 0 else AUDIO_TOKEN for t in p.tokens] for p in rendering.prompts] == case["prompts"]
        assert [list(p.groups) for p in rendering.prompts] == case["segments"], case["name"]
        assert fmt.usage(rendering)["input_tokens"] == case["input_tokens"], case["name"]


def test_a_score_answers_its_median_level() -> None:
    fmt = D1Format(None, "", None, 1.0)  # type: ignore[arg-type]
    _, answer = fmt.answer(q({"type": "score", "criteria": ["low", "mid", "high"]}), [0.2, 0.25, 0.55])
    assert answer["score"] == 2.0  # the expected level would be 1.35
    assert answer["confidence"] == 0.55


@pytest.mark.parametrize("name", ["d1-2B", "d1-omni-2B"])
def test_video_spans_rows_and_placement_are_the_makers(name: str) -> None:
    """Both processors' frame-pair spans, timestamps and set packing; several videos, repeats, pictures before
    videos and a video with an unread audio track. Rendering never runs the tower."""
    path = ROOT / "references/d1" / f"{name}-video.json.gz"
    found = load_reference(path)
    model = found["model"]
    directory = snapshot(
        model["repo"],
        model["revision"],
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    tok = Tokenizer.from_file(
        directory / "tokenizer.json", json.loads((directory / "tokenizer_config.json").read_text())
    )
    fmt = D1Format(tok, "<|startoftext|>", None, 1.0)  # type: ignore[arg-type]
    image = ImageConfig.from_hf(json.loads((directory / "processor_config.json").read_text()))
    fmt.pictures = Pictures(image, True, IMAGE_TOKEN, lambda: pytest.fail("rendering runs no tower"))
    fmt.videos = videos(ModelDir.open(directory), fmt.pictures, tok)
    assert "videos" in fmt.media
    files = media_files(path, found)
    for case in found["requests"]:
        request = reference_request(case, files(case))
        images = {media_digest(m.sha256) for m in request.images}
        for _ in range(2):  # fresh and reused counts must place the same rows
            rendering = fmt.render(request)
            prompts = [
                [t if t >= 0 else IMAGE_TOKEN if media_of(t)[0] in images else fmt.videos.token for t in p.tokens]
                for p in rendering.prompts
            ]
            assert prompts == case["prompts"], case["name"]
            assert [list(p.groups) for p in rendering.prompts] == case["segments"], case["name"]
            assert [[r.position for r in p.readouts] for p in rendering.prompts] == case["readouts"], case["name"]
            assert fmt.usage(rendering)["input_tokens"] == case["input_tokens"], case["name"]
            grids = [list(seen.grid) for seen in fmt.videos.seen(request.videos)[0]]
            assert all(m["video_grid_thw"] == grids for m in case["media_inputs"])


def test_video_decoder_clock_and_temporal_patches_match_liquid() -> None:
    """On M5 Pro, d1-2B and Omni's shared processor: PyAV matches all 120,342,816 RGB channels in 98 frames.
    OpenCV differs by up to 3 levels and in the variable-rate clock (references/d1/README.md). No tower runs."""
    directory = ROOT / "references/d1"
    found = json.loads((directory / "video-processor.json").read_text())
    config = VideoConfig(ImageConfig.from_hf(found["processor"]))
    with np.load(directory / "video-processor.npz") as arrays:
        for i, case in enumerate(found["clips"]):
            clip = load_frames(as_media(ROOT / "references/media" / case["name"], "video"))
            assert clip.total == case["total"]
            assert clip.fps == pytest.approx(case["fps"], abs=1e-12)
            assert clip.indices == tuple(case["indices"])
            np.testing.assert_array_equal(clip.frames, arrays[f"frames_{i}"])
            patches = preprocess_video(clip.frames, config)
            assert list(patches.grid) == case["grid"]
            np.testing.assert_allclose(patches.pixels, arrays[f"patches_{i}"], atol=2e-6, rtol=0)
    for case in found["sampling"]:
        assert sample(case["total"], case["fps"], most=768) == case["indices"]
