"""vllm-sr d3-mini's and d3-flash's pictures and videos against vllm-sr's own runtime (references/d3/
make_media_reference.py): every question's prompt token for token, with each image's and each frame pair's rows and
timestamps; every id's multimodal position; each image's and video's pixels; the frames OpenCV samples; the usage; a
request whose videos are over the creator's budget refused. No tower runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from ippon.adapters.d3 import IMAGE_MAX_PIXELS, D3Format, load_videos
from ippon.adapters.qwen import MIN_PIXELS
from ippon.arch.common import media_digest, media_of
from ippon.arch.qwen35 import places
from ippon.errors import InvalidRequest
from ippon.media import load_rgb, qwen
from ippon.media.paths import Seen
from ippon.media.qwen import QwenImages, pair_digest
from ippon.model_dir import ModelDir
from ippon.readout import CodeReadout
from ippon.tokenizer import Tokenizer
from ippon.verify import media_files, reference_request
from media_references import check_image_requests, hub_files, load, no_tower

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("mini", "flash")
FILES = ("tokenizer.json", "config.json", "decision_config.json")
CONFIGS = ("preprocessor_config.json", "video_preprocessor_config.json")


def d3_format(reference: dict[str, Any], tmp_path: Path) -> D3Format:
    """d3's format with the checkpoint's image and video paths, from its small files (no weights, no tower)."""
    files = hub_files(reference["model"], (*FILES, *CONFIGS))
    for name in ("config.json", *CONFIGS):
        (tmp_path / name).write_bytes(files[name].read_bytes())
    config = json.loads(files["config.json"].read_text())
    codes = tuple(json.loads(files["decision_config.json"].read_text())["codes"])
    tok = Tokenizer.from_file(files["tokenizer.json"])
    image = qwen.ImageConfig.from_hf(json.loads(files["preprocessor_config.json"].read_text()))
    images = QwenImages(image.pixels(MIN_PIXELS, IMAGE_MAX_PIXELS), False, config["image_token_id"], no_tower)
    assert images.token == tok.token_id("<|image_pad|>")
    videos = load_videos(ModelDir.open(tmp_path), images, tok)
    assert videos is not None
    return D3Format(tok, CodeReadout(mx.zeros((255, 1)), codes, 1.0), images, videos)


@pytest.mark.parametrize("size", SIZES)
def test_pictures_are_read_where_and_as_the_maker_reads_them(size: str, tmp_path: Path) -> None:
    """media-qwen.json's requests: EXIF ignored, every decoder's edge case, under the floor, a 12 MP photo cut to
    1.6 MP, 50:1 and 1:8 aspects, the same image twice, a long state, each question its own prompt."""
    reference = load(ROOT / f"references/d3/d3-{size}-media.json.gz")
    fmt = d3_format(reference, tmp_path)
    assert fmt.images is not None
    assert check_image_requests(fmt, reference, fmt.images.config) >= 25


@pytest.mark.parametrize("size", SIZES)
def test_videos_are_read_where_and_as_the_maker_reads_them(size: str, tmp_path: Path) -> None:
    """media-video.json's requests: MP4, QuickTime, WebM and Matroska; 4 to 32 frames sampled, an odd count's last
    twice, a container promising frames it doesn't hold; frames scaled up and down; images before videos; six long
    clips over the 16,384-row budget, refused."""
    path = ROOT / f"references/d3/d3-{size}-video.json.gz"
    reference = load(path)
    fmt = d3_format(reference, tmp_path)
    videos = fmt.videos
    assert videos is not None
    assert fmt.images is not None
    tokens = reference["media_format"]["image_token"], reference["media_format"]["video_token"]
    files = media_files(path, reference)
    checked = 0
    for case in reference["requests"]:
        request = reference_request(case, files(case))
        if "error" in case:
            with pytest.raises(InvalidRequest, match="16,384 tokens"):
                fmt.render(request)
            continue
        rendering = fmt.render(request)
        pictures = {media_digest(m.sha256) for m in request.images}
        grids: dict[int, tuple[int, int, int]] = {}
        seen: dict[str, list[Any]] = {"grids": [], "pixels": []}
        for m in request.images:
            t, h, w = qwen.preprocess(load_rgb(m, exif=False), fmt.images.config).grid
            grids[media_digest(m.sha256)] = (t, h // 2, w // 2)
        frames: list[dict[str, Any]] = []
        shown: dict[str, list[Any]] = {"grids": [], "pixels": []}
        for m in request.videos:
            s = videos.sampling
            source = videos.decoder(m, s.rate, s.least, s.most, None)  # at the source's size
            frames.append({"indices": list(source.indices), "fps": source.fps, "total": source.total})
            frames[-1]["size"] = [source.frames.shape[2], source.frames.shape[1]]
            clip = videos.decode(m)  # each frame resized as it was decoded: the creator's pixels all the same
            patches = qwen.preprocess_video(clip.frames, videos.config)
            shown["grids"].append(list(patches.grid))
            shown["pixels"].append(hashlib.sha256(patches.pixels.astype("<f4").tobytes()).hexdigest())
            t, h, w = patches.grid
            grids |= {pair_digest(m.sha256, i): (1, h // 2, w // 2) for i in range(t)}
        assert frames == case["frames"], case["name"]
        if request.images:
            for m in request.images:
                patches = qwen.preprocess(load_rgb(m, exif=False), fmt.images.config)
                seen["grids"].append(list(patches.grid))
                seen["pixels"].append(hashlib.sha256(patches.pixels.astype("<f4").tobytes()).hexdigest())
        prompts = zip(
            rendering.prompts, case["prompts"], case["positions"], case["images"], case["videos"], strict=True
        )
        for prompt, ids, positions, images, clips in prompts:
            got = [t if t >= 0 else tokens[media_of(t)[0] not in pictures] for t in prompt.tokens]
            assert got == ids, case["name"]
            assert places(np.array(prompt.tokens, np.int64), 0, 0, grids)[0].tolist() == positions, case["name"]
            assert images == seen, case["name"]
            assert clips == shown, case["name"]
            checked += 1
        assert fmt.input_tokens(rendering) == case["input_tokens"], case["name"]
        assert len(rendering.videos) == len(request.videos)
        for info, frame in zip(rendering.videos, frames, strict=True):
            assert info["frames_read"] == len(set(frame["indices"]))
            assert info["duration_s"] == round(frame["total"] / frame["fps"], 1)
    # A 10-minute video capped at 32 frames reports frames_read, duration_s, capped=True and seconds_per_frame.
    long_summary = videos.summary([Seen((0.0,), (16, 4, 4), frames=32, duration_s=600.0)])
    assert long_summary == ({"frames_read": 32, "duration_s": 600.0, "capped": True, "seconds_per_frame": 18.8},)
    assert checked >= 18
