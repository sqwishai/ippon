"""Qwen3.5's image preprocessing and positions (media/qwen.py) against transformers 5.17.0's own, run in the creators'
venv (tests/data/media/make_qwen_fixtures.py): smart_resize over 5,000 sizes under pplx-decider flash v1.2's and
Clef-Flash's pixel ranges, each image's pixel_values bit for bit, get_rope_index's (t, h, w) positions, and the
tower's resampled position table and axial RoPE angles; a video's frame pairs read by a tiny tower once."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from helpers import b64, pattern, sha
from ippon.adapter import Rendering
from ippon.errors import InvalidRequest
from ippon.media import as_media, load_rgb, qwen
from ippon.media.d3_video import load_frames
from ippon.media.paths import Sampling
from ippon.media.qwen import QwenImages, QwenVideos, pair_digest
from ippon.media.video import Clip
from ippon.schema import parse_request
from ippon.weights import Weights
from qwen35_numpy import rope_index
from qwen_vision_numpy import forward
from qwen_vision_numpy import tiny_config as tower_config
from qwen_vision_numpy import tiny_weights as tower_weights

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data/media"
FIXTURE = json.loads((DATA / "qwen_preprocess.json").read_text())
POSITIONS = np.load(DATA / "qwen_positions.npz")
# each checkpoint's processor_config.json, with the pixel range its creator's code sets (pplx-decider:
# decision_config's)
CONFIGS = {
    name: qwen.ImageConfig.from_hf(p).pixels(*FIXTURE["ranges"][name]) for name, p in FIXTURE["processors"].items()
}


def test_every_size_is_the_makers() -> None:
    """Both rounding branches, the floor and ceiling, halves rounding to even, and aspects over 200 refused."""
    wrong = []
    for width, height, *want in FIXTURE["sizes"]:
        for name, expected in zip(CONFIGS, want, strict=True):
            c = CONFIGS[name]
            try:
                got: list[int] | None = list(qwen.smart_resize(height, width, c.factor, c.min_pixels, c.max_pixels))
            except InvalidRequest:
                got = None
            if got != expected:
                wrong.append((width, height, name, got, expected))
    assert len(FIXTURE["sizes"]) > 5000
    assert not wrong


@pytest.mark.parametrize("case", FIXTURE["images"], ids=lambda c: "x".join(map(str, c["size"])))
@pytest.mark.parametrize("name", ["pplx", "clef"])
def test_preprocessing_is_the_makers(case: dict[str, Any], name: str) -> None:
    width, height = case["size"]
    image = pattern(height, width, case["seed"])
    assert sha(image) == case["image_sha256"]
    rgb = load_rgb(as_media(image, "image"), exif=False)  # an array goes through no decoder
    want, config = case[name], CONFIGS[name]
    if "error" in want:  # an aspect over 200
        with pytest.raises(InvalidRequest, match="aspect ratio is over 200"):
            qwen.preprocess(rgb, config)
        return
    patches = qwen.preprocess(rgb, config)
    assert list(patches.grid) == want["grid"]
    assert sha(patches.pixels.astype("<f4")) == want["pixel_sha256"]
    assert patches.rows() == qwen.image_tokens(width, height, config)


@pytest.mark.parametrize("case", FIXTURE["rope"], ids=lambda c: f"{len(c['grids'])}-images-pad-{c['pad']}")
def test_rope_positions_are_the_makers(case: dict[str, Any]) -> None:
    kinds = case["types"]
    starts = [i for i, k in enumerate(kinds) if k and (i == 0 or not kinds[i - 1])]
    images = [(s, tuple(g)) for s, g in zip(starts, case["grids"], strict=True)]
    positions, delta = rope_index(len(kinds), images)
    assert positions.tolist() == case["positions"]
    assert delta == case["delta"]


@pytest.mark.parametrize("grid", sorted(k[4:] for k in POSITIONS.files if k.startswith("pos.")))
def test_position_table_and_rope_angles_are_transformers(grid: str) -> None:
    """The resampled table bit for bit; the axial RoPE's cos and sin for pplx's 64-wide and Clef-Flash's 72-wide
    heads to one float32 ulp (the frequencies and angles are torch's exactly, but its vectorized cos and sin are not
    correctly rounded: about 7% of the values are one ulp off)."""
    h, w = map(int, grid.split("x"))
    np.testing.assert_array_equal(qwen.position_embeddings(POSITIONS["table"], h, w, 2), POSITIONS[f"pos.{grid}"])
    for hidden in (1024, 1152):
        c = qwen.VisionConfig(hidden=hidden, heads=16)
        cos, sin = qwen.rotary_angles(h, w, c.head_dim, c.theta, c.merge)
        np.testing.assert_allclose(cos, POSITIONS[f"cos.{hidden // 16}.{grid}"], rtol=0, atol=2**-24)
        np.testing.assert_allclose(sin, POSITIONS[f"sin.{hidden // 16}.{grid}"], rtol=0, atol=2**-24)


def test_a_video_asked_again_reuses_its_frame_pairs_rows(tmp_path: Path) -> None:
    """A tiny tower reads a clip's frame pairs once, each pair's rows the numpy tower's of that pair; the same video
    asked again is neither decoded nor read again, and its rows are the first answer's."""
    c = tower_config()
    w = tower_weights(c)
    mx.save_safetensors(str(tmp_path / "tower.safetensors"), {"visual." + k: mx.array(v) for k, v in w.items()})
    tower = qwen.QwenVision.load(Weights.from_dir(tmp_path, ["tower.safetensors"]), c)
    reads: list[int] = []

    def counted(images: Sequence[qwen.Patches]) -> mx.array:
        reads.append(len(images))
        return tower(images)

    image = qwen.ImageConfig.from_hf(
        {"image_processor_type": "Qwen2VLImageProcessor", "patch_size": 4, "merge_size": 2, "temporal_patch_size": 2}
        | {"size": {"shortest_edge": 256, "longest_edge": 4096}, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}
    )
    videos = QwenVideos(
        qwen.VideoConfig(image), Sampling(), 7, QwenImages(image, False, 6, lambda: counted), decoder=load_frames
    )
    clip = (ROOT / "references/media/videos/tiny-h264-24x16-4fps-3s.mp4").read_bytes()
    request = parse_request({"state": "s", "questions": {"q": {"type": "noul"}}, "videos": [b64(clip)]})
    video = request.videos[0]
    (seen,), clips = videos.seen(request.videos)  # as a format renders the request
    first = videos.encode([Rendering(request, (), clips=clips)])
    patches = qwen.preprocess_video(clips[video.sha256].frames, videos.config)
    t, h, wd = patches.grid
    expected = forward(c, w, [(patches.pixels, patches.grid)]).reshape(t, 1, h // 2, wd // 2, -1)
    assert seen.grid[0] == t > 1
    assert list(first) == [pair_digest(video.sha256, i) for i in range(t)]
    for i, rows in enumerate(first.values()):
        np.testing.assert_allclose(np.array(rows), expected[i], atol=1e-4)
    assert videos.seen(request.videos)[1] == {}  # seen before: not decoded again
    again = videos.encode([Rendering(request, ())])
    assert reads == [1]
    assert again.keys() == first.keys()
    assert all(mx.array_equal(again[d], first[d]).item() for d in first)

    # a long clip (70 frame pairs, as d1 samples a minute of video) is kept whole too: asked again, not read again
    frames = np.random.default_rng(0).integers(0, 256, (140, 16, 24, 3), np.uint8)
    long = QwenVideos(
        qwen.VideoConfig(image), Sampling(most=140), 7, QwenImages(image, False, 6, lambda: counted),
        decoder=lambda *_: Clip(frames, tuple(range(140)), 2.0, 140),
    )  # fmt: skip
    request = parse_request({"state": "s", "questions": {"q": {"type": "noul"}}, "videos": [b64(clip + b"long")]})
    for _ in range(2):
        (pairs,), clips = long.seen(request.videos)
        assert pairs.grid[0] == 70
        long.encode([Rendering(request, (), clips=clips)])
    assert reads == [1, 1]
