"""Holding a Qwen3.5 format's image prompts to a media reference made from its creator's code (pplx-decider's and
Clef-Flash's make_media_reference.py): token for token, position for position, pixel for pixel."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ippon.adapter import Rendering
from ippon.arch.common import media_digest
from ippon.arch.qwen35 import places
from ippon.errors import ContextTooLong
from ippon.media import load_rgb, qwen
from ippon.questions import with_media
from ippon.schema import parse_request

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "references/media/images"


def load(path: Path) -> dict[str, Any]:
    reference: dict[str, Any] = json.loads(gzip.decompress(path.read_bytes()))
    return reference


def hub_files(model: dict[str, Any], names: tuple[str, ...]) -> dict[str, Path]:
    """The checkpoint's small files, from the Hugging Face cache (downloaded once), or skip offline."""
    try:
        from huggingface_hub import hf_hub_download

        return {n: Path(hf_hub_download(model["repo"], n, revision=model["revision"])) for n in names}
    except Exception as error:  # offline, rate-limited
        pytest.skip(f"{model['repo']}'s files are not available: {error}")


def no_tower() -> qwen.QwenVision:
    """A format's tower loader for checking prompts: rendering never reads an image's rows."""
    raise AssertionError("rendering a prompt needs no vision tower")


def check_image_requests(fmt: Any, reference: dict[str, Any], image: qwen.ImageConfig) -> int:
    """Render every request of `reference` with its images' bytes and hold each prompt to the creator's: its ids, with
    ippon's placeholders where the creator's ids are the image token; every id's (t, h, w) position as the
    backbone places it (arch/qwen35.places, with each image's grid from media/qwen.preprocess, as the tower's rows
    carry it); each image's grid and pixel_values; the usage; Clef's spans. A request the creator refuses is refused.
    Returns how many prompts were checked."""
    token, checked = reference["media_format"]["image_token"], 0
    for case in reference["requests"]:
        files = {m["name"]: (IMAGES / m["name"]).read_bytes() for m in case["media"]}
        body = parse_request(dict(case["request"], images=None))
        request = with_media(body, [files[name] for name in case["request"]["images"]], None)
        if "error" in case:
            with pytest.raises(ContextTooLong):
                fmt.render(request)
            continue
        rendering: Rendering = fmt.render(request)
        grids: dict[int, tuple[int, int, int]] = {}
        seen: dict[str, list[Any]] = {"grids": [], "pixels": []}
        for media in request.images:
            patches = qwen.preprocess(load_rgb(media, exif=fmt.images.exif), image)
            t, h, w = patches.grid
            grids[media_digest(media.sha256)] = (t, h // image.merge, w // image.merge)
            seen["grids"].append([t, h, w])
            seen["pixels"].append(hashlib.sha256(patches.pixels.astype("<f4").tobytes()).hexdigest())
        prompts = zip(rendering.prompts, case["prompts"], case["positions"], case["images"], strict=True)
        for prompt, ids, positions, images in prompts:
            assert [t if t >= 0 else token for t in prompt.tokens] == ids, case["name"]
            assert places(np.array(prompt.tokens, np.int64), 0, 0, grids)[0].tolist() == positions, case["name"]
            assert seen == images, case["name"]
            checked += 1
        assert fmt.input_tokens(rendering) == case["input_tokens"], case["name"]
        if "readouts" in case:  # Clef's: each question's instruction and option spans, and option ids
            spans = [
                [list(s.question), [list(o) for o in s.options], list(s.option_ids)] for s in rendering.meta["spans"]
            ]
            assert spans == case["readouts"], case["name"]
    return checked
