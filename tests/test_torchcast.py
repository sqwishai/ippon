"""Torchcast Decision: its prompts, with text the creator's tokenizer splits at combining marks and a prompt cut to the
16,384-token budget, against the creator's own serving package (references/torchcast); its letters as the package probes
them; its yes/no floor; its pictures where the package places them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from adapter_fixtures import question as q
from adapter_fixtures import renderings, snapshot
from ippon.adapters.qwen import MIN_PIXELS
from ippon.adapters.torchcast import BUDGET, MAX_PIXELS, TorchcastFormat, letter_ids
from ippon.arch.common import media_digest
from ippon.arch.qwen35 import places
from ippon.errors import InvalidRequest
from ippon.media import load_rgb
from ippon.media.qwen import ImageConfig, QwenImages
from ippon.tokenizer import Tokenizer
from ippon.verify import load_reference, media_files, reference_request
from media_references import load, no_tower

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/torchcast/torchcast-decision-4b.tokens.json.gz"
MEDIA = ROOT / "references/torchcast/torchcast-decision-4b-media.json.gz"


def torchcast_format() -> TorchcastFormat:
    model = load_reference(REFERENCE)["model"]
    d = snapshot(model["repo"], model["revision"], "tokenizer.json", "tokenizer_config.json", "decision_config.json")
    config = json.loads((d / "decision_config.json").read_text())
    tok = Tokenizer.from_file(d / "tokenizer.json", json.loads((d / "tokenizer_config.json").read_text()))
    letters = letter_ids(tok)
    assert list(letters) == config["letter_token_ids"]  # as the package checks them at load
    return TorchcastFormat(tok, None, letters, config["temperature_by_type"], config["noul_floor"])


def test_prompts_and_the_budget_cut_are_the_makers() -> None:
    found = load_reference(REFERENCE)
    assert [c["name"] for c in found["maker"]["left_out"]] == ["choice-40"]  # 26 options at most in one pass
    for case, rendering in renderings(torchcast_format(), found["requests"], 26):
        assert [r.question for p in rendering.prompts for r in p.readouts] == case["keys"], case["name"]
        assert all(r.position == len(p.tokens) - 1 for p in rendering.prompts for r in p.readouts)
        assert bool(rendering.cuts) == (case["name"] == "over-budget"), case["name"]
        assert max(len(p.tokens) for p in rendering.prompts) <= BUDGET


def test_a_weak_yes_or_no_moves_to_the_floor_and_keeps_its_answer() -> None:
    fmt = TorchcastFormat(None, None, (), {"noul": 0.9}, 0.801)  # type: ignore[arg-type]
    noul = q({"type": "noul"})
    for p_yes, committed in ((0.6, 0.801), (0.3, 1 - 0.801), (0.5, 0.801), (0.9, 0.9), (0.1, 0.1)):
        probabilities, answer = fmt.answer(noul, [p_yes, 1 - p_yes])
        assert answer == {"type": "noul", "noul": pytest.approx(committed)}
        assert probabilities == pytest.approx((p_yes, 1 - p_yes))  # the model's, which verify compares


def test_pictures_go_where_the_package_puts_them() -> None:
    """The image requests against the package's own decide (references/torchcast): each picture at its "<image>" mark
    in a text or a JSON state or as all of it, else ahead of the state ("Image 1: ..." lines for several); its rows,
    pixels (no EXIF turn, alpha dropped, at most 1,048,576 pixels: 12 MP photos scaled down) and every id's multimodal
    position; the usage. Marks that do not match the pictures are refused, as the package refuses them. No tower
    runs."""
    found = load(MEDIA)
    d = snapshot(found["model"]["repo"], found["model"]["revision"], "preprocessor_config.json")
    image = ImageConfig.from_hf(json.loads((d / "preprocessor_config.json").read_text())).pixels(MIN_PIXELS, MAX_PIXELS)
    fmt = torchcast_format()
    token = found["media_format"]["image_token"]
    fmt.images = QwenImages(image, False, token, no_tower)
    files = media_files(MEDIA, found)
    for case in found["requests"]:
        request = reference_request(case, files(case))
        if "error" in case:
            with pytest.raises(InvalidRequest):
                fmt.render(request)
            continue
        rendering = fmt.render(request)
        assert [[t if t >= 0 else token for t in p.tokens] for p in rendering.prompts] == case["prompts"], case["name"]
        assert fmt.usage(rendering)["input_tokens"] == case["input_tokens"], case["name"]
        patches = [fmt.images.patches(load_rgb(m, exif=False)) for m in request.images]
        assert [list(p.grid) for p in patches] == case["image_grids"], case["name"]
        pixels = np.concatenate([p.pixels for p in patches]).astype("<f4").tobytes()
        assert hashlib.sha256(pixels).hexdigest() == case["pixel_values_sha256"], case["name"]
        merged = [(p.grid[0], p.grid[1] // image.merge, p.grid[2] // image.merge) for p in patches]
        grids = {media_digest(m.sha256): g for m, g in zip(request.images, merged, strict=True)}
        for prompt, positions in zip(rendering.prompts, case["positions"], strict=True):
            assert places(np.array(prompt.tokens, np.int64), 0, 0, grids)[0].tolist() == positions, case["name"]
    assert [c["name"] for c in found["requests"] if "error" in c] == ["marks-too-few"]
