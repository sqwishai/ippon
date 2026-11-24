"""elephant-1: its prompts, each question's block boundary and its usage against its creator's runtime (OneJev's qev
with the checkpoint's serve.patch, references/elephant), with special-token text, no state, object instructions and
two-letter labels; its slot labels as qev probes them; its pictures where the creator places them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from adapter_fixtures import renderings, snapshot
from ippon.adapters.elephant import LABELS, SYSTEM, ElephantFormat
from ippon.adapters.qwen import chat
from ippon.errors import InvalidRequest
from ippon.media import load_rgb
from ippon.media.qwen import ImageConfig, QwenImages
from ippon.readout import answer_tokens
from ippon.tokenizer import Tokenizer
from ippon.verify import load_reference, media_files, reference_request
from media_references import load, no_tower

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/elephant/elephant-1.json.gz"
MEDIA = ROOT / "references/elephant/elephant-1-media.json.gz"


def elephant_format() -> ElephantFormat:
    model = load_reference(REFERENCE)["model"]
    d = snapshot(model["repo"], model["revision"], "tokenizer.json", "tokenizer_config.json")
    tok = Tokenizer.from_file(d / "tokenizer.json", json.loads((d / "tokenizer_config.json").read_text()))
    slots = answer_tokens(tok, chat(SYSTEM, "<state>\n\n</state>\n\nQuestion: q"), LABELS)
    assert slots[:26] == tuple(range(32, 58))  # "A".."Z", as qev's engine probes them
    return ElephantFormat(tok, None, slots, {})


def test_prompts_blocks_and_usage_are_the_makers() -> None:
    for case, rendering in renderings(elephant_format(), load_reference(REFERENCE)["requests"], 27):
        assert all(p.marks == (case["prefix_tokens"],) for p in rendering.prompts), case["name"]
        assert all(r.position == len(p.tokens) - 1 for p in rendering.prompts for r in p.readouts)


def test_pictures_sit_in_the_states_block_where_the_maker_places_them() -> None:
    """The image requests against the creator's training and evaluation path (train.common and train.mm, as
    references/elephant runs it): each picture at its <image:N> in a text or a JSON state, at the state's end or as
    all of it, or at the state's start when it names none; its rows and pixels (no EXIF turn, alpha dropped, the
    processor's own range up to 16.7 MP); the state's block ending after the pictures; the usage. Placeholders out of
    order or naming too few pictures are refused, as the creator refuses them. No tower runs."""
    found = load(MEDIA)
    d = snapshot(found["model"]["repo"], found["model"]["revision"], "processor_config.json")
    image = ImageConfig.from_hf(json.loads((d / "processor_config.json").read_text()))
    fmt = elephant_format()
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
        assert all(p.marks == (case["prefix_tokens"],) for p in rendering.prompts), case["name"]
        assert fmt.usage(rendering)["input_tokens"] == case["input_tokens"], case["name"]
        patches = [fmt.images.patches(load_rgb(m, exif=False)) for m in request.images]
        assert [list(p.grid) for p in patches] == case["image_grids"], case["name"]
        pixels = np.concatenate([p.pixels for p in patches]).astype("<f4").tobytes()
        assert hashlib.sha256(pixels).hexdigest() == case["pixel_values_sha256"], case["name"]
