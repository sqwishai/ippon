"""AutoTrust-Decider-Mini-9B's format against the creator's own System 1 code
(references/autotrust/decider-mini-9b.json.gz, from the model card's
transformers + peft example): every prompt token for token, and every request the card's code refuses, refused."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import snapshot_download

from ippon.adapters.autotrust import AutoTrustFormat
from ippon.errors import InvalidRequest
from ippon.schema import parse_request
from ippon.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references" / "autotrust" / "decider-mini-9b.json.gz"
REPO, REVISION = "autotrust/AutoTrust-Decider-Mini-9B", "7ac669b8fb476ba926491e3d1c39f9330d68dbed"


def decider_format() -> AutoTrustFormat:
    """The decider's format with its own tokenizer (from the Hub cache or downloaded; skips offline) and a head that
    isn't its, which rendering never reads."""
    try:
        directory = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["tokenizer*.json"]))
    except Exception as error:  # offline, rate-limited
        pytest.skip(f"The decider's tokenizer is not available: {error}")
    ranges = {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}
    head = mx.zeros((24, 1))
    return AutoTrustFormat(Tokenizer.from_file(directory / "tokenizer.json"), head, mx.zeros(24), ranges, {})


def test_prompts_and_refusals_match_the_creators_own_code() -> None:
    fmt = decider_format()
    reference = json.loads(gzip.decompress(REFERENCE.read_bytes()))
    for case in reference["requests"]:
        request = parse_request(case["request"])
        if "error" in case:
            with pytest.raises(InvalidRequest):
                fmt.render(request)
            continue
        rendering = fmt.render(request)
        assert [list(p.tokens) for p in rendering.prompts] == case["prompts"], case["name"]
        assert fmt.input_tokens(rendering) == case["input_tokens"], case["name"]
