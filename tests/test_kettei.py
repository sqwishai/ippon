"""Kettei Core and Dot against the sqwishai SDK's own local code (references/kettei): every prompt's tokens, the
requests it refuses, and the tokens it counts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from adapter_fixtures import renderings, snapshot
from ippon import decide_api
from ippon.adapters.kettei import kettei_format
from ippon.errors import InvalidRequest
from ippon.model_dir import ModelDir
from ippon.schema import parse_request
from ippon.tokenizer import Tokenizer
from ippon.verify import load_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", ["dot", "core"])
def test_prompts_refusals_and_counts_are_the_sdks(size: str) -> None:
    found = load_reference(ROOT / f"references/kettei/kettei-{size}.json.gz")
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    model = ModelDir.open(snapshot(found["model"]["repo"], found["model"]["revision"], *files))
    tokenizer = Tokenizer.from_file(model.file("tokenizer.json"), model.json("tokenizer_config.json"))
    fmt = kettei_format(model, tokenizer, None)  # type: ignore[arg-type]
    for case in (c for c in found["requests"] if "error" in c):  # a question without instructions
        with pytest.raises(InvalidRequest):
            fmt.render(parse_request(case["request"]))
    one = {
        "state": "A page.",
        "questions": {"next": {"type": "choice", "instructions": "Which?", "criteria": {"done": "Done"}}},
    }
    with pytest.raises(InvalidRequest):  # one option: the SDK refuses it, and it once divided by zero here
        fmt.render(parse_request(one))
    renderings(fmt, [c for c in found["requests"] if "error" not in c], 20)


@pytest.mark.parametrize("size", ["dot", "core"])
def test_decide_requests_read_as_the_sdk_writes_them(size: str) -> None:
    """Sqwish's decide API (`/v1/decide`) on ippon: every decision's prompt, answer labels and the request's input
    tokens as sqwishai.kettei renders them (references/kettei/kettei-*-decide.json.gz, make_decide_reference.py)."""
    model_of = load_reference(ROOT / f"references/kettei/kettei-{size}.json.gz")["model"]
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    model = ModelDir.open(snapshot(model_of["repo"], model_of["revision"], *files))
    tokenizer = Tokenizer.from_file(model.file("tokenizer.json"), model.json("tokenizer_config.json"))
    fmt = kettei_format(model, tokenizer, None)  # type: ignore[arg-type]
    found = json.loads(gzip.decompress((ROOT / f"references/kettei/kettei-{size}-decide.json.gz").read_bytes()))[
        "requests"
    ]
    assert len(found) >= 9
    for case in found:
        systemone, decisions, _ = decide_api.to_systemone(case["request"])
        rendering = fmt.render(parse_request(systemone))
        assert [list(p.tokens) for p in rendering.prompts] == case["prompts"], case["name"]
        assert [list(rendering.meta[d.id]) for d in decisions] == case["labels"], case["name"]
        assert fmt.usage(rendering)["input_tokens"] == case["input_tokens"], case["name"]
