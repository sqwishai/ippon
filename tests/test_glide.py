"""Fastino's GLiDE against the creator's own `glide` package (references/glide): every prompt's tokens and the
tokens it counts."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from adapter_fixtures import renderings, snapshot
from ippon.adapters.glide import GlideFormat, codes, decision_config
from ippon.model_dir import ModelDir
from ippon.readout import CODE_ROWS, CodeReadout
from ippon.tokenizer import Tokenizer
from ippon.verify import load_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("file", ["glide-2b", "glide-9b"])
def test_prompts_and_counts_are_the_packages(file: str) -> None:
    found = load_reference(ROOT / f"references/glide/{file}.json.gz")
    files = ("decision_config.json", "tokenizer.json", "tokenizer_config.json")
    model = ModelDir.open(snapshot(found["model"]["repo"], found["model"]["revision"], *files))
    config = decision_config(model)
    tokenizer = Tokenizer.from_file(model.file("tokenizer.json"), model.json("tokenizer_config.json"))
    readout = CodeReadout(mx.zeros((CODE_ROWS, 1)), tuple(codes(model, config, tokenizer)), 1.0)
    renderings(GlideFormat(tokenizer, readout), found["requests"], 20)
