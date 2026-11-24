"""Every answer says which artifact produced it: the build a load read, in `Result.diagnostics` and in what the server
says of its models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from ippon import Decider
from ippon.gguf import builds
from ippon.gguf.source import Gguf
from ippon.model_dir import ModelDir
from ippon.provenance import describe
from ippon.serve import Batcher
from test_decider import REQUEST
from test_generic_adapters import LABELS, qwen3_classifier
from test_serve import call, serving
from test_serve import running as running

COMMIT = "c8d48c08dada0191c6b362820f055cfb49c99426"  # the commit d1-Omni-2B is checked at (verified.py)


def test_an_answer_names_the_build_repository_and_commit_it_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running: tuple[str, Batcher, Decider]
) -> None:
    """A Hub snapshot is named by its repository and the commit read (and its short name, given by name, by Hub id
    or by the snapshot's own folder), a GGUF file by its sha256; the decider's answers carry the build in their
    diagnostics, and the server in its `?diagnostics=1` answers and its `/v1/models`."""
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
    folder = tmp_path / "hub" / "models--LiquidAI--d1-Omni-2B" / "snapshots" / COMMIT
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    snapshot = ModelDir.open(folder)
    expected = {
        "name": "d1-omni-2b:bf16",
        "repo": "LiquidAI/d1-Omni-2B",
        "revision": COMMIT,
        "base": None,
        "licence": {"name": "LFM Open License v1.0", "url": "https://huggingface.co/LiquidAI/d1-Omni-2B#license"},
    }
    for asked in ("d1-omni-2b", "d1-omni-2b:bf16", "LiquidAI/d1-Omni-2B", f"LiquidAI/d1-Omni-2B@{COMMIT}", str(folder)):
        assert describe(asked, snapshot) == expected, asked
    other = tmp_path / "hub" / "models--LiquidAI--d1-Omni-2B" / "snapshots" / ("9" * 40)  # a commit no check covers
    other.mkdir(parents=True)
    (other / "config.json").write_text("{}")
    unchecked = describe(f"LiquidAI/d1-Omni-2B@{'9' * 40}", ModelDir.open(other))
    assert (unchecked["name"], "checked at revision c8d48c08dada" in unchecked["unchecked"]) == (None, True)
    gguf = tmp_path / "mine.gguf"
    gguf.write_bytes(b"GGUF" + bytes(range(256)))
    build = builds.Build("org/mine-GGUF", "a" * 40, "mine.gguf", "0" * 64, "", "org/mine", "c" * 40, "lfm3")
    built = describe(gguf, ModelDir(tmp_path, {}, Gguf(tmp_path, gguf, build)))
    assert built == {
        "name": None,
        "path": str(gguf),
        "file": "mine.gguf",
        "sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
        "base": None,
        "licence": {"name": "not stated: read the model's page", "url": ""},
    }

    model = tmp_path / "classifier"
    model.mkdir()
    qwen3_classifier(model)
    loaded = Decider.load(model, keep_awake=None, pin_memory=False)  # the build is recorded by the load itself
    url, _, stop = serving(loaded)
    try:
        recorded = describe(str(model), ModelDir.open(model))
        assert recorded["path"] == str(model)
        questions: dict[str, Any] = {"team": {"type": "choice", "criteria": list(LABELS.values())}}
        assert loaded.decide(state="w1 w2", questions=questions).diagnostics["build"] == recorded
        _, body = call(url, "/v1/systemone?diagnostics=1", {"state": "w1 w2", "questions": questions})
        assert body["diagnostics"]["build"] == recorded
        _, models = call(url, "/v1/models")
        assert models["data"][0]["details"]["build"] == recorded
    finally:
        stop()
        loaded.close()
    _, _, made = running
    assert "build" not in made.decide(**REQUEST).diagnostics  # made from loaded parts: no artifact to name
