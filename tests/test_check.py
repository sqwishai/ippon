"""`ippon check`: a model's adapter, download, memory per precision on a given Mac and what is recorded as
verified of it, without its weights; and the record docs/models.md's table is written from."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import huggingface_hub.constants
import mlx.core as mx
import pytest

from ippon import check as check_module
from ippon import hub, locate, registry
from ippon.adapter import AdapterBase
from ippon.device import Device
from ippon.gguf import builds
from ippon.gguf.header import read_header
from ippon.names import for_model
from ippon.safetensors import Tensor
from ippon.verified import BUILDS
from test_gguf import base_dir, tiny_gguf

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30


def mac(working_set: int, macos: tuple[int, ...] = (27, 0)) -> Device:
    return Device("Apple M5 Pro", "applegpu_g17s", 2 * working_set, working_set, macos, True)


class Found(AdapterBase):
    name, files = "acme/toy", ("config.json", "*.safetensors")


def test_a_models_builds_are_found_by_any_of_their_ids() -> None:
    d1 = [b.load for b in for_model("liquidai/d1-2b@main")]
    assert d1 == [
        "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4",
        "LiquidAI/d1-2B-GGUF:Q8_0",
        "ollama://d1:2b",
        "ollama://d1:2b-mxfp8",
    ]
    assert [b.load for b in for_model("ollama://d1:2b")] == d1
    assert [b.load for b in for_model(f"LiquidAI/d1-2B-GGUF@{'e' * 40}:Q8_0")] == d1  # a GGUF build at a revision
    assert for_model("acme/unknown") == ()
    for b in BUILDS:  # a build listed under another model would take that model's name and licence
        assert len({n.name.partition(":")[0] for n in for_model(b.load) if n.name}) == 1, b.load


def test_the_docs_table_is_the_record() -> None:
    spec = importlib.util.spec_from_file_location("models_table", ROOT / "scripts/models_table.py")
    assert spec is not None
    assert spec.loader is not None
    table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(table)
    page = (ROOT / "docs/models.md").read_text()
    assert f"{table.START}\n{table.table()}\n{table.END}" in page  # scripts/models_table.py docs/models.md rewrites it


def test_a_folder_is_checked_from_its_files_and_each_precision_against_this_mac_and_its_hub_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "toy"}')
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"w": mx.zeros((1024, 1024)), "e": mx.zeros((4096, 1024))})
    monkeypatch.setattr(registry, "select", lambda directory, adapter: Found())
    found = check_module.check(str(tmp_path), device=mac(4 * GIB, macos=(26, 1)))
    assert (found["adapter"], found["architecture"], found["download"]) == ("acme/toy", "toy", {"files": 0, "bytes": 0})
    memory = {m["precision"]: m for m in found["memory"]}
    assert memory["fp32"]["bytes"] == 2 * memory["bf16"]["bytes"] == 4 * 5 * 1024 * 1024
    assert memory["q8"]["changes_weights"]
    assert not memory["bf16"]["changes_weights"]
    assert set(memory) == {"auto", "bf16", "fp32", "q8", "q6", "q4"}
    assert memory["q4"]["low_bit"]
    assert not memory["q8"]["low_bit"]
    assert "drift from the creator's" in check_module.report(found)  # a low-bit line says it drifts
    assert found["verified"] == []
    assert "nothing recorded" in check_module.report(found)
    cached = tmp_path / "hub/models--LiquidAI--d1-2B/snapshots" / ("b" * 40)  # a Hub cache folder: its model's builds
    cached.mkdir(parents=True)
    for name in ("config.json", "model.safetensors"):
        (cached / name).write_bytes((tmp_path / name).read_bytes())
    assert (
        check_module.check(str(cached), device=mac(4 * GIB))["verified"][0]["load"]
        == "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4"
    )


def test_a_hub_model_is_checked_without_its_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "a" * 40
    snapshot = tmp_path / commit
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type": "lfm3_vl"}')
    asked: dict[str, Any] = {}

    def remote(model: str, patterns: tuple[str, ...]) -> tuple[list[Tensor], int, int]:
        asked["model"] = model
        return [Tensor("w", "BF16", (GIB // 2, 4))], 2, 4 * GIB  # 2 GiB of values, 4 GiB in bf16

    monkeypatch.setattr(hub, "fetch", lambda model, patterns: snapshot)
    monkeypatch.setattr(registry, "select", lambda directory, adapter: Found())
    monkeypatch.setattr(locate, "_remote", remote)
    found = check_module.check("LiquidAI/d1-2B", device=mac(5 * GIB))
    assert asked["model"] == f"LiquidAI/d1-2B@{commit}"  # the weights' headers at the commit read first
    assert found["revision"] == commit
    assert found["download"] == {"files": 2, "bytes": 4 * GIB}
    memory = {m["precision"]: m for m in found["memory"]}
    assert memory["bf16"]["fits"]
    assert not memory["fp32"]["fits"]  # 8 GiB, past 5 GiB less the room kept for a request
    assert found["verified"][0]["load"] == "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4"


def test_a_downloaded_gguf_build_is_checked_from_its_cached_file_as_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GGUF build in the Hub cache: its file's bytes from disk, the creator's small files only, and memory at the
    file's size for the precisions that keep its weights as stored."""
    cache = tmp_path / "hub"
    snapshot = cache / "models--org--tiny-GGUF/snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    text, _ = tiny_gguf(snapshot / "tiny-Q8_0.gguf")
    base = base_dir(tmp_path, text)
    digest = read_header(snapshot / "tiny-Q8_0.gguf").digest
    build = builds.Build("org/tiny-GGUF", "a" * 40, "tiny-Q8_0.gguf", "b" * 64, digest, "org/tiny", "c" * 40)
    monkeypatch.setattr(builds, "BUILDS", (build,))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    asked: list[tuple[str, tuple[str, ...]]] = []

    def fetch(model: str, patterns: Any) -> Path:
        asked.append((model, tuple(patterns)))
        return base

    monkeypatch.setattr(hub, "fetch", fetch)
    found = check_module.check("org/tiny-GGUF:Q8_0", device=mac(4 * GIB))
    size = (snapshot / "tiny-Q8_0.gguf").stat().st_size
    assert asked == [("org/tiny@" + "c" * 40, builds.SMALL_FILES)]
    assert (found["adapter"], found["revision"]) == ("liquid/d1", "a" * 40)
    assert found["download"] == {"files": 1, "bytes": size}
    assert {m["precision"]: m["bytes"] for m in found["memory"]} == {"auto": size, "bf16": size, "fp32": size}
    assert found["downloads"][0] == {
        "model": "org/tiny-GGUF@" + "a" * 40,
        "patterns": ["tiny-Q8_0.gguf"],
        "gguf": True,
        "sha256": "b" * 64,
    }
