"""Downloads: a model downloaded before loads from the Hub cache without asking the Hub, while every file of it is
still there, and every file of it from one commit; a fake Hub, so no network."""

from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pytest

from ippon import downloads, hub, locate, names, registry
from ippon.adapter import AdapterBase

FILES = {"config.json": "{}", "tokenizer/tokenizer.json": "{}", "model.safetensors": "w", "model.py": "code"}


def test_a_downloaded_model_loads_without_the_hub_while_its_files_are_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    branch = {"main": "a" * 40}
    asked: list[tuple[str, str | None]] = []

    def snapshot_download(
        repo: str, revision: str | None = None, allow_patterns: Any = (), ignore_patterns: Any = (), **options: Any
    ) -> str:
        asked.append((repo, revision))
        commit = revision if revision and len(revision) == 40 else branch[revision or "main"]
        folder = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
        for name, text in FILES.items():
            if any(fnmatch(name, p) for p in allow_patterns) and not any(fnmatch(name, n) for n in ignore_patterns):
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(text)
        return str(folder)

    monkeypatch.setattr(
        downloads,
        "fetch_weights",
        lambda model, patterns: Path(
            snapshot_download(
                names.split(model)[0],
                revision=names.split(model)[1],
                allow_patterns=patterns,
                ignore_patterns=downloads.NEVER,
            )
        ),
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    small, weights = ("*.json", "*.txt"), ("*.json", "*.safetensors")
    first = hub.fetch("org/model", small)  # Decider.load's two steps, twice: the Hub is asked the first time only
    assert hub.fetch("org/model", small) == first
    for _ in range(2):
        hub.fetch(hub.pinned("org/model", first), weights)
    assert asked == [("org/model", None), ("org/model", "a" * 40)]

    branch["main"] = "b" * 40  # the branch moves: loads stay at the commit downloaded
    assert hub.fetch("org/model", small) == first
    assert len(asked) == 2

    (first / "tokenizer" / "tokenizer.json").unlink()  # a file gone from the Hub cache: downloaded again, as it is now
    newer = hub.fetch("org/model", small)
    hub.fetch(hub.pinned("org/model", newer), weights)
    assert newer.name == hub.known("org/model") == "b" * 40
    assert len(asked) == 4
    assert hub.fetch("org/model", small) == newer
    assert hub.fetch(hub.pinned("org/model", newer), weights) == newer
    assert len(asked) == 4
    assert not (newer / "model.py").exists()  # code is never fetched

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)  # HF_HUB_OFFLINE=1: the cache's folder, never recorded
    for _ in range(2):
        hub.fetch("org/other", small)
    assert asked[4:] == [("org/other", None)] * 2

    record = tmp_path / "cache" / hub.DOWNLOADS  # offline, with the branch's commit unrecorded and the Hub cache's
    data = json.loads(record.read_text())  # refs/main naming a stray folder: the complete download ippon recorded
    data["resolved"].clear()
    record.write_text(json.dumps(data))
    calls = len(asked)
    assert hub.fetch("org/model", small) == newer
    assert len(asked) == calls


def test_every_adapter_s_files_can_be_downloaded_and_the_first_fetch_brings_what_matches_need() -> None:
    """A Hub model loads by name only when the download filter lets through every file its adapter names (GLiDE's
    render_tokenizer/ once failed it, as a bare "*"), and the first, small fetch brings each pickle an adapter
    recognises its model by (Kev's head.pt, without which no adapter matched)."""
    for name, adapter in registry.adapters().items():
        unsafe = [p for p in adapter.files if not p.endswith(hub.ALLOWED) and p not in adapter.read_safely]
        assert not unsafe, (name, unsafe)
        assert set(adapter.read_safely) <= set(hub.probe()), name


def test_an_unreadable_download_record_is_set_aside_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record names the snapshots `ippon remove all --models` may delete: a new download moves an unreadable one
    aside, as it was, before writing a fresh one."""
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
    record = tmp_path / "cache" / hub.DOWNLOADS
    record.parent.mkdir(parents=True)
    record.write_text('{"snapshots": {"org/model@')
    assert hub.known("org/model") is None  # reads as nothing downloaded
    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    hub.record_download("org/model", ("*.json",), snapshot)
    assert (tmp_path / "cache" / hub.UNREADABLE).read_text() == '{"snapshots": {"org/model@'
    assert hub.known("org/model") == "a" * 40


def test_downloads_finishing_at_once_each_keep_their_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two ippon processes finishing downloads at once (an agent's server loading while `ippon pull` runs) both
    read the record, add theirs and write it: each keeps its own, since the last write never drops another's."""
    import threading
    import time

    monkeypatch.setenv("IPPON_CACHE", str(tmp_path / "cache"))
    reading = hub._read_downloads

    def slow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        found = reading(*args, **kwargs)
        time.sleep(0.05)  # widens the window between reading the record and writing it back
        return found

    monkeypatch.setattr(hub, "_read_downloads", slow)
    models = [f"org/model-{i}" for i in range(12)]

    def finish(model: str) -> None:
        snapshot = tmp_path / "hub" / f"models--{model.replace('/', '--')}" / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        hub.record_download(model, ("*.json",), snapshot)

    threads = [threading.Thread(target=finish, args=(m,)) for m in models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [hub.known(m) for m in models] == ["a" * 40] * len(models)


def test_a_hub_model_reads_every_file_from_the_commit_its_first_download_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    snapshot = tmp_path / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    asked: list[str] = []

    def fetch(model: str, patterns: Any) -> Path:
        asked.append(model)
        return snapshot

    class Found(AdapterBase):
        files = ("*.safetensors",)

    monkeypatch.setattr(hub, "fetch", fetch)
    monkeypatch.setattr(registry, "select", lambda directory, adapter: Found())
    locate.open_model("org/name")
    assert asked == ["org/name", f"org/name@{commit}"]
    assert hub.pinned("org/name@main", snapshot) == f"org/name@{commit}"
    assert hub.pinned("org/name", tmp_path) == "org/name"  # not a snapshot: as asked
