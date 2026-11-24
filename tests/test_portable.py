"""Published evidence must not contain an author's local filesystem roots."""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

from ippon import bench
from ippon._portable import dumps
from ippon.processes import run_json
from ippon.verify import Report


def test_tracked_text_has_no_absolute_personal_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    pattern = re.compile(r"/(?:Users|home|private)/|/(?:var)/folders/")
    found = []
    for name in filter(None, files):
        if not (root / name).is_file():  # an unstaged removal is no longer published text
            continue
        raw = (root / name).read_bytes()
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(content):
            found.append(name)
    assert not found, "Local paths in tracked text: " + ", ".join(found)


def test_result_serialization_preserves_numbers_and_embedded_json(tmp_path: Path) -> None:
    home = Path("/") / "Users" / "example"
    temporary = Path("/") / "private" / "var/folders/ab/random/T"
    embedded = {"model": str(home / "models/a"), "answer": 0.7803078889846802}
    original = {
        "content": json.dumps(embedded),
        "command": ["python", str(home / "a script.py"), "CACHE=" + str(temporary / "run")],
        "linux": str(Path("/") / "home" / "example/.cache/model"),
        "cache": str(home / "Library/Caches/model"),
        "scratch": str(Path("/") / "private" / "tmp/run"),
        "metrics": [0, -0.0, 1e-20, 2**64, True, None],
        "repo": "/".join(("https://example.org", "home", "model")),
    }
    out = tmp_path / "bench.json"
    bench.save(str(out), original)
    result = json.loads(out.read_text())
    assert result == json.loads(dumps(original))
    report = Report(str(home / "models/a"), 0.01, (), against=str(temporary / "reference.json"))
    assert report.to_json()["model"] == "~/models/a"
    assert report.to_json()["against"] == "<tmp>/reference.json"
    assert json.loads(result["content"]) == embedded | {"model": "~/models/a"}
    assert result["command"] == ["python", "~/a script.py", "CACHE=<tmp>/run"]
    assert result["linux"] == result["cache"] == "<cache>/model"
    assert result["scratch"] == "<tmp>/run"
    assert result["metrics"] == original["metrics"]
    assert result["repo"] == original["repo"]
    assert original["content"] == json.dumps(embedded)


def test_busy_names_stay_live_but_not_in_published_results(tmp_path: Path) -> None:
    private = tmp_path / "child.json"
    cell = bench.Cell(100, 1, (12.5, 13.0), 140, busy=("editor (73%)", "build (104%)"))
    original = {"cells": [cell.to_json()], "activity": ["editor (73%)"], "note": "confidence (95%)"}
    command = [
        sys.executable,
        "-c",
        "import json, sys; from ippon.bench import save; from ippon._portable import record, private_report; "
        "save(sys.argv[1], record(json.loads(sys.argv[2]), process_names=private_report()))",
        str(private),
        json.dumps(original),
    ]
    received = run_json(command, private, "measurement child")
    assert received == original
    assert "editor (73%)" in bench.table([cell])  # live output names them; suite's saved logs don't (test_suite)
    public = tmp_path / "published.json"
    bench.save(str(public), received)
    published = json.loads(public.read_text())
    assert published == original | {
        "cells": [cell.to_json() | {"busy": ["process (73%)", "process (104%)"]}],
        "activity": ["process (73%)"],
    }
    assert received == original
