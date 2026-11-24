"""The suite runs every step of every model it finds, records failures and goes on, and writes its summary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ippon import bench, processes, suite
from ippon._portable import record


def test_every_found_model_gets_every_step_and_a_missing_one_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    busy = bench.Cell(100, 1, (12.5, 13.0), 140, busy=("editor (73%)", "build (104%)"))
    table = bench.table([busy])  # what bench prints, naming what else ran while it timed

    def fake(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        args = command[len(processes.command()) :]
        if args[0] == "verify":
            body = {"passed": "--invariance" not in args, "worst": 0.002}
            return subprocess.CompletedProcess(command, 0 if body["passed"] else 1, json.dumps(body), "")
        if args[0] == "bench":
            cell = {"scenario": "fresh", "words": 300, "questions": 1, "input_tokens": 400, "median_ms": 30.0,
                    "median_ci_ms": [29.0, 31.0], "p95_ms": 33.0}  # fmt: skip
            if "--compare" in args:
                arm = {"median_ms": 45.0, "speedup": 1.5, "verdict": "faster"}
                cell |= {"ippon": {"median_ms": 30.0}, "arms": {"mlx-vlm-model": arm}}
            Path(args[args.index("--out") + 1]).write_text(json.dumps({"cells": [cell]}))
            return subprocess.CompletedProcess(command, 0, table, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(suite, "spawn", fake)
    monkeypatch.setattr(suite, "fitting", lambda path: ("mlx-vlm-model",) if path.name == "d1-Omni-2B" else ())
    models, out = tmp_path / "models", tmp_path / "out"
    (models / "d1-2B").mkdir(parents=True)
    (models / "d3-mini").mkdir(parents=True)
    (models / "d1-Omni-2B").mkdir(parents=True)
    only = ["d1-2b-bf16", "d3-mini", "d1-omni-2b-bf16", "pplx-flash-bf16"]  # a build, or every build of a family
    report = suite.run(
        str(models), str(out), arms=["mlx-vlm-model=/x/python"], only=only, log=lambda m: None, kernels=True
    )
    missing = [s for s in report["steps"] if s["model"] == "pplx-flash-bf16"]
    folder = models / "pplx-decider-flash-v1.3"
    assert missing == [{"model": "pplx-flash-bf16", "step": "find", "ok": False, "note": f"no {folder}"}]
    steps = [(s["model"], s["step"]) for s in report["steps"]]
    for model in ("d1-2b-bf16", "d3-mini-bf16"):
        assert [s for m, s in steps if m == model] == [
            "verify", "invariance", "ab-kernels", "bench", "bench-short", "bench-repeat"
        ]  # fmt: skip
    assert next(s for s in report["steps"] if s["step"] == "invariance")["passed"] is False  # recorded, and on
    # a model an arm fits is compared with it
    assert [s for m, s in steps if m == "d1-omni-2b-bf16"] == [
        "verify", "invariance", "ab-kernels", "bench", "bench-short", "bench-repeat", "compare", "compare-repeat",
    ]  # fmt: skip
    path = str(models / "d1-Omni-2B")
    ran = {
        c[c.index("--out") + 1]: c[len(processes.command()) :] for c in calls if "--out" in c
    }  # each bench step by the file it writes
    file, counts = str(out / "d1-omni-2b-bf16"), ["--n", "20", "--rounds", "4"]
    assert ran[f"{file}.ab-kernels.json"] == [
        "bench", path, "--sizes", "100,1000,4000", "--questions", "1,5", "--n", "10", "--rounds", "4",
        "--ab", "--switch gemm=mlx", "--switch gemm=kernel", "--out", f"{file}.ab-kernels.json",
    ]  # fmt: skip
    assert ran[f"{file}.compare.json"] == [
        "bench", path, "--sizes", "300,1000,4000", "--questions", "1,5", *counts,
        "--compare", "mlx-vlm-model=/x/python", "--out", f"{file}.compare.json",
    ]  # fmt: skip
    assert ran[f"{file}.compare-repeat.json"] == [
        "bench", path, "--sizes", "1000", "--questions", "3", "--scenario", "repeat", *counts,
        "--compare", "mlx-vlm-model=/x/python", "--out", f"{file}.compare-repeat.json",
    ]  # fmt: skip
    plain = suite.run(str(models), str(out / "m1"), only=["d1-2b-bf16"], log=lambda m: None, kernels=False)
    skipped = next(s for s in plain["steps"] if s["step"] == "ab-kernels")
    assert skipped["note"].startswith("skipped: no neural accelerators")  # both arms would be MLX's own matmuls
    summary = (out / "summary.md").read_text()
    assert "| d1-2b-bf16 | verify | passed, worst 0.0020 |" in summary
    assert "## d1-2b-bf16.bench" in summary
    assert "Speed change: positive means ippon is faster." in summary
    assert "ippon 30.0 ms, mlx-vlm-model 45.0 ms, speed change +50.0% (faster)" in summary
    log = (out / "d1-2b-bf16.bench.log").read_text()  # a saved transcript names no other process
    assert log == table.replace("editor (73%)", "process (73%)").replace("build (104%)", "process (104%)") + "\n"
    published = json.loads((out / "suite.json").read_text())
    assert published["steps"] == record(report)["steps"]
    missing_note = next(step["note"] for step in published["steps"] if step["model"] == "pplx-flash-bf16")
    assert missing_note.startswith("no <tmp>/")
    assert str(models) in missing[0]["note"]  # the caller's live report keeps its usable path


def test_a_build_past_its_bound_as_published_passes_at_its_recorded_distance_and_fails_beyond() -> None:
    """d1-Omni-2B's Q8_0 file is 0.045 from the fp32 reference by its own quantization: a verify at 0.0442 matches
    the record (testers' suites had marked it FAILED on every Mac); 0.06 is past it, and stays a failure."""
    recorded = next(m for m in suite.SUITE if m.name == "d1-omni-2b-q8_0").recorded
    near = {"ok": False, "passed": False, "worst": 0.0442}
    suite.as_recorded(near, recorded)
    assert near == {"ok": True, "passed": True, "worst": 0.0442, "recorded": 0.045}
    far = {"ok": False, "passed": False, "worst": 0.06}
    suite.as_recorded(far, recorded)
    assert far["passed"] is False
