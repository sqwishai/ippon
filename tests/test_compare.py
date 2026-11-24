"""compare's requests, answer agreement, arm checks and rounds, with a stand-in for the arms' processes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ippon import ab, asks, bench, compare, compare_arms, processes
from ippon.compare_arms import lookup
from ippon.errors import DecisionError


def test_every_arm_gets_the_same_requests_in_a_round() -> None:
    spec = bench.Spec((30, 60), (1, 2), "fresh", n=3, warmup=1, seed=5, gaps=(0.0,))
    a = compare.requests(spec)
    assert a == compare.requests(spec)
    assert len(a) == 2 * 2 * (3 + 1)
    assert [x["warm"] for x in a[:4]] == [True, False, False, False]
    assert len({x["body"]["state"] for x in a}) == len(a)  # fresh: every state its own
    sent = [b for words, count, _, seed in spec.cells() for b in bench.bodies("fresh", words, count, seed, 4)]
    assert [x["body"] for x in a] == sent  # what bench sends for the same spec
    idle = compare.requests(bench.Spec((30,), (1,), "idle", n=2, warmup=0, seed=5, gaps=(1.0, 5.0)))
    assert [x["gap"] for x in idle] == [1.0, 1.0, 5.0, 5.0]


def test_agreement_compares_the_options_both_name() -> None:
    ours = [{"q0": {"yes": 0.9, "no": 0.1}, "q1": {"a": 0.4, "b": 0.6}}]
    theirs = [{"q0": {"yes": 0.88, "no": 0.12}, "q1": {"a": 0.55, "b": 0.45}}]
    got = compare_arms.agreement(ours, theirs)
    assert got["max_probability_difference"] == pytest.approx(0.15)
    assert got["top_answers_differ"] == 1
    assert got["unmatched"] == 0
    assert compare_arms.agreement(ours, [{"q0": {"true": 0.9, "false": 0.1}}])["unmatched"] == 2


def test_rounds_rotate_the_arms_and_pair_their_answers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        ours = "ippon.compare_arms.ours" in command
        at = command.index("ippon.compare_arms.ours") + 2 if ours else len(command) - 2
        requests, out = command[at], command[at + 1]
        items = json.loads(Path(requests).read_text())
        ms = 100.0 if ours else 130.0 if "mlx_vlm_model.py" in command[2] else 80.0
        results = [
            {"ms": ms + j % 3, "answers": {"q0": {"yes": 0.7, "no": 0.3}}, "input_tokens": 50}
            for j in range(len(items))
        ]
        Path(out).write_text(json.dumps({"runtime": "ours" if ours else "theirs", "results": results}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(processes, "spawn", fake)
    monkeypatch.setattr(ab, "COOLDOWN", 0.0)
    arms = {
        "mlx-vlm-model": ("/x/python", "/x/mlx_vlm_model.py"),
        "transformers": ("/x/python", "/x/transformers_byo.py"),
    }
    monkeypatch.setattr(lookup, "arms_for", lambda model, wanted: arms)
    monkeypatch.setattr(compare, "asker_for", lambda model: asks.standard)
    spec = bench.Spec((30,), (1,), "fresh", n=12, warmup=1, seed=1, gaps=(0.0,))
    text, data = compare.run(
        str(tmp_path), ["mlx-vlm-model=/x/python", "transformers=/x/python"], ["--precision", "fp16"], spec, 2
    )
    firsts = ["ours" in " ".join(c) for c in calls]
    assert firsts == [True, False, False, False, False, True]  # each round starts from another arm
    assert calls[0][-2:] == ["--precision", "fp16"]
    cell = data["cells"][0]["arms"]["mlx-vlm-model"]
    assert cell["speedup"] == pytest.approx(1.3, abs=0.02)
    assert cell["verdict"] == "faster"
    assert cell["max_probability_difference"] == 0.0
    assert data["cells"][0]["ippon"]["n"] == 12
    assert "speed change: positive means ippon is faster" in text
    assert "| speed change | 95% CI | p95 speed change |" in text
    assert "| 101.0 | 131.0 | +29.7% |" in text
    assert "ippon against mlx-vlm-model: 1 faster, 0 same, 0 slower" in text
    assert data["cells"][0]["arms"]["transformers"]["speedup"] == pytest.approx(0.8, abs=0.02)
    assert "| 101.0 | 81.0 | -19.8% |" in text
    assert "ippon against transformers: 0 faster, 0 same, 1 slower" in text  # the promotion rule, both ways


def test_concurrent_requests_are_refused(tmp_path: Path) -> None:
    """Every arm answers its requests one after another, so a concurrent cell would be timed as a sequential one."""
    spec = bench.Spec((30,), (1,), "concurrent", n=4, warmup=1)
    with pytest.raises(DecisionError, match="one after another"):
        compare.run(str(tmp_path), ["mlx-vlm-model=/x/python"], [], spec, 2)
