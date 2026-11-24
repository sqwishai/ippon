"""bench's requests, statistics and A/B comparison: the same seed gives the same requests, each scenario asks what it
says, and the verdict follows the promotion rule."""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import pytest

from ippon import ab, bench, machine, processes, stats
from ippon.cli import main
from toy import decider as toy


def test_the_requests_depend_only_on_the_seed_and_each_scenario_asks_what_it_says() -> None:
    """Two runs with the same seed time the same requests (ab and compare pair their answers on it): fresh asks the
    same questions of a new state each time, repeat new questions of one state, grow a state that only grows."""
    for scenario in bench.SCENARIOS:
        assert list(bench.bodies(scenario, 30, 2, 7, 4)) == list(bench.bodies(scenario, 30, 2, 7, 4))
        assert list(bench.bodies(scenario, 30, 2, 7, 4)) != list(bench.bodies(scenario, 30, 2, 8, 4))
    fresh = list(bench.bodies("fresh", 30, 2, 1, 3))
    assert len({b["state"] for b in fresh}) == 3
    assert all(b["questions"] == fresh[0]["questions"] for b in fresh)
    repeat = list(bench.bodies("repeat", 30, 2, 1, 3))
    assert len({b["state"] for b in repeat}) == 1
    assert len({json.dumps(b["questions"]) for b in repeat}) == 3
    grow = [b["state"] for b in bench.bodies("grow", 30, 2, 1, 3)]
    assert all(later.startswith(earlier) and len(later) > len(earlier) for earlier, later in itertools.pairwise(grow))


def test_repeated_and_grown_states_reuse_what_was_read() -> None:
    """Fresh cells reuse nothing, the next cell included (each cell's seeds follow the last's); repeat and grow reuse
    the state on every timed request."""
    d = toy()
    fresh = bench.run(d, bench.Spec((40,), (1, 2), "fresh", n=4, warmup=1))
    repeat = bench.run(d, bench.Spec((40,), (2,), "repeat", n=4, warmup=1))
    grow = bench.run(d, bench.Spec((40,), (2,), "grow", n=4, warmup=1))
    assert [(c.words, c.questions, len(c.milliseconds), c.kept_hits) for c in fresh] == [(40, 1, 4, 0), (40, 2, 4, 0)]
    assert repeat[0].kept_hits == 4
    assert grow[0].kept_hits == 4
    assert len(repeat[0].answers) == 4
    assert all(len(a) >= 2 * 2 for a in repeat[0].answers)  # two questions of two options or more


def test_a_cell_reports_percentiles_and_an_interval_of_the_median() -> None:
    cell = bench.Cell(10, 1, tuple(float(t) for t in range(1, 21)), 5)
    data = cell.to_json()
    assert data["median_ms"] == 10.5
    assert data["p95_ms"] == pytest.approx(19.05)
    lo, hi = data["median_ci_ms"]
    assert lo <= 10.5 <= hi
    assert stats.quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def ms(center: float, spread: float = 0.5) -> list[float]:
    return [center + spread * ((i * 7) % 11 - 5) / 5 for i in range(20)]


def arm(spec: str, cells: dict[tuple[str, int, int, float], list[float]]) -> ab.Arm:
    a = ab.Arm.parse(spec)
    a.add({"cells": [{"scenario": k[0], "words": k[1], "questions": k[2], "input_tokens": 1, "ms": v,
                      "answers": [[0.5, 0.5]] * len(v), "busy": []} for k, v in cells.items()]})  # fmt: skip
    return a


def test_the_verdict_follows_the_promotion_rule() -> None:
    key, other = ("fresh", 300, 1, 0.0), ("fresh", 1000, 1, 0.0)
    a = arm("", {key: ms(100), other: ms(200)})
    faster = arm("--switch glu=mlx", {key: ms(90), other: ms(200)})
    rows = ab.compare(a, faster)
    assert [r["verdict"] for r in rows] == ["faster", "same"]
    assert ab.verdict(rows) == "promote"
    assert rows[0]["max_probability_difference"] == 0.0
    text = ab.table(a, faster, rows)
    assert "Time change: B relative to A; negative means B is faster." in text
    assert "| time change | 95% CI | p95 time change |" in text
    assert "| 100.0 | 90.0 | -10.0% |" in text
    slower = arm("", {key: ms(90), other: ms(210)})
    assert ab.verdict(ab.compare(a, slower)) == "mixed"
    assert ab.verdict(ab.compare(a, arm("", {key: ms(100.2), other: ms(200)}))) == "no difference"
    # a faster median can't hide a significantly slower tail (the p95's interval clear of 1)
    assert ab.judge(0.90, (0.88, 0.92), 1.10, (1.05, 1.15)) == "slower"
    assert ab.judge(0.90, (0.88, 0.92), 1.10, (0.98, 1.15)) == "faster"


def test_the_ab_arms_share_every_option_but_their_own(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each arm's process gets the bench options given, once, and its own; never --ab, --rounds or the run's --n."""
    commands: list[list[str]] = []

    def fake(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        n = int(command[command.index("--n") + 1])
        cell = {"scenario": "repeat", "words": 300, "questions": 1, "input_tokens": 9, "busy": [],
                "ms": ms(100)[:n], "answers": [[1.0]] * n}  # fmt: skip
        Path(command[command.index("--out") + 1]).write_text(json.dumps({"cells": [cell]}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(processes, "spawn", fake)
    monkeypatch.setattr(ab, "COOLDOWN", 0.0)
    argv = ["bench", str(tmp_path), "--sizes", "300", "--ab", "x", "y", "--n=8", "--rounds", "2"]
    assert main([*argv, "--precision", "fp16", "--scenario=repeat", "--seed", "4"]) == 0
    assert commands
    for command in commands:
        given = command[command.index(str(tmp_path)) + 1 : command.index("--n")]
        assert given == ["--precision", "fp16", "--sizes", "300", "--scenario", "repeat"]
        assert (command.count("--n"), command.count("--seed")) == (1, 1)
        assert command[-1] in ("x", "y")
        assert not {"--ab", "--rounds"} & set(command)


def test_ab_alternates_fresh_processes_on_the_same_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        assert env is not None
        calls.append((command, env))
        n = int(command[command.index("--n") + 1])
        slow = "--chunk" in command
        cell = {"scenario": "fresh", "words": 300, "questions": 1, "input_tokens": 9, "busy": [],
                "ms": ms(120 if slow else 100)[:n], "answers": [[1.0]] * n}  # fmt: skip
        Path(command[command.index("--out") + 1]).write_text(json.dumps({"cells": [cell]}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(processes, "spawn", fake)
    monkeypatch.setattr(ab, "COOLDOWN", 0.0)
    model = str(Path.home() / "models/example")
    cache = str(Path("/") / "private" / "tmp/run")
    text, data = ab.run(model, ["", f"X_Y=2 --chunk 8 IPPON_CACHE={cache}"], ["--sizes", "300"], n=20, rounds=4, seed=3)
    assert all(command[command.index("bench") + 1] == model for command, _ in calls)
    assert calls[1][1]["IPPON_CACHE"] == cache
    assert data["model"] == "~/models/example"
    assert data["b"]["spec"].endswith("IPPON_CACHE=<tmp>/run")
    assert len(calls) == 8
    order = ["--chunk" in c for c, _ in calls]
    assert order == [False, True, True, False, False, True, True, False]
    seeds = [c[c.index("--seed") + 1] for c, _ in calls]
    assert seeds[0] == seeds[1]
    assert seeds[2] == seeds[3]
    assert seeds[0] != seeds[2]
    assert calls[1][0][-2:] == ["--chunk", "8"]  # an arm's options after the shared ones, its NAME=value as environment
    assert calls[1][1]["X_Y"] == "2"
    assert "X_Y" not in calls[0][1]
    assert data["verdict"] == "reject"
    assert "| 100.1 | 120.1 | +20.0% |" in text
    assert "verdict for B: reject" in text


def test_the_revision_is_the_checkouts_or_the_one_its_wheel_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    here = Path(machine.__file__).parent
    head = subprocess.run(
        ["git", "-C", str(here), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    assert (machine.revision() or "").rstrip("+") == head.stdout.strip()
    installed = tmp_path / "venv" / "site-packages" / "ippon"  # a regular install, inside another repository
    installed.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    monkeypatch.setattr(machine, "__file__", str(installed / "machine.py"))
    assert machine.revision() is None  # never the other repository's commit
    (installed / "REVISION").write_text("abc1234\n")
    assert machine.revision() == "abc1234"


def test_a_report_records_the_environment_without_its_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Results get shared: the API key a server reads must not land in them, while the switches that change speed do."""
    monkeypatch.setenv("IPPON_API_KEY", "sk-private")
    monkeypatch.setenv("IPPON_SWITCHES", "glu=mlx")
    d = toy()
    spec = bench.Spec((20,), (1,), n=1, warmup=0)
    env = bench.report("toy", d, bench.run(d, spec), spec)["env"]
    assert env["IPPON_API_KEY"] == "<set>"
    assert env["IPPON_SWITCHES"] == "glu=mlx"
