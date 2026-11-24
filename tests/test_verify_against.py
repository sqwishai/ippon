"""verify --against: ippon's answers on a build compared with another runtime's on the same requests, the arm
run as bench --compare runs it (here a fake arm answering what ippon answers, shifted)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ippon import Decider, compare_arms, processes, verify
from ippon.compare_arms import answers, lookup
from ippon.errors import DecisionError
from toy import decider as toy

QUESTIONS: dict[str, Any] = {
    "refund": {"type": "noul", "instructions": "Is a refund asked for?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "sales", "support"]},
}
CASES = [
    {"name": "ticket", "request": {"state": "Charged twice for one order. " * 6, "questions": QUESTIONS}},
    {"name": "parcel", "request": {"state": "Where is my parcel? " * 8, "questions": {"refund": QUESTIONS["refund"]}}},
]


def arm(
    monkeypatch: pytest.MonkeyPatch,
    shift: float = 0.0,
    published: float | None = None,
    rename: bool = False,
) -> list[tuple[list[str], dict[str, str] | None]]:
    """A fake arm answering ippon's own answers, `shift` moved from the first option to the second; with
    `published`, an arm that also computes in fp32 (ARM_DTYPE), moved by `shift` there and by `published` as
    published. The calls and their environments."""
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    own = toy()

    def spawn(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        calls.append((command, env))
        exact = env is not None and env.get(compare_arms.ARM_DTYPE) == "float32" and published is not None
        moved = shift if exact or published is None else published
        results = []
        for item in json.loads(Path(command[-2]).read_text()):
            response = own.systemone(item["body"])
            got = answers(response)
            for options in got.values():
                first, second = list(options)[:2]
                options[first] -= moved
                options[second] += moved
            if rename:
                got = {q: {f"{o}!": p for o, p in options.items()} for q, options in got.items()}
            results.append({"ms": 1.0, "answers": got, "input_tokens": response["usage"]["input_tokens"]})
        dtype = "float32" if exact else None
        Path(command[-1]).write_text(json.dumps({"runtime": "fake 1.0", "dtype": dtype, "results": results}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(processes, "spawn", spawn)
    monkeypatch.setattr(lookup, "arms_for", lambda model, wanted: {"fake": ("/x/python", "/x/fake.py")})
    monkeypatch.setattr(Decider, "load", lambda *args, **kwargs: toy())
    return calls


def test_an_arm_without_fp32_is_compared_as_published_within_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asked for fp32 once, an arm that answers as published is compared as published, against BOUND unless a
    bound is given; answers past it fail, naming the arm."""
    calls = arm(monkeypatch)
    report = verify.verify_against("m", "fake=/x/python", CASES)
    ((command, env),) = calls
    assert command == ["/x/python", "-I", "/x/fake.py", "m", command[-2], command[-1]]
    assert env is not None
    assert env[compare_arms.ARM_DTYPE] == "float32"
    assert report.passed
    assert report.worst < 1e-6
    assert report.bound == verify.BOUND
    assert "against fake (fake 1.0; it has no fp32 run): 2 requests" in report.text()
    arm(monkeypatch, shift=0.02)
    report = verify.verify_against("m", "fake=/x/python", CASES)
    assert not report.passed
    assert abs(report.worst - 0.02) < 1e-6
    assert all("from fake (fake 1.0; it has no fp32 run) (bound 0.01)" in f for f in report.failures)
    assert verify.verify_against("m", "fake=/x/python", CASES, bound=0.05).passed


def test_the_bound_is_the_arms_own_precision_from_its_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    """ippon 0.02 from the arm's fp32: past the floor when the arm's published run is 0.005 from its fp32, within
    it when 0.03 from it."""
    calls = arm(monkeypatch, shift=0.02, published=0.025)
    near = verify.verify_against("m", "fake=/x/python", CASES)
    assert [(env or {}).get(compare_arms.ARM_DTYPE) for _, env in calls] == ["float32", None]
    assert near.bound == verify.BOUND
    assert not near.passed
    assert "fake as published (fake 1.0) is 0.00500 from its fp32" in near.text()
    arm(monkeypatch, shift=0.02, published=0.05)
    far = verify.verify_against("m", "fake=/x/python", CASES)
    assert far.bound == 0.0305
    assert far.passed
    assert far.against == "fake in fp32 (fake 1.0)"


def test_other_options_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    arm(monkeypatch, rename=True)
    with pytest.raises(DecisionError, match="other questions or options"):
        verify.verify_against("m", "fake=/x/python", CASES)
