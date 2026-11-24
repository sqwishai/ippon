"""verify --invariance on the toy model, whose arithmetic every path does alike: every check runs and finds no
difference; a path that changed the answers would be caught."""

from __future__ import annotations

from typing import Any

from ippon import invariance
from toy import decider as toy

STATE = "The customer was charged twice for one order and wants the second charge back. " * 5
QUESTIONS: dict[str, Any] = {
    "refund": {"type": "noul", "instructions": "Is a refund asked for?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "sales", "support"]},
}
CASES = [
    {"name": "ticket", "request": {"state": STATE, "questions": QUESTIONS}},
    {
        "name": "other",
        "request": {"state": "Where is my parcel? " * 12, "questions": dict(list(QUESTIONS.items())[:1])},
    },
]


def test_every_check_runs_and_finds_the_paths_alike() -> None:
    report = invariance.run(toy(), CASES, model="toy")
    assert [c.name for c in report.checks] == [
        "chunks",
        "whole",
        "kept",
        "grown",
        "together",
    ]  # toy: questions by place
    assert all(c.worst < 1e-6 for c in report.checks), report.text()
    assert report.passed
    assert "passed" in report.text()


def test_a_path_that_changes_the_answers_fails(monkeypatch: Any) -> None:
    d = toy()
    real = d.answer_rendered

    def drifting(renderings: Any) -> Any:  # answers that depend on the chunk size
        results = real(renderings)
        if d.chunk != 4096:
            for r in results:
                for a in r.answers.values():
                    object.__setattr__(a, "probabilities", tuple(reversed(a.probabilities)))
        return results

    monkeypatch.setattr(d, "answer_rendered", drifting)
    report = invariance.run(d, CASES, model="toy")
    chunks = next(c for c in report.checks if c.name == "chunks")
    assert not chunks.passed(report.tolerance)
    assert not report.passed
