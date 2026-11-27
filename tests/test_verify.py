from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ippon.verify import Report, RequestReport, compare, load_reference, verify_model
from test_generic_adapters import ids, qwen3_classifier


def test_same_checkpoint_reference_rejects_a_changed_ollama_tag_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ippon import Decider, UnsupportedModel

    closed = []

    @contextmanager
    def loaded(*args: Any, **kwargs: Any) -> Iterator[Any]:
        try:
            yield SimpleNamespace(details={"checkpoint": "current-manifest"})
        finally:
            closed.append(True)

    monkeypatch.setattr(Decider, "load", loaded)
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ippon-reference/1",
                "reference_kind": "checkpoint_dequantized",
                "model": {"revision": "previous-manifest"},
                "requests": [{"name": "a"}],
            }
        )
    )
    with pytest.raises(UnsupportedModel, match="different Ollama checkpoint"):
        verify_model("ollama://toy", path)
    assert closed == [True]


@pytest.mark.parametrize(
    "fields",
    [
        {"requests": []},
        {"complete": False},
        {"complete": True, "planned_requests": 2, "completed_requests": 1},
        {"complete": True, "planned_requests": 1, "completed_requests": 0},
    ],
)
def test_reference_rejects_empty_or_interrupted_evidence(tmp_path: Path, fields: dict[str, Any]) -> None:
    path = tmp_path / "reference.json"
    reference = {"schema": "ippon-reference/1", "requests": [{"name": "finished"}], **fields}
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match=r"no requests|incomplete"):
        load_reference(path)


def test_reference_accepts_legacy_and_completed_evidence(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    reference = {"schema": "ippon-reference/1", "requests": [{"name": "finished"}]}
    for markers in ({}, {"complete": True, "planned_requests": 1, "completed_requests": 1}):
        expected = reference | markers
        path.write_text(json.dumps(expected))
        assert load_reference(path) == expected


def test_compare_finds_the_largest_difference_a_changed_top_answer_and_threshold_crossings() -> None:
    """A changed top answer fails however close, unless the reference's top two are within `tie` (a near tie);
    threshold crossings are reported even when the top answer agrees."""
    c = compare([[0.6, 0.4], [0.1, 0.2, 0.7]], [[0.62, 0.38], [0.1, 0.25, 0.65]])
    assert c.max_abs_difference == pytest.approx(0.05)
    assert c.top_agrees
    assert not compare([[0.51, 0.49]], [[0.49, 0.51]]).top_agrees
    near = compare([[0.51, 0.49]], [[0.49, 0.51]], tie=0.03)  # the reference's top two 0.02 apart: a near tie
    assert (near.top_agrees, near.near_ties) == (True, 1)
    assert not compare([[0.6, 0.4]], [[0.45, 0.55]], tie=0.03).top_agrees  # 0.1 apart: a real disagreement
    crossed = compare([[0.701, 0.299]], [[0.699, 0.301]], thresholds=(0.5, 0.7, 0.9))
    assert crossed.top_agrees
    assert dict(crossed.threshold_crossings) == {0.5: 0, 0.7: 1, 0.9: 0}


def test_compare_refuses_answers_it_cannot_compare() -> None:
    """Other shapes, nothing, or a probability that isn't finite on either side: a NaN after a finite difference would
    be ignored by max(), falsely passing the bound."""
    with pytest.raises(ValueError, match="shape"):
        compare([[0.5, 0.5]], [[0.5, 0.25, 0.25]])
    with pytest.raises(ValueError, match="nothing"):
        compare([], [])
    with pytest.raises(ValueError, match="finite probabilities"):
        compare([[0.5, float("nan")]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="finite probabilities"):
        compare([[0.5, 0.5]], [[0.5, float("nan")]])


def test_the_report_needs_every_request_within_bound_and_top() -> None:
    def report(bound: float, **comparisons: Any) -> Report:
        return Report("m", bound, tuple(RequestReport(n, True, c, True) for n, c in comparisons.items()))

    good, close, flipped = (
        compare([[0.5, 0.5]], [[0.505, 0.495]]),
        compare([[0.7, 0.3]], [[0.75, 0.25]]),
        compare([[0.51, 0.49]], [[0.49, 0.51]]),
    )
    assert report(0.01, a=good).passed
    verdict = report(0.03, a=good, b=close, c=flipped)
    assert not verdict.passed
    assert verdict.worst == pytest.approx(0.05)
    assert [f.split(":")[0] for f in verdict.failures] == ["b", "c"]
    assert "different top answer" in verdict.failures[1]
    # weights quantized as they loaded (q8) are graded as `ippon setup` tags them: past full precision's bound, but
    # under 10 points with every top answer the same, they pass, listing what is past; a changed top answer fails
    held = Report("m", 0.03, verdict.requests[:2], quantized="q8")
    assert held.passed
    assert held.level == "close"
    assert "  b: 0.05000 from the reference (full precision's bound 0.03)" in held.text().splitlines()
    assert not Report("m", 0.03, verdict.requests, quantized="q8").passed


def test_a_tokens_only_reference_checks_prompts_counts_and_refusals_on_a_checkpoint(tmp_path: Path) -> None:
    """Each request's prompt and its count against the creator's, without running the model; a request the creator's
    code refuses matches only when ippon refuses it too. A media reference's text-only request reads as it is."""
    qwen3_classifier(tmp_path)
    choice = {"type": "choice", "criteria": ["support", "billing"]}
    unlabelled = {"type": "noul", "instructions": "Is it about money?"}  # names none of the model's labels
    cases: list[dict[str, Any]] = [
        {"name": "same", "request": {"state": "w1 w2 w3", "questions": {"c": choice}}, "prompts": [ids("w1 w2 w3")]},
        {"name": "other", "request": {"state": "w1 w2", "questions": {"c": choice}}, "prompts": [ids("w1 w2 w3")]},
        {"name": "counted", "request": {"state": "w4", "questions": {"c": choice}}, "prompts": [ids("w4")]},
        {"name": "refused", "request": {"state": "w9", "questions": {"q": unlabelled}}, "error": "InvalidRequest"},
        {"name": "answered", "request": {"state": "w9", "questions": {"c": choice}}, "error": "InvalidRequest"},
    ]
    cases[0]["input_tokens"], cases[2]["input_tokens"] = 5, 4  # w4's prompt is 3 tokens
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"schema": "ippon-reference/1", "media_format": {"image_token": 63}, "requests": cases})
    )
    report = verify_model(str(tmp_path), reference, precision="fp32", keep_awake=None)
    found = {r.name: (r.tokens_match, r.comparison, r.input_tokens_match) for r in report.requests}
    assert found == {
        "same": (True, None, True),
        "other": (False, None, True),
        "counted": (True, None, False),
        "refused": (True, None, True),
        "answered": (False, None, False),
    }
    assert [f.split(":")[0] for f in report.failures] == ["other", "counted", "answered", "answered"]
    assert report.text().startswith(f"{tmp_path}: 5 requests, prompt tokens and counts only")
