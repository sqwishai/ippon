"""`ippon eval`: labelled rows answered by each model in turn (one resident at a time) through the batched path,
a row's refusal its own; accuracy, macro-F1, confusion, thresholds and disagreements from the models' own,
unchanged probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from ippon import Decider, Loaded
from ippon.adapter import Hidden, Rendering
from ippon.errors import InsufficientMemory, InvalidRequest
from ippon.evaluate import Answered, choose, copyable, flagging, held_out, load, report, run, text
from ippon.schema import Question
from toy import ToyBackbone, ToyFormat

QUESTIONS: dict[str, Any] = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?", "criteria": ["billing", "other"]},
    "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "angry"]},
}


class Flipped(ToyFormat):  # another model: every question's probabilities in reverse
    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        return super().answer(question, list(reversed(scores)))


class Crashing(ToyFormat):  # a model that fails, not on purpose, on one row's state
    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        if "Ticket 3" in str(rendering.request.state):
            raise RuntimeError("a bug")
        return super().scores(rendering, hidden)


resident: list[Decider] = []


class Tracked(Decider):
    def close(self) -> None:
        resident.remove(self)
        super().close()


def loader(fmt: type[ToyFormat]) -> Any:
    def load() -> Decider:
        assert not resident  # the model before was closed first
        backbone = ToyBackbone()
        decider = Tracked(Loaded(backbone, fmt(backbone)), prefix_budget=0)
        resident.append(decider)
        return decider

    return load


def test_eval_answers_every_row_per_model_in_turn_and_reports_from_their_own_probabilities(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    labels = [
        {"refund": "yes", "team": "billing", "mood": "angry"},
        {"refund": False, "team": "other", "mood": 0},
        {"refund": "true", "team": "billing", "mood": "1"},
        {"refund": "no", "team": "other"},
        {"refund": 1, "team": "billing", "mood": "calm"},
    ]
    lines = [{"id": f"t{i}", "state": f"Ticket {i}: charged {i} times.", "labels": x} for i, x in enumerate(labels)]
    lines.insert(2, {"state": "A photo.", "images": ["R0lGODlhAQABAAAAACw="], "labels": {"refund": "no"}})
    data.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    rows = load(data, QUESTIONS)
    assert [r.labels.get("mood") for r in rows] == ["1", "0", None, "1", None, "0"]

    def refused() -> Decider:
        raise InsufficientMemory("the model needs about 17.5 GB, and 14.0 GB is free now")

    results, about = run([("a", loader(ToyFormat)), ("x", refused), ("b", loader(Flipped))], rows, batch=4)
    assert set(results) == {"a", "b"}  # a model that can't load is reported, and the others still answered
    assert about["x"]["error"]["type"] == "insufficient_memory"
    assert not resident
    # a pass that fails is answered again row by row: the row that fails gets its error, the other its answer
    crashed = run([("c", loader(Crashing))], rows, batch=4)[0]["c"]
    assert [x.error.kind if x.error else None for x in crashed[4:]] == ["internal", None]
    a = results["a"]
    assert "images" in str(a[2].error)
    alone = loader(ToyFormat)()
    for row, answered in zip(rows, a, strict=True):  # the probabilities are the model's own
        if answered.error is None:
            own = alone.decide(row.body["state"], row.body["questions"])
            assert answered.probabilities == {q: x.by_option for q, x in own.answers.items()}
    alone.close()

    found = report(results, about)
    assert set(found["about"]) == {"a", "x", "b"}
    assert set(found["models"]["a"]) == {"refund", "team", "mood"}
    shown = text(found)
    assert "x: not loaded: the model needs about 17.5 GB" in shown
    assert "Where the models' answers differ" in shown
    assert "flagged when P(true) is at least t" in shown


def test_the_report_scores_each_question_from_the_top_answers(tmp_path: Path) -> None:
    """Five rows, worked by hand: model a refuses one, says yes to a no (line 2) and is otherwise right on refund; on
    team it is right twice and wrong twice; model b differs from a on line 4's refund only."""
    data = tmp_path / "data.jsonl"
    labels = [(True, "billing"), (False, "other"), (True, "billing"), (False, "other"), (True, "billing")]
    data.write_text("".join(json.dumps({"state": "x", "labels": {"refund": r, "team": t}}) + "\n" for r, t in labels))
    rows = load(data, QUESTIONS)

    def said(row: int, p_yes: float, billing: float) -> Answered:
        return Answered(rows[row], {"refund": {"false": 1 - p_yes, "true": p_yes},
                                    "team": {"billing": billing, "other": 1 - billing}})  # fmt: skip

    a = [said(0, 0.9, 0.8), said(1, 0.7, 0.6), Answered(rows[2], error=InvalidRequest("x")), said(3, 0.2, 0.3)]
    a.append(said(4, 0.6, 0.45))
    b = [*a[:3], said(3, 0.6, 0.3), a[4]]
    found = report({"a": a, "b": b})
    refund, team = found["models"]["a"]["refund"], found["models"]["a"]["team"]
    assert (refund["rows"], refund["answered"], refund["refused"]) == (5, 4, 1)
    assert refund["accuracy"] == 0.75
    assert refund["macro_f1"] == pytest.approx((4 / 5 + 2 / 3) / 2)  # true: 2 right, 1 false yes; false: 1 of 2
    assert refund["confusion"] == {"false": {"false": 1, "true": 1}, "true": {"false": 0, "true": 2}}
    kept = [(t["threshold"], t["kept"], t["accuracy"], t["false_yes"], t["false_no"]) for t in refund["thresholds"]]
    assert kept == [
        (0.0, 4, 0.75, 1, 0), (0.5, 4, 0.75, 1, 0), (0.6, 4, 0.75, 1, 0), (0.7, 3, pytest.approx(2 / 3), 1, 0),
        (0.8, 2, 1.0, 0, 0), (0.9, 1, 1.0, 0, 0), (0.95, 0, None, 0, 0), (0.99, 0, None, 0, 0),
    ]  # fmt: skip
    assert "flagging" in refund  # a yes/no question's alone
    assert (team["accuracy"], team["macro_f1"]) == (0.5, 0.5)
    assert "false_yes" not in team["thresholds"][0]
    assert "flagging" not in team
    # where a and b differ, and only there: not the row a refused
    assert [d["line"] for d in found["disagreements"]["refund"]] == [4]
    assert "team" not in found["disagreements"]


def test_eval_stops_on_a_row_it_cannot_score_naming_its_line(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    with pytest.raises(InvalidRequest, match=r"can't read.*data\.jsonl"):
        load(data, QUESTIONS)
    data.write_bytes(b"\xff\xfe")
    with pytest.raises(InvalidRequest, match="UTF-8"):
        load(data, QUESTIONS)
    data.write_text(json.dumps({"state": "x", "labels": {"team": "sales"}}) + "\n")
    with pytest.raises(InvalidRequest, match=r"line 1: 'sales' is not an answer to team \(billing, other\)"):
        load(data, QUESTIONS)
    data.write_text(json.dumps({"state": "x", "labels": {"urgent": True}}) + "\n")
    with pytest.raises(InvalidRequest, match="line 1: a label for 'urgent', which no question asks"):
        load(data, QUESTIONS)


def test_a_yes_no_judge_is_flagged_at_its_own_threshold_checked_on_held_out_rows() -> None:
    """Husain and Saha's example: every failure at 0.40 and every pass at 0.10 catches nothing at 0.5 and everything at
    the threshold chosen, on held-out rows too, and the calibration table shows where P(true) sits. A judge whose
    scores overlap meets no floors of 0.8 on all its rows; each half of them alone does, and the threshold each half
    chooses catches half as well on the other: what choosing on the rows it is judged on hides."""
    scores = [(True, 0.4)] * 6 + [(False, 0.1)] * 4
    f = flagging(scores, 0.8, 0.8)
    assert f["at_half"] == {"flagged": 0, "precision": None, "recall": 0.0, "f1": 0.0}
    assert f["best"] == {"threshold": 0.4, "use": 0.4, "flagged": 6, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    # the threshold to copy flags exactly the chosen one's rows: 0.7985 rounded to 0.799 would leave its row out
    assert copyable([(True, 0.7985218167304993), (False, 0.61)], 0.7985218167304993) == 0.7
    held = f["held_out"]
    assert (held["folds"], held["without_threshold"], held["thresholds"]) == (4, 0, [0.4] * 4)
    assert (held["precision"], held["recall"], held["f1"]) == (1.0, 1.0, 1.0)
    assert f["brier"] == pytest.approx((6 * 0.6**2 + 4 * 0.1**2) / 10)
    assert [(c["rows"], c["observed"]) for c in f["calibration"]] == [(4, 0.0), (6, 1.0)]
    assert [c["predicted"] for c in f["calibration"]] == pytest.approx([0.1, 0.4])

    overlap = [(True, 0.9), (True, 0.3), (False, 0.8), (False, 0.2)]
    assert choose(overlap) == 0.3  # F1 0.8, the highest
    assert choose(overlap, 0.8, 0.8) is None
    assert flagging(overlap, 0.8, 0.8)["best"] is None
    halves = held_out(overlap, 0.8, 0.8)
    assert halves is not None
    assert (halves["without_threshold"], halves["thresholds"]) == (0, [0.3, 0.9])
    assert (halves["precision"], halves["recall"]) == (0.5, 0.5)
    # two traces of a true and a false step each: a trace's steps stay in one fold, so the threshold judged on
    # trace y is chosen on trace x alone, and trace y alone offers none
    traces = held_out(overlap, 0.8, 0.8, groups=["x", "y", "y", "x"])
    assert traces is not None
    assert (traces["without_threshold"], traces["thresholds"], traces["recall"]) == (1, [0.9], 0.0)


def test_eval_refuses_only_the_row_with_unsupported_media(tmp_path: Path) -> None:
    """One HEIC row in a JSONL eval file is recorded as refused rather than aborting the whole file."""
    import base64

    heic = base64.b64encode(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic").decode()
    data = tmp_path / "mixed.jsonl"
    lines = [
        {"id": "ok1", "state": "Ticket 1", "labels": {"refund": True}},
        {"id": "bad_heic", "state": "Photo.", "images": [heic], "labels": {"refund": False}},
        {"id": "ok2", "state": "Ticket 2", "labels": {"refund": False}},
    ]
    data.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows = load(data, QUESTIONS)
    assert len(rows) == 3
    assert rows[1].error is not None
    assert "HEIC" in str(rows[1].error)
    results, _ = run([("a", loader(ToyFormat))], rows, batch=4)
    answered = results["a"]
    assert [a.row.id for a in answered] == ["ok1", "bad_heic", "ok2"]
    assert answered[0].error is None
    assert answered[2].error is None
    assert answered[1].error is not None
    assert "HEIC" in str(answered[1].error)
