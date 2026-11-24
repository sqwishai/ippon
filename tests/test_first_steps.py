"""A new user's first commands, on the toy model: `pull` (the newest revision, then a sample decision), `template`,
`lint`, `decide --states` and `eval`, with their exit codes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ippon import decider, hub
from ippon.cli import main
from ippon.templates import SAMPLE_TEXT
from starters import starters
from test_pull import remote as remote
from test_serve import toy

GIF = "R0lGODlhAQABAAAAACw="  # an image, which the toy model doesn't read


@pytest.fixture
def toys(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every model loads as the toy; the names loaded, in order."""
    loaded: list[str] = []

    def load(model: str, **options: Any) -> decider.Decider:
        loaded.append(model)
        return toy()

    monkeypatch.setattr(decider.Decider, "load", load)
    return loaded


def test_pull_reads_the_newest_revision_then_answers_a_sample(
    remote: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], toys: list[str]
) -> None:
    remote.license = "apache-2.0"
    old = tmp_path / "hub" / "models--org--model" / "snapshots" / ("a" * 40)
    old.mkdir(parents=True)
    (old / "config.json").write_text("{}")
    hub.record_download("org/model", ("*.json",), old)  # an earlier pull read revision a
    remote.sha = "b" * 40
    assert main(["pull", "org/model"]) == 0
    out = capsys.readouterr().out
    assert f"revision {'b' * 12}: the newest, which loads read from now on (it was {'a' * 12})" in out
    assert toys == ["org/model"]
    expected = toy().decide(**SAMPLE_TEXT)
    for qid, answer in expected.answers.items():
        top = max(answer.by_option.items(), key=lambda kv: kv[1])
        shown = {"true": "yes", "false": "no"}.get(top[0], top[0])
        assert f"  {qid} ({answer.question.instructions}): {shown} {top[1]:.2f}" in out
    assert "tune" not in out
    assert "`ippon verify` checks the chosen build" in out
    assert main(["pull", "org/model", "--no-sample", "--json"]) == 0  # the files cached now, read from disk
    assert json.loads(capsys.readouterr().out)["revision"] == "b" * 40
    assert len(toys) == 1


def test_a_starter_request_linted_decided_on_states_and_evaluated_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], toys: list[str]
) -> None:
    moderation = starters()["moderation"]  # docs/evaluating.md's, as a user copies it
    request = tmp_path / "request.json"
    request.write_text(json.dumps(moderation))
    assert main(["lint", str(request)]) == 0
    assert capsys.readouterr().out.strip() == "nothing to change"
    assert main(["lint", str(request), "--model", "d1-2b"]) == 0  # rendered as the model reads it
    assert "prompt for rule: " in capsys.readouterr().out
    vague = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Refund?"}}}
    assert main(["lint", json.dumps(vague)]) == 1
    assert "q: 'Refund?' is too short" in capsys.readouterr().out

    states = tmp_path / "states.jsonl"
    lines = ['"Charged twice."', '{"id": "t2", "state": "Where is my parcel?"}', "{oops", "",
             json.dumps({"state": "A photo.", "images": [GIF]}), '{"text": "x"}', '{"id": "t7"}',
             json.dumps({"state": "A clip.", "videos": [GIF]})]  # fmt: skip
    states.write_text("\n".join(lines))
    review = {**moderation, "unsure_below": 1.0}  # every answer unsure: each line keeps the request's options
    assert main(["decide", "d1-2b", json.dumps(review), "--states", str(states)]) == 1
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [o["line"] for o in out] == [1, 2, 3, 5, 6, 7, 8]
    model = toy()
    for o, state in ((out[0], "Charged twice."), (out[1], "Where is my parcel?")):
        assert {k: v for k, v in o.items() if k not in ("line", "id")} == model.systemone({**review, "state": state})
        assert o["unsure"]
    assert out[1]["id"] == "t2"
    assert [o["error"]["type"] for o in out[2:]] == ["invalid_request"] * 5
    assert "nothing to ask about" in out[5]["error"]["message"]  # an id alone
    assert "images" in out[3]["error"]["message"]
    assert "unknown field" not in out[6]["error"]["message"]  # a line may give videos as a request may

    data = tmp_path / "labelled.jsonl"
    data.write_text("\n".join(json.dumps({"state": s, "labels": {"remove": r}}) for s, r in [("a", True), ("b", "no")]))
    assert main(["eval", str(data), "--questions", str(request), "--model", "d1-2b", "--model", "d3-mini"]) == 0
    report = capsys.readouterr().out
    assert "accuracy" in report
    assert "remove: expected (rows) by answered (columns)" in report
    assert toys[-2:] == [
        "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4",
        "vllm-sr/d3-mini@97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e",
    ]  # one after the other
    results = [json.loads(line) for line in (tmp_path / "labelled.results.jsonl").read_text().splitlines()]
    assert [(r["model"], r["line"]) for r in results] == [("d1-2b", 1), ("d1-2b", 2), ("d3-mini", 1), ("d3-mini", 2)]
    assert main(["eval", str(data), "--model", "d1-2b"]) == 3  # no questions for the rows: refused before any load
