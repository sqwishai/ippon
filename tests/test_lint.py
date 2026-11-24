"""`ippon lint` and `template`: every template is a request a model can be asked and lint finds nothing in it;
lint finds each kind of problem in a request, and with a model what it can't be asked and what would be cut."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ippon import Decider, Loaded
from ippon.adapter import Cut, Rendering
from ippon.errors import InvalidRequest
from ippon.lint import against, lint
from ippon.schema import Request, parse_request
from ippon.templates import SAMPLE_TEXT
from starters import starters
from toy import ToyBackbone, ToyFormat

BAD: dict[str, Any] = {
    "state": " ",
    "questions": {
        "bare": {"type": "noul"},
        "short": {"type": "noul", "instructions": "Refund?"},
        "two": {"type": "noul", "instructions": "Is it a refund? Is it urgent?"},
        "open": {"type": "noul", "instructions": "Which team should handle it?"},
        "overlap": {
            "type": "choice",
            "instructions": "What does the customer want from us?",
            "criteria": {"refund": "Money back.", "refund request": None, "Exchange": None, "exchange ": None},
        },
        "one": {"type": "choice", "instructions": "Which plan is the customer on?", "criteria": ["pro"]},
        "again": {"type": "choice", "instructions": "Which team should handle it?", "criteria": ["a", "b", "c"]},
        "long": {
            "type": "choice",
            "instructions": "Which part of the codebase does this commit change?",
            "criteria": {
                part: "modules: " + ", ".join(f"{part} module {j}" for j in range(90))
                for part in ("core", "terminal", "login")
            }
            | {"other": "anything the rest don't cover"},
        },
        "levels": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["calm", "Calm", "angry"],
        },
    },
}


def test_the_starter_requests_lint_clean_and_lint_names_each_problem_and_what_to_change() -> None:
    for name, body in [*starters().items(), ("sample", SAMPLE_TEXT)]:
        parse_request(body)
        assert lint(body) == [], name
    found = [(f.question, f.message) for f in lint(BAD)]
    expected = [
        ("", "the state is empty"),
        ("bare", "no instructions"),
        ("short", "too short"),
        ("two", "asks 2 things"),
        ("open", "a noul is answered yes or no"),
        ("overlap", "'refund' and 'refund request' may overlap"),
        ("overlap", "'Exchange' and 'exchange ' are one written twice"),
        ("overlap", "no catch-all among its 4 options"),
        ("one", "a choice of one option"),
        ("again", "no catch-all among its 3 options"),
        ("again", "asks what open asks"),
        ("long", "its 4 options run to"),
        ("levels", "'calm' and 'Calm' are one written twice"),
    ]
    assert len(found) == len(expected)
    for (question, message), (wanted, says) in zip(found, expected, strict=True):
        assert question == wanted
        assert says in message
    assert "not a System One request" in lint({"questions": {}})[0].message


def test_with_a_model_lint_says_what_it_cannot_be_asked_and_what_would_be_cut() -> None:
    class Cutting(ToyFormat):  # a window that cuts any state, and a readout with codes for three options
        def render(self, request: Request) -> Rendering:
            if any(len(q.options) > 3 for q in request.questions):
                raise InvalidRequest("no single-token code left for 4 options")
            return replace(super().render(request), cuts=(Cut("", "state", 120, 80),))

    backbone = ToyBackbone()
    decider = Decider(Loaded(backbone, Cutting(backbone)), name="toy")
    found, seen = against(decider, SAMPLE_TEXT)
    assert [str(f) for f in found] == [
        "the state would be cut to fit toy's window: 120 of its 200 tokens left out, unread; shorten it, or send only "
        "what the questions need"
    ]
    assert [p["tokens"] for p in seen["prompts"]] == [len(SAMPLE_TEXT["state"].encode()) + 2] * 2
    found, seen = against(decider, starters()["ticket-routing"])
    assert [str(f) for f in found] == ["toy can't be asked this: no single-token code left for 4 options"]
    assert seen == {}
