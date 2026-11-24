"""Input past a model's window: the formats that cut it as their creators' code does (Clef) say what they cut
(`Rendering.cuts`), the response lists it (`truncated`), truncation="error" refuses such a request, and
`Decider.inspect` shows what the model reads."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ippon import Decider, Loaded
from ippon.adapter import Cut, Rendering
from ippon.errors import ContextTooLong
from ippon.schema import Request, parse_request
from test_clef import clef_format
from test_decider import REQUEST
from toy import ToyBackbone, ToyFormat

WORDS = " ".join(f"word{i}" for i in range(3000))


def opposite_endings() -> tuple[dict[str, Any], dict[str, Any]]:
    """Two requests whose states share their first 3,000 words and end in opposite facts."""
    ask = {"refund": {"type": "noul", "instructions": "Is a refund asked for?"}}
    return (
        {"state": WORDS + " The customer wants a refund.", "questions": ask},
        {"state": WORDS + " The customer does not want a refund.", "questions": ask},
    )


def test_clef_says_it_cut_the_state_its_questions_share() -> None:
    fmt = replace(clef_format(), window=600)
    request = parse_request(opposite_endings()[0])
    full = replace(fmt, window=16384).render(request).prompts[0].tokens
    assert isinstance(request.state, str)
    state = tuple(fmt.tokenizer.encode(request.state))
    start = next(i for i in range(len(full)) if full[i : i + len(state)] == state)
    kept = 600 - (len(full) - len(state))
    rendering = fmt.render(request)
    (cut,) = rendering.cuts
    assert (cut.question, cut.part, cut.kept, cut.dropped) == ("", "state", kept, len(state) - kept)
    assert rendering.prompts[0].tokens == full[:start] + state[:kept] + full[start + len(state) :]


class Cutting(ToyFormat):
    """The toy format, saying it cut the state's end."""

    def render(self, request: Request) -> Rendering:
        return replace(super().render(request), cuts=(Cut("", "state", 7, 3),))


def cutting(truncation: Any = "creator") -> Decider:
    backbone = ToyBackbone()
    return Decider(Loaded(backbone, Cutting(backbone)), prefix_budget=0, truncation=truncation)


def test_the_response_and_inspect_list_what_was_cut_and_error_refuses_it() -> None:
    cut = [{"question": "", "part": "state", "tokens_dropped": 7, "tokens_kept": 3}]
    assert cutting().systemone(REQUEST)["truncated"] == cut
    decision = {"id": "refund", "question": "Is a refund asked for?", "kind": "binary"}
    assert cutting().sqwish({"context": REQUEST["state"], "decisions": [decision]})["truncated"] == cut  # decide API
    found = cutting().inspect(REQUEST)  # what the model reads: each prompt, and what was cut
    assert found["truncated"] == cut
    assert sorted(q for p in found["prompts"] for q in p["questions"]) == sorted(REQUEST["questions"])
    assert all(p["tokens"] == len(p["ids"]) for p in found["prompts"])
    with pytest.raises(ContextTooLong, match="state, 7 of 10 tokens"):
        cutting("error").systemone(REQUEST)
    backbone = ToyBackbone()
    plain = Decider(Loaded(backbone, ToyFormat(backbone)), prefix_budget=0, truncation="error")
    assert "truncated" not in plain.systemone(REQUEST)  # nothing cut: nothing listed, nothing refused
