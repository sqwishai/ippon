"""A state that grows between requests (an agent appending to its conversation) is resumable from the kept
prefix: the previous prompt keeps a position that the next one still shares, for text and JSON states alike."""

from __future__ import annotations

from typing import Any

import pytest

from ippon.plan import plan
from ippon.schema import parse_request

MESSAGES = [
    "customer: Hi, I was charged twice for my order #48213.",
    "agent: Sorry about that. Can you confirm the last four digits of the card?",
    "customer: It's 4417. Both charges were $129.",
    "agent: Thanks, I've refunded the duplicate charge.",
]
QUESTIONS = {
    "resolved": {"type": "noul", "instructions": "Has the customer's problem been resolved?"},
    "next": {"type": "choice", "instructions": "What next?", "criteria": {"refund": "Refund", "close": "Close"}},
}


def formats() -> dict[str, Any]:
    from test_clef import clef_format
    from test_code_adapters import d3_format

    return {"d3": d3_format, "clef": clef_format}


# Not Clef with a JSON object: it sorts the object's keys, as its creator's code does, so a growing list that sorts
# before another key grows mid-prompt
@pytest.mark.parametrize(
    ("name", "shape"),
    [("d3", "text"), ("d3", "json list"), ("d3", "json object"), ("clef", "text"), ("clef", "json list")],
)
def test_each_step_can_resume_from_the_previous_one(name: str, shape: str) -> None:
    fmt = formats()[name]()
    previous = None
    for n in range(1, len(MESSAGES) + 1):
        state: Any = "\n".join(MESSAGES[:n])
        if shape == "json list":
            state = MESSAGES[:n]
        elif shape == "json object":
            state = {"ticket": 48213, "conversation": MESSAGES[:n]}
        p = plan(fmt.render(parse_request({"state": state, "questions": QUESTIONS})).prompts)
        if previous is not None:
            common = next(
                (i for i, (a, b) in enumerate(zip(previous.prefix, p.prefix, strict=False)) if a != b),
                min(len(previous.prefix), len(p.prefix)),
            )
            usable = [m for m in previous.marks if m <= common]
            assert usable, (n, previous.marks, common)
            assert max(usable) >= common - 8  # resumes close to where the states part
        previous = p
