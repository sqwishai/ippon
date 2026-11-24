"""The requests AutoTrust-Decider-Mini-9B's reference is made on: python3 make_autotrust.py > autotrust.json

Deterministic. The decider's head scores a noul, a score on six levels (0 to 5) and a choice among 2 to 16 options. So:
every standard request (make_standard.py), the ones it refuses included (a score that isn't six levels, a choice of more
than 16 options), then requests for what the standard set leaves out: scores on six levels, with and without level
names, a choice with descriptions and one of all 16 options, a JSON object, a list and a text with non-Latin script as
the state, every kind in one request, and the refusals at the edges (a choice of one option, of 17, a score of 7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STANDARD = Path(__file__).resolve().parent / "standard.json"
LEVELS = ["Not at all", "Slightly", "Somewhat", "Moderately", "Very", "Extremely"]


def requests() -> list[dict[str, Any]]:
    out = json.loads(STANDARD.read_text())
    ticket = (
        "Order #48213 was charged twice on my card: once on 3 October and again on 4 October for $129.00. "
        "I only placed one order. Please reverse the second charge today."
    )
    cases: list[tuple[str, Any, dict[str, Any]]] = [
        ("score-six-named", ticket, {"urgency": {"type": "score", "instructions": "How urgent is this?",
                                                 "criteria": LEVELS}}),
        ("score-six-digits", ticket, {"anger": {"type": "score", "instructions": "How angry is the customer?",
                                               "criteria": ["0", "1", "2", "3", "4", "5"]}}),
        ("choice-described", ticket, {"team": {"type": "choice", "instructions": "Which team should take this?",
                                              "criteria": {"billing": "Charges, refunds and invoices",
                                                           "shipping": "Parcels and delivery",
                                                           "account": None}}}),
        ("choice-sixteen", "The user wants to move a meeting to next Tuesday afternoon.",
         {"tool": {"type": "choice", "instructions": "Which tool fits the next step?",
                   "criteria": [f"tool_{k:02d}" for k in range(15)] + ["calendar.reschedule"]}}),
        ("json-state", {"ticket": 912, "customer": "Ana", "message": "Is this my invoice?", "paid": True},
         {"invoice": {"type": "noul", "instructions": "Does the customer ask about an invoice?"}}),
        ("list-state", ["Customer: the parcel arrived damaged.", {"order": 77, "items": 2}],
         {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}),
        ("non-latin", "客户说包裹损坏了，想要退款。Клиент просит вернуть деньги.",  # noqa: RUF001
         {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}),
        ("every-kind", ticket, {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {"type": "choice", "instructions": "Which team should take this?",
                     "criteria": ["billing", "shipping", "account", "sales"]},
            "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": LEVELS},
        }),
        ("choice-one", ticket, {"team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing"]}}),
        ("choice-seventeen", ticket, {"team": {"type": "choice", "instructions": "Which team?",
                                              "criteria": [f"team_{k}" for k in range(17)]}}),
        ("score-seven", ticket, {"urgency": {"type": "score", "instructions": "How urgent is this?",
                                             "criteria": [*LEVELS, "Critical"]}}),
    ]  # fmt: skip
    out += [{"name": name, "request": {"state": state, "questions": questions}} for name, state, questions in cases]
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
