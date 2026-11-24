"""The standard request set every adapter's reference is made on: python3 make_standard.py > standard.json

Deterministic. It covers each question type and the format edge cases most decision models have: yes/no with and
without criteria, choices of 2, 10, 11, 26 and 40 options, scores of 2, 5 and 10 levels, JSON states (with a long
array), non-ASCII text, several questions about one state, 40 at once, and states of about 300, 1,000, 3,000 and
6,000 tokens (past some models' windows).
"""

from __future__ import annotations

import json
import random

TICKET = (
    "Order #48213 was charged twice on my card ending 4417: once on 3 October for $129.00 and again on 4 October "
    "for the same amount. I only placed one order. Your chat assistant told me a refund would arrive in five days, "
    "but nothing has come back after two weeks. I have been a customer since 2019 and this is the first problem I "
    "have had. Please reverse the second charge and confirm by email. If this is not fixed by Friday I will dispute "
    "it with my bank."
)
WORDS = [
    "account",
    "order",
    "refund",
    "shipping",
    "invoice",
    "delay",
    "password",
    "warranty",
    "discount",
    "address",
    "payment",
    "card",
    "email",
    "return",
    "delivery",
    "support",
    "manager",
    "receipt",
    "subscription",
    "cancel",
    "upgrade",
]


def long_text(rng: random.Random, sentences: int) -> str:
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16))).capitalize() + "." for _ in range(sentences)
    )


def requests() -> list[dict]:
    rng = random.Random(7)
    out = [
        {
            "name": "noul",
            "request": {
                "state": TICKET,
                "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
            },
        },
        {
            "name": "noul-criteria",
            "request": {
                "state": TICKET,
                "questions": {
                    "urgent": {
                        "type": "noul",
                        "instructions": "Is it urgent?",
                        "criteria": {"true": "A deadline or threat is stated", "false": "It can wait"},
                    }
                },
            },
        },
        {
            "name": "noul-false-first",
            "request": {
                "state": TICKET,
                "questions": {
                    "urgent": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"false": "It can wait"}}
                },
            },
        },
        {
            "name": "noul-without-instructions",
            "request": {
                "state": TICKET,
                "questions": {"bank": {"type": "noul", "criteria": {"true": "The customer threatens a bank dispute"}}},
            },
        },
        {
            "name": "choice-2",
            "request": {
                "state": TICKET,
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Charges, refunds and invoices", "sales": None},
                    }
                },
            },
        },
        {
            "name": "choice-list",
            "request": {
                "state": TICKET,
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": ["billing", "sales", "support"],
                    }
                },
            },
        },
        {
            "name": "score-5",
            "request": {
                "state": TICKET,
                "questions": {
                    "mood": {
                        "type": "score",
                        "instructions": "How upset is the customer?",
                        "criteria": ["calm", "mildly annoyed", "annoyed", "angry", "furious"],
                    }
                },
            },
        },
        {
            "name": "score-2",
            "request": {
                "state": TICKET,
                "questions": {
                    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["minor", "major"]}
                },
            },
        },
        {
            "name": "score-10",
            "request": {
                "state": TICKET,
                "questions": {
                    "priority": {
                        "type": "score",
                        "instructions": "Priority from 0 to 9?",
                        "criteria": [f"priority {i}" for i in range(10)],
                    }
                },
            },
        },
        {
            "name": "unicode",
            "request": {
                "state": "Mi pedido llegó dañado — ¿pueden enviarme otro? 我需要退款。 Ça ne marche pas…",
                "questions": {
                    "lang": {
                        "type": "choice",
                        "instructions": "Which language comes first?",
                        "criteria": {"Spanish": None, "Chinese": None, "French": None, "English": None},
                    }
                },
            },
        },
        {
            "name": "json-state",
            "request": {
                "state": {
                    "ticket": {
                        "id": 48213,
                        "messages": [
                            {"from": "customer", "text": TICKET},
                            {"from": "agent", "text": "We are looking into it."},
                        ],
                    },
                    "history": [
                        {"order": 40000 + i, "status": rng.choice(["delivered", "returned", "lost"])} for i in range(12)
                    ],
                },
                "questions": {"lost": {"type": "noul", "instructions": "Has any earlier order been lost?"}},
            },
        },
        {
            "name": "several-questions",
            "request": {
                "state": TICKET,
                "questions": {
                    "refund": {"type": "noul", "instructions": "Is a refund asked for?"},
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {
                            "billing": "Charges, refunds and invoices",
                            "sales": "New purchases and upgrades",
                            "support": "Using the product",
                        },
                    },
                    "mood": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
                    "bank": {"type": "noul", "instructions": "Does the customer mention their bank?"},
                },
            },
        },
        {
            "name": "40-questions",  # more than one group of 32 in engines that read questions in groups
            "request": {
                "state": TICKET,
                "questions": {
                    f"q{i:02d}": {"type": "noul", "instructions": f"Does the ticket mention the number {i}?"}
                    for i in range(40)
                },
            },
        },
    ]
    for n in (10, 11, 26, 40):
        out.append(
            {
                "name": f"choice-{n}",
                "request": {
                    "state": TICKET,
                    "questions": {
                        "topic": {
                            "type": "choice",
                            "instructions": "Which topic fits best?",
                            "criteria": {f"topic_{i}": f"About {WORDS[i % len(WORDS)]} number {i}" for i in range(n)},
                        }
                    },
                },
            }
        )
    for tokens, sentences in ((300, 18), (1000, 60), (3000, 180), (6000, 360)):
        out.append(
            {
                "name": f"state-{tokens}",
                "request": {
                    "state": long_text(rng, sentences),
                    "questions": {
                        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
                        "topic": {
                            "type": "choice",
                            "instructions": "Main topic?",
                            "criteria": {
                                "billing": "Charges and refunds",
                                "shipping": "Delivery and returns",
                                "account": "Login and profile",
                                "other": "Anything else",
                            },
                        },
                        "tone": {
                            "type": "score",
                            "instructions": "How negative is the text?",
                            "criteria": ["neutral", "somewhat negative", "very negative"],
                        },
                    },
                },
            }
        )
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), ensure_ascii=False, indent=1))
