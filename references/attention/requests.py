"""Deterministic long text requests for attention verification, without running a model.

The state lengths are tokenizer tokens, not complete prompt lengths; every creator reference records its actual
prompt lengths. The middle case leaves room within Nimble's 8,192-token contract (a model ippon no longer supports).
The last is an over-limit case for that model, not permission to truncate it. Use the same JSON for creator and ippon
runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the current customer explicitly request a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should investigate the current customer's main unresolved issue?",
        "criteria": {
            "billing": "Charges, invoices and reimbursements",
            "shipping": "Shipment tracking and delivery",
            "account": "Sign-in and account access",
            "technical": "Product failures and troubleshooting",
            "sales": "Pre-purchase questions",
            "returns": "Returning a received item",
            "fraud": "Unauthorized transactions",
            "subscriptions": "Recurring plans and cancellation",
            "warranty": "Coverage and repairs",
            "feedback": "General suggestions and compliments",
            "other": "An issue outside the other teams' responsibilities",
        },
    },
    "mood": {
        "type": "score",
        "instructions": "How upset is the current customer, based only on their own wording?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    records = []
    details = (
        "A delivery address was corrected and the parcel subsequently arrived.",
        "An invoice was explained; the customer accepted the price and kept the service.",
        "An old device was replaced after a hardware inspection.",
        "A password reset restored access; there was no outstanding payment problem.",
        "A subscription was renewed after its owner confirmed the plan.",
        "A duplicate charge was reversed after an unrelated customer's complaint.",
    )
    for number in range(2400):
        records.append(f"Archive entry {number:04d}. {details[(number * 7 + number // 13) % len(details)]}")
    archive = tokenizer.encode("\n".join(records), add_special_tokens=False).ids
    beginning = "These closed archive entries are background. Only the CURRENT customer needs a new decision.\n"
    ending = (
        "\nCURRENT customer: The replacement arrived yesterday, but my card still shows the old charge and a new "
        "pending authorization. Your colleague said the pending amount should disappear. I can wait until Monday, "
        "but please check what happened and explain the next step. I'm disappointed this needed another message."
    )
    cases = []
    for label, length in (("4k", 4000), ("8k", 7600), ("12k", 12000)):
        state = beginning + tokenizer.decode(archive[:length]) + ending
        cases.append({"name": f"long-{label}-mixed", "request": {"state": state, "questions": QUESTIONS}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(cases)} requests -> {args.out}")


if __name__ == "__main__":
    main()
