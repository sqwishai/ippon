"""Answer three decisions about a support ticket with Kettei Core, on this Mac.

uv init --bare && uv add ippon && uv run python quickstart.py [kettei-core|kettei-dot]
"""

import sys

from ippon import Choice, Decider, Score, YesNo

# Kettei Core at 8-bit by default (about half its memory, so it fits a 16 GB Mac), or kettei-dot
with Decider.load(sys.argv[1] if len(sys.argv) > 1 else "kettei-core:q8") as model:
    result = model.decide(
        "Order #48213 was charged twice on my card. Please reverse the second charge by Friday or I go to my bank.",
        {
            "refund": YesNo("Is the customer asking for a refund?"),
            "team": Choice("Which team should handle it?", ["billing", "shipping", "support"]),
            "mood": Score(
                "How upset is the customer?",
                [
                    "Calm: no complaint",
                    "Annoyed: unhappy but patient",
                    "Furious: threatening to leave or escalate",
                ],
            ),
        },
    )

print(f"refund: {result['refund'].p_yes:.3f}")
print(f"team:   {result['team'].top} {result['team'].by_option}")
print(f"mood:   {result['mood'].expected_level:.2f} of 2")
print(f"{result.response['usage']['input_tokens']} tokens read")
