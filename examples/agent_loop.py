"""An agent that asks the same decisions after every step of a growing conversation.

    uv init --bare && uv add ippon && uv run python agent_loop.py [kettei-core|kettei-dot]

Each request re-sends the transcript so far. ippon, which runs the model, keeps what it has read, so when the next
transcript starts with the same text it reads only the new message, and each step answers in a few milliseconds.
"""

import sys
import time

from ippon import Choice, Decider, YesNo

questions = {
    "resolved": YesNo("Has the customer's problem been resolved?"),
    "next": Choice("What should the agent do next?", ["ask for details", "refund", "escalate", "close"]),
}

# Kettei Core at 8-bit by default (about half its memory, so it fits a 16 GB Mac), or kettei-dot
# accept_license=True accepts the Kettei Community Model License, which Kettei asks for once before downloading
with Decider.load(sys.argv[1] if len(sys.argv) > 1 else "kettei-core:q8", accept_license=True) as model:
    transcript = ""
    for role, text in [
        ("customer", "Hi, I was charged twice for my order #48213."),
        ("agent", "Sorry about that. Can you confirm the last four digits of the card?"),
        ("customer", "It's 4417. Both charges were $129."),
        ("agent", "Thanks, I've refunded the duplicate charge; it will show in 3-5 days."),
        ("customer", "Great, that's all I needed. Thanks!"),
    ]:
        transcript += f"{role}: {text}\n"
        started = time.perf_counter()
        result = model.decide(transcript, questions)
        ms = (time.perf_counter() - started) * 1000
        print(
            f"after {transcript.count(chr(10))} messages, {ms:.0f} ms: "
            f"resolved {result['resolved'].p_yes:.2f}, next {result['next'].top}"
        )
