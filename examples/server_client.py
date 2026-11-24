"""Call a model served by `ippon serve` from any language that speaks HTTP; here, Python's standard library.

uv run ippon serve kettei-core:q8 --port 8000        # in one terminal
uv run python examples/server_client.py                     # in another

Requests that arrive while the model is busy are answered together, so clients can send in parallel.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

URL = "http://127.0.0.1:8000/v1/systemone"
TICKETS = [
    "My card was charged twice for order #48213. Please refund the second charge.",
    "The parcel arrived broken and the box was soaked. I want a replacement.",
    "How do I change the email address on my account?",
]


def ask(ticket: str) -> dict[str, Any]:
    body = {
        "state": ticket,
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle it?",
                "criteria": {"billing": "charges and refunds", "shipping": "deliveries", "account": "login"},
            },
            "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        },
    }
    request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer: dict[str, Any] = json.loads(response.read())
    return answer


with ThreadPoolExecutor(len(TICKETS)) as pool:
    for ticket, answer in zip(TICKETS, pool.map(ask, TICKETS), strict=True):
        a = answer["answers"]
        print(f"{a['team']['choice']:8s} refund={a['refund']['noul']:.2f}  {ticket[:50]}")
