"""Print boundary requests for the creator's 32,768-token default, without loading a model.

    python references/perplexity/overlength_requests.py > default-limit.requests.json

Feed that file to make_reference.py --tokens-only. The resulting reference includes these requests, the creator's
exact tokens for two questions that each fit with the full state although together they would not, and its rejection
of an oversized single question.
"""

import json

CASES = [
    {
        "name": "default-limit-pair",
        "request": {
            "state": "order " * 16_300,
            "questions": {f"q{i}": {"type": "noul", "instructions": "review " * 16_300} for i in range(2)},
        },
    },
    {
        "name": "default-limit-single-refused",
        "request": {"state": "order " * 32_768, "questions": {"q": {"type": "noul", "instructions": "Approve?"}}},
    },
]

if __name__ == "__main__":
    print(json.dumps(CASES))
