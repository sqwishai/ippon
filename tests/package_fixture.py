"""Export a tiny checkpoint and independent NumPy answers for installed-package checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

from adapter_fixtures import softmax
from test_generic_adapters import LABELS, ids, logits, qwen3_classifier


def export(directory: Path) -> None:
    mx.set_default_device(mx.cpu)
    directory.mkdir(parents=True, exist_ok=True)
    config, weights = qwen3_classifier(directory)
    requests = []
    for index, text in enumerate(("w1 w2", "w3", "w4 w5 w6", "w7 w8 " * 20)):
        requests.append(
            {
                "name": f"text-{index}",
                "request": {
                    "state": text,
                    "questions": {"team": {"type": "choice", "criteria": list(LABELS.values())}},
                },
                "probabilities": {"team": softmax(logits(config, weights, ids(text))).tolist()},
            }
        )
    (directory / "reference.json").write_text(
        json.dumps({"schema": "ippon-reference/1", "bound": 1e-4, "requests": requests}) + "\n"
    )


if __name__ == "__main__":
    export(Path(sys.argv[1]))
