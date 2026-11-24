"""The starter requests docs/evaluating.md shows, by name: every one parses, lints clean and answers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DOC = Path(__file__).resolve().parents[1] / "docs/evaluating.md"


def starters() -> dict[str, dict[str, Any]]:
    section = DOC.read_text().split("## Starter requests", 1)[1]
    found = re.findall(r"\*\*([a-z-]+)\*\* [^\n]*(?:\n[^\n`][^\n]*)*\n\n```json\n(.*?)\n```", section, re.S)
    assert found, f"{DOC}: no starter requests"
    return {name: json.loads(body) for name, body in found}
