"""Pinned adapter inputs and shared creator-reference checks; numerical oracles stay independent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ippon.adapter import Rendering
from ippon.schema import Question, parse_request
from media_references import load


def snapshot(repo: str, revision: str, *patterns: str) -> Path:
    """Fetch only the named small checkpoint files; retain the existing offline skip."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo, revision=revision, allow_patterns=list(patterns)))
    except Exception as error:
        pytest.skip(f"{repo}'s files are not available: {error}")


def cases(path: Path) -> list[dict[str, Any]]:
    return list(load(path)["requests"])


def renderings(fmt: Any, found: list[dict[str, Any]], minimum: int) -> list[tuple[dict[str, Any], Rendering]]:
    """Check every creator prompt and usage, then expose family-specific readouts to the caller."""
    assert len(found) >= minimum
    rendered = []
    for case in found:
        rendering = fmt.render(parse_request(case["request"]))
        assert [list(p.tokens) for p in rendering.prompts] == case["prompts"], case["name"]
        assert fmt.usage(rendering)["input_tokens"] == case["input_tokens"], case["name"]
        rendered.append((case, rendering))
    return rendered


def question(body: dict[str, Any]) -> Question:
    return parse_request({"state": "s", "questions": {"q": {"instructions": "?", **body}}}).questions[0]


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    e = np.exp((x - x.max()) / temperature)
    return np.asarray(e / e.sum())
