"""Kev: prompts and token counts as Kev's own server builds them (references/kev, made by make_reference.py from
github.com/jaredpalmer/kev), and the pointer head's probabilities as model.PointerHead computes them."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from adapter_fixtures import cases, renderings, snapshot, softmax
from ippon.adapter import Hidden
from ippon.adapters.kev import CLOSE, DECIDE, OPEN, QUESTION, STATE, Head, KevFormat
from ippon.schema import parse_request
from ippon.tokenizer import Tokenizer
from media_references import load

REFERENCE = Path(__file__).resolve().parents[1] / "references" / "kev" / "kev-0.8b.json.gz"


def kev_format(head: Head | None = None) -> KevFormat:
    model = load(REFERENCE)["model"]
    d = snapshot(model["repo"], model["revision"], "tokenizer.json", "tokenizer_config.json")
    tok = Tokenizer.from_file(d / "tokenizer.json", json.loads((d / "tokenizer_config.json").read_text()))
    marks = {t: tok.token_id(t) for t in (STATE, QUESTION, OPEN, CLOSE, DECIDE)}
    return KevFormat(tok, head, {k: int(v) for k, v in marks.items() if v is not None})  # type: ignore[arg-type]


def test_prompts_and_token_counts_are_kevs() -> None:
    fmt = kev_format()
    for _, rendering in renderings(fmt, cases(REFERENCE), 15):
        for prompt, q in zip(rendering.prompts, rendering.request.questions, strict=True):
            close = [i for i, t in enumerate(prompt.tokens) if t == fmt.marks[CLOSE]]
            assert [r.position for r in prompt.readouts] == [
                *close[-len(prompt.readouts) + 1 :],
                len(prompt.tokens) - 1,
            ]
            assert prompt.tokens[prompt.marks[0]] == fmt.marks[QUESTION], q.id


def test_the_pointer_head_scores_as_kevs() -> None:
    rng = np.random.default_rng(0)
    d, dp, t = 16, 8, 2.35
    qw, qb, kw, kb = (rng.normal(size=s).astype(np.float32) for s in ((dp, d), (dp,), (dp, d), (dp,)))
    head = Head(mx.array(qw), mx.array(qb), mx.array(kw), mx.array(kb), t)
    fmt = KevFormat(None, head, {})  # type: ignore[arg-type]
    request = parse_request(
        {
            "state": "s",
            "questions": {
                "go": {"type": "noul", "instructions": "?"},
                "team": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, "c": None}},
            },
        }
    )
    rows = {"go": rng.normal(size=(3, d)).astype(np.float32), "team": rng.normal(size=(4, d)).astype(np.float32)}
    hidden = Hidden({k: mx.array(v) for k, v in rows.items()})
    got = fmt.scores(type("R", (), {"request": request})(), hidden)
    for qid, h in rows.items():
        logits = ((h[:-1] @ kw.T + kb) @ (h[-1] @ qw.T + qb)) / math.sqrt(dp)  # PointerHead.forward
        expected = softmax(logits, t)
        if qid == "go":  # Kev scores no, then yes; ippon answers true first
            expected = expected[[1, 0]]
        np.testing.assert_allclose(np.array(got[qid]), expected, atol=1e-5)
