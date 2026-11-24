"""Jeff: a tiny Jeff-shaped checkpoint (Gemma 4 12B's unified config, full layers read both ways, Jeff's tokenizer, a
random decision head) answered end to end against an independent numpy of the creator's math, and a wide choice in
decide_wide's groups and product."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import gemma4_numpy
from adapter_fixtures import cases, renderings, snapshot
from ippon import Decider
from ippon.adapters import qwen
from ippon.adapters.jeff import LETTERS, WIDE, JeffFormat, gemma_chat, options, prompt_text, split_groups
from ippon.schema import Question
from ippon.tokenizer import Tokenizer
from media_references import load

REPO, REVISION = "jgeuter/Jeff-1.1-Medium", "540c40924dd67c715d3367309ba6744bac754b00"
RULE = {"choice_t": 1.42, "noul_t": 1.35, "score_t": 1.06, "noul_edge": 0.0}


def tiny_jeff(tmp_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    source = snapshot(REPO, REVISION, "tokenizer.json", "tokenizer_config.json")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(source / name, tmp_path / name)
    vocab = Tokenizer.from_file(tmp_path / "tokenizer.json").backend.get_vocab_size(with_added_tokens=True)
    c = gemma4_numpy.tiny_config(enable_moe_block=False, vocab_size=vocab, bidirectional="full")
    w = gemma4_numpy.tiny_weights(c)
    full = {"head_dim": c["global_head_dim"], "num_key_value_heads": c["num_global_key_value_heads"]}
    stored = {k: v for k, v in c.items() if k not in ("global_head_dim", "num_global_key_value_heads", "bidirectional")}
    stored |= {"model_type": "gemma4_unified_text", "per_layer_config": {
        f"{i:02d}": full for i, kind in enumerate(c["layer_types"]) if kind == "full_attention"
    }}  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(stored))
    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), {k: mx.array(v) for k, v in w.items()})
    index = {"weight_map": dict.fromkeys(w, "model-00001-of-00001.safetensors")}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    rng = np.random.default_rng(3)
    d = c["hidden_size"]
    head = {
        "base_rows": rng.standard_normal((16, d)).astype(np.float32),
        "delta.weight": (rng.standard_normal((16, d)) * 0.3).astype(np.float32),
        "delta.bias": (rng.standard_normal(16) * 0.3).astype(np.float32),
        "mu": (rng.standard_normal(d) * 0.1).astype(np.float32),
        "sigma": (1 + rng.random(d)).astype(np.float32),
    }
    mx.save_safetensors(str(tmp_path / "head.safetensors"), {k: mx.array(v) for k, v in head.items()})
    for name, body in {
        "jeff_config.json": {"prompt_version": "direct-options-v1", "bidirectional": "full", "max_input_tokens": 8192},
        "head_config.json": {"kind": "residual", "k_slots": 16, "extra_layers": [], "softcap": 30.0},
        "serving_rule.json": RULE,
    }.items():
        (tmp_path / name).write_text(json.dumps(body))
    return c, w, head


def maker_probabilities(
    c: dict[str, Any], w: dict[str, np.ndarray], head: dict[str, np.ndarray], ids: list[int], k: int
) -> np.ndarray:
    """head.py's DecisionHead on the last position of the numpy forward (its hidden state rounded to bf16 first)."""
    h = gemma4_numpy.forward(c, w, ids)[-1].astype(np.float32)
    h = np.array(mx.array(h).astype(mx.bfloat16).astype(mx.float32))
    z = 30.0 * np.tanh(h @ head["base_rows"][:k].T / 30.0)
    z = z + ((h - head["mu"]) / head["sigma"]) @ head["delta.weight"][:k].T + head["delta.bias"][:k]
    e: np.ndarray = np.exp(z - z.max())
    return e / e.sum()


def test_a_tiny_jeff_answers_as_the_makers_math(tmp_path: Path) -> None:
    c, w, head = tiny_jeff(tmp_path)
    decider = Decider.load(tmp_path, precision="fp32", pin_memory=False, keep_awake=None)
    tok = Tokenizer.from_file(tmp_path / "tokenizer.json")
    state = {"ticket": "Refund not received after 14 days", "tier": "gold"}
    asked: dict[str, Any] = {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": None}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    }
    result = decider.decide(state, asked)
    for qid in asked:
        q = result.answers[qid].question
        ids = tok.encode(prompt_text(state, q, gemma_chat))
        expected = maker_probabilities(c, w, head, ids, len(options(q)))
        np.testing.assert_allclose(result.answers[qid].probabilities, expected, atol=2e-4)
    served = decider.systemone({"state": state, "questions": asked})["answers"]
    p_yes = result.answers["urgent"].probabilities[0]
    tempered = 1 / (1 + math.exp(-math.log(p_yes / (1 - p_yes)) / RULE["noul_t"]))
    assert served["urgent"]["noul"] == (0.801 if 0.5 < tempered < 0.8 else 0.199 if 0.2 < tempered < 0.5 else tempered)
    assert set(served["team"]) == {"type", "choice", "probabilities"}
    assert gemma_chat("S", "U").startswith("<bos><|turn>system\nS<turn|>\n<|turn>user\nU<turn|>\n<|turn>model\n")


def test_a_wide_choice_takes_decide_wides_groups_and_product() -> None:
    names = [f"o{i}" for i in range(17)]
    groups = split_groups(names, len(LETTERS))
    assert [len(g) for g in groups] == [8, 9]  # split_groups: bounds at round(8.5) = 8
    first = [np.full(len(g), 1 / len(g)).tolist() for g in groups]
    first[1] = [0.5] + [0.5 / 8] * 8
    final = {"o0": 0.25, "o8": 0.75}  # each group's winner: o0 (a tie, the first), o8
    q = Question("q", "choice", "which?", tuple((n, None) for n in names))
    p = WIDE.combine(q, groups, first, final, WIDE)
    assert math.isclose(sum(p), 1.0)
    assert math.isclose(p[0], 0.25 / 8)
    assert math.isclose(p[8], 0.75 * 0.5)
    assert math.isclose(p[9], 0.75 * 0.5 / 8)


QWEN_REFERENCE = Path(__file__).resolve().parents[1] / "references" / "jeff" / "jeff-1.1-qwen-4b.json.gz"


def test_jeff_on_qwen_prompts_and_counts_as_the_makers() -> None:
    """Jeff-1.1-Qwen-4B's prompts, Qwen's chat template with thinking off, token for token as the creator's server
    (v1.4.0) builds them."""
    model = load(QWEN_REFERENCE)["model"]
    d = snapshot(model["repo"], model["revision"], "tokenizer.json", "tokenizer_config.json")
    tok = Tokenizer.from_file(d / "tokenizer.json", json.loads((d / "tokenizer_config.json").read_text()))
    fmt = JeffFormat(tok, None, {}, qwen.chat)  # type: ignore[arg-type]
    renderings(fmt, cases(QWEN_REFERENCE), 15)
