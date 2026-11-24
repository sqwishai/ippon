"""Checks tests/gemma4_numpy.py (the numpy Gemma 4 the MLX decoder is tested against) against transformers' own
Gemma4TextModel, in float64 on tiny random checkpoints: MoE and dense, crossing the sliding window. Needs torch and
transformers >= 5.17 (not ippon's environment):

    python -I scripts/check_gemma4_numpy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from gemma4_numpy import ROOT, forward, tiny_config, tiny_weights


def check(name: str, **extra: object) -> float:
    c = tiny_config(**extra)
    w = tiny_weights(c, seed=3)
    config = Gemma4TextConfig(**{k: v for k, v in c.items() if k != "model_type"})
    config._experts_implementation = "eager"  # Gemma4TextExperts.forward itself (grouped_mm takes no float64)
    model = Gemma4TextModel(config).double().eval()
    state = {k.removeprefix(ROOT): torch.tensor(v, dtype=torch.float64) for k, v in w.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all("rotary" in k or "embed_scale" in k for k in missing), missing
    ids = [int(t) for t in np.random.default_rng(7).integers(0, c["vocab_size"], 19)]
    with torch.no_grad():
        got = model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0].numpy()
        x, at = torch.zeros(1, len(ids), 1, dtype=torch.float64), torch.arange(len(ids))[None]
        own = {kind: tuple(a[0].numpy() for a in model.rotary_emb(x, at, kind)) for kind in set(c["layer_types"])}
    scale = float(model.embed_tokens.embed_scale)
    worst = float(np.abs(got - forward(c, w, ids, own, scale)).max())  # transformers' float32 tables and scale
    exact = float(np.abs(got - forward(c, w, ids)).max())
    print(f"{name}: max |transformers - numpy| = {worst:.2e} with its tables, {exact:.2e} with exact ones")
    return worst


worst = max(
    check("moe, k_eq_v, proportional RoPE"),
    check("dense", enable_moe_block=False),
    check("separate values", attention_k_eq_v=False),
)
assert worst < 1e-4, worst  # transformers runs its norms, router softmax and RoPE tables in float32
print("ok")
