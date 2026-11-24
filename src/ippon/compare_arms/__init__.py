"""The runtimes `ippon bench --compare` and `ippon verify --against` time and check ippon against, one script each.

Every arm is a script run as its own process by the Python that has its runtime installed:

    PYTHON -I ARM.py MODEL_DIR REQUESTS.json OUT.json

It loads the model the way the model's card documents, answers every request in REQUESTS.json in order (sleeping
an item's "gap" seconds first), timing each call in its own process, and writes {"runtime": ..., "dtype": ...,
"results": [{"ms", "answers": {question: {option: probability}}, "input_tokens"}, ...]} to OUT.json. With ARM_DTYPE
set (float32) in its environment, an arm that can computes in that dtype and records it as its "dtype". An arm that
runs another runtime on its own (community_clef, glide_package, transformers_byo) imports that runtime and the
standard library only, never ippon, so it runs in that runtime's own environment as it is; the arms that run another
runtime's layers under ippon's prompts and heads (the *_model arms), and ippon itself (ours), share harness.py's loop,
and the *_model arms its `ippon_format`.

This module is the registry: which arms there are and how their answers compare (`difference`, which `ippon verify`
and `ippon bench --ab` measure answers by too); `lookup` finds the ones that run a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Arm:
    script: str  # in this folder
    needs: str | None = None  # a file the model's folder must hold for the arm to run it


ARMS = {
    "community-clef": Arm("community_clef.py", needs="clef_mlx.py"),  # a community build's own loader
    "glide-package": Arm("glide_package.py", needs="glide/model.py"),  # Fastino's own package, on MPS
    "mlx-vlm-model": Arm("mlx_vlm_model.py"),  # mlx-vlm's layers under ippon's prompts and head
    "mlx-lm-model": Arm("mlx_lm_model.py"),  # mlx-lm's layers under ippon's prompts and head (bring-your-own decoders)
    "mlx-embeddings-model": Arm("mlx_embeddings_model.py"),  # mlx-embeddings' encoders, ippon's prompts and head
    "transformers": Arm("transformers_byo.py"),  # transformers' own model and head, on the GPU through MPS
}
ARM_DTYPE = "IPPON_ARM_DTYPE"  # the dtype an arm computes in, where it can recast a build's weights


def answers(response: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """A System One response's answers as {question: {option: probability}} (a yes/no: yes = P(true))."""
    out = {}
    for name, a in response["answers"].items():
        if a.get("type") == "noul" or "noul" in a:
            out[name] = {"yes": float(a["noul"]), "no": 1 - float(a["noul"])}
        else:
            out[name] = {str(k): float(v) for k, v in a["probabilities"].items()}
    return out


def top(probabilities: Sequence[float]) -> int:
    """The most likely option's place (the first, on a tie)."""
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def difference(got: Sequence[Sequence[float]], want: Sequence[Sequence[float]]) -> tuple[float, list[int]]:
    """How far paired answers are apart, each one question's probabilities with its options in one order: the largest
    difference between two probabilities (0 for none), and which of them have another top answer. ValueError when
    they don't pair up (another number of answers, or of an answer's options): no difference can be measured."""
    if len(got) != len(want) or any(len(g) != len(w) for g, w in zip(got, want, strict=True)):
        raise ValueError("answers and reference differ in shape")
    pairs = list(zip(got, want, strict=True))
    largest = max((abs(a - b) for g, w in pairs for a, b in zip(g, w, strict=True)), default=0.0)
    return largest, [i for i, (g, w) in enumerate(pairs) if g and top(g) != top(w)]


def agreement(
    ours: Sequence[dict[str, dict[str, float]]], theirs: Sequence[dict[str, dict[str, float]]]
) -> dict[str, Any]:
    """The largest probability difference over the options both name, how many questions' most likely option
    differs, and how many questions' options don't match (a different option naming), which are left out."""
    got: list[list[float]] = []
    want: list[list[float]] = []
    unmatched = 0
    for a, b in zip(ours, theirs, strict=True):
        for question, pa in a.items():
            pb = b.get(question)
            if pb is None or set(pa) != set(pb):
                unmatched += 1
                continue
            got.append(list(pa.values()))
            want.append([pb[o] for o in pa])
    largest, changed = difference(got, want)
    return {"max_probability_difference": round(largest, 6), "top_answers_differ": len(changed), "unmatched": unmatched}
