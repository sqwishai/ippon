"""The loop and the output of the arms that run on ippon's code (ours, the *_model arms): every request
of REQUESTS.json answered in order, an item's "gap" slept first and each call timed, written as compare.py reads it.
The arms that run another runtime alone (community_clef, glide_package, transformers_byo) can't import ippon in that
runtime's environment, so they keep their own copy of this loop."""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx

from ..adapter import LoadContext
from ..model_dir import ModelDir
from ..registry import select
from . import ARM_DTYPE, answers

ASKED = os.environ.get(ARM_DTYPE)  # the dtype `verify --against` asks for (float32), recorded by `write`
DTYPE = ASKED or "float16"  # what the *_model arms compute in: fp16, as people run them, unless asked
COMPUTE = getattr(mx, DTYPE)


def ippon_format(model_dir: str, arm: str) -> tuple[ModelDir, str, Any, list[str]]:
    """What a *_model arm runs another runtime's layers under: the model's folder, its adapter's name, ippon's format
    (its prompts and head; the layers ippon loaded with it freed) and the model*.safetensors files that runtime
    loads. Exits when the model's adapter doesn't list `arm` in its `compare_arms`."""
    directory = ModelDir.open(model_dir)
    adapter = select(directory)
    if arm not in adapter.compare_arms:
        raise SystemExit(f"{adapter.name}: {arm} doesn't run its models")
    fmt = adapter.load(directory, LoadContext(dtype=COMPUTE)).format
    gc.collect()
    mx.clear_cache()
    return directory, adapter.name, fmt, sorted(p.name for p in directory.path.glob("model*.safetensors"))


def time_requests(requests: str, answer: Callable[[dict[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each item's body answered by `answer` (a System One response), timed: {"ms", "answers", "input_tokens"}."""
    with open(requests, encoding="utf-8") as f:
        items = json.load(f)
    results = []
    for item in items:
        if item["gap"]:
            time.sleep(item["gap"])
        start = time.perf_counter()
        response = answer(item["body"])
        ms = (time.perf_counter() - start) * 1000
        results.append({"ms": ms, "answers": answers(response), "input_tokens": response["usage"]["input_tokens"]})
    return results


def write(out: str, runtime: str, results: list[dict[str, Any]], dtype: str | None = None) -> None:
    """The arm's output; `dtype` the one it computed in, when it was asked for one (ARM_DTYPE)."""
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"runtime": runtime, "dtype": dtype, "results": results}, f)


def budget_groups(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Consecutive prompts (their indices) in batches of at most `budget` tokens once padded to the batch's longest;
    a prompt longer than that alone."""
    out: list[list[int]] = []
    group: list[int] = []
    longest = 0
    for i, n in enumerate(lengths):
        if group and max(longest, n) * (len(group) + 1) > budget:
            out.append(group)
            group, longest = [], 0
        group.append(i)
        longest = max(longest, n)
    return [*out, group] if group else out
