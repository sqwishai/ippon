"""Prefixes kept across requests: a state sent again is not read again, a grown or changed state resumes from the
last position kept before it changes, and the answers are those of reading every request fresh."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from backbone_fixtures import QWEN_ATTENTION, QWEN_DELTANET, qwen35_model, tokens
from ippon import Decider, Loaded
from ippon.arch.base import Backbone
from ippon.options import AWAKE_WINDOW_S
from ippon.plan import Prompt, Readout, plan
from ippon.prefixes import Prefixes
from ippon.runtime import run
from ippon.schema import parse_request
from ippon.serve import Model, Models
from toy import ToyBackbone, ToyFormat

STATE = "The customer was charged twice for one order and wants the second charge back."


def request(state: str, n: int = 3) -> dict[str, Any]:
    return {"state": state, "questions": {f"q{i}": {"type": "noul", "instructions": f"Q{i}?"} for i in range(n)}}


def decider(**options: Any) -> tuple[Decider, ToyBackbone]:
    backbone = ToyBackbone()
    options.setdefault("prefix_budget", 1 << 20)
    return Decider(Loaded(backbone, ToyFormat(backbone)), **options), backbone


def fresh(body: dict[str, Any]) -> Any:
    return decider(prefix_budget=0)[0].systemone(body)["answers"]


@pytest.mark.parametrize(
    ("state", "questions", "resumed", "entries"),
    [
        (STATE, 3, len(STATE), 1),
        (STATE, 5, len(STATE), 1),
        (STATE + " It happened again today.", 3, len(STATE), 1),
        (STATE[:21] + "x" + STATE[22:], 3, 16, 2),
    ],
    ids=["repeat", "other-questions", "grown", "changed"],
)
def test_cached_states_read_only_the_unshared_tokens(state: str, questions: int, resumed: int, entries: int) -> None:
    d, backbone = decider(chunk=8)
    d.systemone(request(STATE))
    read = backbone.tokens_read
    body = request(state, questions)
    result = d.answer(parse_request(body))
    assert result.response["answers"] == fresh(body)
    assert result.diagnostics["kept_tokens"] == resumed  # what the request itself says it didn't read again
    assert backbone.tokens_read - read == len(state) - resumed + questions * 2
    assert d.prefixes is not None
    assert d.prefixes.stats.hits == 1
    assert len(d.prefixes) == entries


def test_the_least_recently_used_prefix_goes_first_beyond_the_budget() -> None:
    d, _ = decider(prefix_budget=64)  # the toy keeps 32 bytes per position, and one position per prefix here
    assert d.prefixes is not None
    states = [STATE, STATE.upper(), STATE.lower()]
    d.systemone(request(states[0]))
    d.systemone(request(states[1]))
    d.systemone(request(states[0]))  # used again: states[1] is now the oldest
    d.systemone(request(states[2]))
    assert len(d.prefixes) == 2
    assert d.prefixes.nbytes <= 64
    hits = d.prefixes.stats.hits
    d.systemone(request(states[0]))
    assert d.prefixes.stats.hits == hits + 1
    d.systemone(request(states[1]))
    assert d.prefixes.stats.hits == hits + 1  # dropped


def test_nothing_is_kept_without_a_budget_or_for_a_non_causal_model() -> None:
    assert decider(prefix_budget=0)[0].prefixes is None
    backbone = ToyBackbone()
    backbone.causal = False
    assert Decider(Loaded(backbone, ToyFormat(backbone)), prefix_budget=1 << 20).prefixes is None


def prompts(state: list[int], marks: tuple[int, ...] = ()) -> list[Prompt]:
    tails = [tokens(n, 20 + n) for n in (3, 5, 2)]
    return [Prompt(tuple(state + t), (Readout(f"q{i}", len(state) + len(t) - 1),), marks) for i, t in enumerate(tails)]


@pytest.mark.parametrize("layer_types", [QWEN_ATTENTION, QWEN_DELTANET])
@pytest.mark.parametrize("head", ["branches", "every-row"])
def test_resumed_hybrid_heads_match_fresh_reads(tmp_path: Path, layer_types: list[str], head: str) -> None:
    m, _, _ = qwen35_model(tmp_path, layer_types)
    store = Prefixes(1 << 30)
    state = tokens(23, 1)
    states = [state, state, state + tokens(6, 2), state[:13] + tokens(10, 3)]
    if head == "branches":
        states.append(state)  # later requests must never mutate the original kept keys
    first = None
    for i, s in enumerate(states):
        question = tokens(7, 6) if i == 1 else tokens(4, 5)
        ps = every_row(s, question, (12, len(s))) if head == "every-row" else prompts(s, marks=(12,))
        p = plan(ps)
        kept = run(m, p, chunk=5, prefixes=store)
        reference = run(m, p, strategy="reference")
        for k, r in zip(kept, reference, strict=True):
            np.testing.assert_allclose(np.array(k), np.array(r), atol=2e-5, rtol=2e-5)
        if i == 0:
            first = [np.array(k) for k in kept]
    assert store.stats.hits == len(states) - 1
    assert store.stats.tokens_reused == 23 + 23 + 12 + (23 if head == "branches" else 0)
    if head == "branches":
        assert first is not None
        for k, f in zip(kept, first, strict=True):
            np.testing.assert_allclose(np.array(k), f, atol=1e-6)


def test_new_questions_about_a_kept_state_keep_it_once(tmp_path: Path) -> None:
    """A lone question's prefix runs past the state's mark into the question. Kept, that stretch matched no later
    question and made a second copy of the whole state, evicting the first under a budget that holds one."""
    m, _, _ = qwen35_model(tmp_path, QWEN_ATTENTION)
    store = Prefixes(1 << 30)
    state = tokens(23, 1)
    for question in [tokens(4, 5), tokens(6, 9)]:
        p = plan([Prompt(tuple(state + question), (Readout("q", len(state) + len(question) - 1),), (len(state),))])
        np.testing.assert_allclose(
            np.array(run(m, p, chunk=5, prefixes=store)[0]),
            np.array(run(m, p, strategy="reference")[0]),
            atol=2e-5,
            rtol=2e-5,
        )
    assert store.stats.tokens_reused == len(state)
    assert len(store) == 1


def every_row(state: list[int], question: list[int], marks: tuple[int, ...]) -> list[Prompt]:
    """A head that reads every position (Clef's): one prompt, a readout per token."""
    ids = tuple(state + question)
    return [Prompt(ids, tuple(Readout("q", i) for i in range(len(ids))), marks)]


@pytest.mark.metal
@pytest.mark.parametrize(
    ("family", "chunk", "marks", "allowance"),
    [("hybrid", 512, (1400, 1450), 2.0), ("attention", 4096, (1400,), 1.25)],
)
def test_kept_states_release_the_pass_they_came_from(
    gpu: None, tmp_path: Path, family: str, chunk: int, marks: tuple[int, ...], allowance: float
) -> None:
    """Recurrent snapshots and attention keys own compact evaluated arrays, not the pass or its cache views."""
    import gc

    import mlx.core as mx

    import dense_numpy
    import qwen35_numpy
    from ippon.arch import dense, qwen35

    load: Callable[..., Backbone]
    if family == "hybrid":
        config = qwen35_numpy.tiny_config(QWEN_ATTENTION) | {
            "hidden_size": 256,
            "intermediate_size": 512,
            "linear_num_key_heads": 8,
            "linear_key_head_dim": 32,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 64,
        }
        weights = {"model." + k: v for k, v in qwen35_numpy.tiny_weights(config).items()}
        load = qwen35.load
    else:
        config = dense_numpy.tiny_config("llama", hidden_size=256, num_attention_heads=8, num_key_value_heads=8)
        weights = dense_numpy.tiny_weights(config)
        load = dense.load
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    m = load(str(tmp_path), config, dtype=mx.float32)
    store = Prefixes(1 << 30)
    gc.collect()
    before = mx.get_active_memory()
    for seed in range(6):
        state = tokens(1500, 100 + seed)
        run(m, plan(prompts(state, marks=marks)), chunk=chunk, prefixes=store)
    mx.synchronize()
    gc.collect()
    grown = mx.get_active_memory() - before
    assert len(store) == 6
    assert grown < allowance * store.nbytes + (8 << 20), (grown, store.nbytes)


@pytest.mark.parametrize("case", ["budget", "short-of-memory", "default-budget"])
def test_a_state_grown_each_request_keeps_resuming_within_the_budget(tmp_path: Path, case: str) -> None:
    """An agent's watcher: each request's state is the last one plus a step. Each kept entry inherits the marks of
    the one it extends, and on a hybrid model each mark holds a recurrent state, so keeping all of them outgrows any
    budget after a few dozen requests and later requests stop resuming (H2O-Lightning-4B, M5 Pro: past 5k tokens
    under a server's share, 50 MB a mark). A server short of memory shrinks the budget below one such entry, and
    the growing state still stays, alone; so does one outgrowing a default budget, until the server's requests stop."""
    m, _, _ = qwen35_model(tmp_path, QWEN_DELTANET)
    store = Prefixes(1 << 30) if case != "default-budget" else Prefixes(1, grows=True)
    models = Models([Model("m", lambda: None, decider=SimpleNamespace(prefixes=store, closed=False))])  # type: ignore[arg-type,return-value]
    state = tokens(9, 1)
    for i in range(40):
        p = plan([Prompt(tuple(state + tokens(3, 7)), (Readout("q", len(state) + 2),), (len(state),))])
        before = store.stats.tokens_reused
        kept = run(m, p, chunk=5, prefixes=store)
        if i == 0:
            if case == "short-of-memory":  # a server's share cut to almost nothing: the state stays alone in it
                models.prefix_scale = 1e-9
                models.share()
            elif case == "budget":
                store.limit(4 * store.nbytes)  # room for a few entries of four marks: forty would not fit
            else:
                models.share()  # a server on the default budget: the growing state outgrows its 1 byte
        else:
            assert store.stats.tokens_reused - before == len(state) - 2  # all of the last state
            np.testing.assert_allclose(np.array(kept[0]), np.array(run(m, p, strategy="reference")[0]), atol=2e-5)
        state = state + tokens(2, 30 + i)
    if case != "budget":
        models.ready(models.models[0])  # the last request's answer: rest counts from here
        models.rest(time.monotonic() + 1)  # requests came within the last 300 s: still kept
        assert len(store) == 1
        models.rest(time.monotonic() + AWAKE_WINDOW_S)  # none since: given back
        assert store.nbytes == 0
