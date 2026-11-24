"""What every architecture's Backbone promises the runtime, checked on each family's tiny random checkpoint against
independent NumPy references and the backbone's own reading of every prompt whole:
reading in chunks, question branches off one prefix, resuming a kept prefix, a padded batch, the runtime reading a
model that reads both ways whole, the measuring Ops (precision's PeakOps, profile's TimedOps) changing no number,
and compiled layers giving the eager layers' bits. The sliding-window families' windows (5) are shorter than these
prompts, so chunks and branches cross them.
A family added to arch/ joins through backbone_fixtures and the cases below."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

import backbone_fixtures as fixtures
from bert_numpy import forward as bert_forward
from deberta_numpy import log_bucket
from helpers import encoder_tokens as tokens
from helpers import tokens as decoder_tokens
from ippon.arch import deberta, lfm2
from ippon.arch.base import Backbone
from ippon.arch.ops import FUSED
from ippon.errors import ContextTooLong, UnsupportedModel
from ippon.options import Strategy
from ippon.plan import Prompt, Readout, plan
from ippon.precision import PeakOps
from ippon.prefixes import Prefixes
from ippon.profile import TimedOps
from ippon.runtime import batched, run
from ippon.weights import Weights
from lfm3_numpy import forward as lfm3_forward
from modernbert_numpy import forward as modern_forward

Build = Callable[[Path], Backbone]


def dense(variant: str) -> Build:
    return lambda p: fixtures.dense_model(p, variant)[0]


DECODERS: dict[str, Build] = {
    **{v: dense(v) for v in ("llama", "llama-yarn", "llama-groups", "qwen3")},
    "llama tied": dense("llama-rope3"),
    "qwen2 biases": dense("qwen2"),
    "qwen35 ends in attention": lambda p: fixtures.qwen35_model(p, fixtures.QWEN_ATTENTION)[0],
    "qwen35 ends in deltanet": lambda p: fixtures.qwen35_model(p, fixtures.QWEN_DELTANET)[0],
    "lfm2 ends in attention": lambda p: fixtures.lfm2_model(p, fixtures.LFM_ATTENTION)[0],
    "lfm2 ends in conv": lambda p: fixtures.lfm2_model(p, fixtures.LFM_CONV)[0],
    "gemma3": lambda p: fixtures.gemma3_model(p)[0],
    "gemma4 experts": lambda p: fixtures.gemma4_model(p)[0],
    "gemma4 dense": lambda p: fixtures.gemma4_model(p, enable_moe_block=False)[0],
    "gemma4 ple+kv_shared": lambda p: fixtures.gemma4_model(
        p, enable_moe_block=False, hidden_size_per_layer_input=8, num_kv_shared_layers=2, use_double_wide_mlp=True
    )[0],
}
READ_BOTH_WAYS: dict[str, Build] = {
    "qwen35 non-causal": lambda p: fixtures.qwen35_model(p, fixtures.QWEN_ATTENTION, causal=False)[0],
    "lfm3 set": lambda p: fixtures.lfm3_model(p)[0],
    "modernbert": lambda p: fixtures.modernbert_model(p)[0],
    "bert": lambda p: fixtures.bert_model(p, "bert")[0],
    "xlm-roberta": lambda p: fixtures.bert_model(p, "xlm-roberta")[0],
    "deberta": lambda p: fixtures.deberta_model(p)[0],
}
EVERY = DECODERS | READ_BOTH_WAYS
# rows read from prompts of 9, 4, 13 and 1 tokens: the ends, a few out of order, every row (mean pooling), the only one
ROWS = [[0, 8], [3, 1], list(range(13)), [0]]


def close(got: Any, want: Any) -> None:
    np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-4, rtol=1e-4)


def whole(m: Backbone, prompt: Prompt) -> mx.array:
    """The prompt read in one pass from an empty cache, at its readouts."""
    return m.prefill(list(prompt.tokens), m.new_cache(len(prompt.tokens)), [r.position for r in prompt.readouts])


def questions(state: list[int], marks: tuple[int, ...] = (), *, gpu_branch: bool = False) -> list[Prompt]:
    """Four questions after one state: tails of 3, 6, 1 and 4 tokens, the second read at two positions."""
    tails = [(decoder_tokens if gpu_branch else tokens)(n, 10 + n) for n in (3, 6, 1, 4)]
    prompts = [
        Prompt(tuple(state + t), (Readout(f"q{i}", len(state) + len(t) - 1),), marks) for i, t in enumerate(tails)
    ]
    if not gpu_branch:
        prompts[1] = Prompt(prompts[1].tokens, (Readout("q1", len(state) + 1), Readout("q1", len(state) + 5)), marks)
    return prompts


@pytest.mark.parametrize("family", list(DECODERS))
@pytest.mark.parametrize("chunk", [1, 4])  # a token at a time; chunks shorter than the window (5)
def test_reading_in_chunks_matches_one_pass(tmp_path: Path, family: str, chunk: int) -> None:
    m = DECODERS[family](tmp_path)
    prompt = Prompt(tuple(tokens(21, 3)), (Readout("q", 5), Readout("q", 20)))
    (got,) = run(m, plan([prompt]), chunk=chunk)
    close(got, whole(m, prompt))


@pytest.mark.parametrize(
    ("family", "gpu_branch"),
    [
        *[(f, False) for f in DECODERS],
        *[pytest.param(f"qwen35 ends in {end}", True, marks=pytest.mark.metal) for end in ("attention", "deltanet")],
    ],
)
def test_branches_match_reading_every_prompt_whole(
    tmp_path: Path, family: str, gpu_branch: bool, request: pytest.FixtureRequest
) -> None:
    """Four questions off one prefix, more question rows than a chunk: read in passes of at most a chunk, the later
    ones continuing the cache the first left (so a branch that moved the prefix's state or length shows), each
    question's rows as reading its prompt whole gives them."""
    m = DECODERS[family](tmp_path)
    passes: list[tuple[int, int]] = []  # (head rows, tail rows) of each branch pass
    real = m.branch

    def branch(tails: Any, cache: Any, rows: Any, head: Any = (), marks: Any = ()) -> Any:
        passes.append((len(head), sum(len(t) for t in tails)))
        return real(tails, cache, rows, head, marks)

    m.branch = branch  # type: ignore[method-assign]
    if gpu_branch:
        request.getfixturevalue("gpu")
    prompts = questions((decoder_tokens if gpu_branch else tokens)(17, 4), gpu_branch=gpu_branch)
    for prompt, got in zip(prompts, run(m, plan(prompts), chunk=5), strict=True):
        want = whole(m, prompt)
        if gpu_branch:
            # M5 fp32 recurrence rounds differently across paths; the original GPU bound is 1% of the peak.
            np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-2 * float(mx.max(mx.abs(want))), rtol=0)
        else:
            close(got, want)
    assert [t for _, t in passes] == [3, 6, 5]  # 3 + 6 is past the chunk; a tail of 6 is read alone
    assert [h for h, _ in passes][1:] == [0, 0]  # only the first pass reads the prefix's end
    passes.clear()  # a chunk of 10: the prefix's end and three tails in one pass, each tail's attention apart
    for prompt, got in zip(prompts, run(m, plan(prompts), chunk=10), strict=True):
        want = whole(m, prompt)
        if gpu_branch:
            np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-2 * float(mx.max(mx.abs(want))), rtol=0)
        else:
            close(got, want)
    assert [t for _, t in passes] == [10, 4]


@pytest.mark.parametrize("family", list(EVERY))
def test_prompts_of_mixed_lengths_are_batched_by_length_and_answered_in_their_order(
    tmp_path: Path, family: str
) -> None:
    from ippon.runtime import batched

    m = EVERY[family](tmp_path)
    padded: list[int] = []  # each batch's prompts x its longest
    real = m.batch

    def batch(prompts: Any, rows: Any, groups: Any = None) -> Any:
        padded.append(len(prompts) * max(len(p) for p in prompts))
        return real(prompts, rows, groups)

    m.batch = batch  # type: ignore[method-assign]
    lengths = (3, 17, 4, 15, 2, 16)
    prompts = [Prompt(tuple(tokens(n, 20 + n)), (Readout("q", n - 1),)) for n in lengths]
    got = batched(m, [p.tokens for p in prompts], [[r.position for r in p.readouts] for p in prompts], budget=51)
    for prompt, hidden in zip(prompts, got, strict=True):
        close(hidden, whole(m, prompt))
    assert sum(padded) == 3 * 4 + 3 * 17  # the short three, then the long three: 63 (in their order: 3 x 17 + 3 x 16)


def test_a_batch_the_gpu_cannot_hold_is_read_in_halves(tmp_path: Path) -> None:
    """Metal out of memory on a pass of several prompts (a small Mac): the pass is read in halves, the answers the
    same; a prompt that fails alone fails the request."""
    m = DECODERS["llama"](tmp_path)
    real, sizes = m.batch, []

    def batch(prompts: Any, rows: Any, groups: Any = None) -> Any:
        sizes.append(len(prompts))
        if len(prompts) > 1:
            raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory (00000008)")
        return real(prompts, rows, groups)

    m.batch = batch  # type: ignore[method-assign]
    prompts = [Prompt(tuple(tokens(n, 20 + n)), (Readout("q", n - 1),)) for n in (9, 9, 9)]
    got = batched(m, [p.tokens for p in prompts], [[r.position for r in p.readouts] for p in prompts])
    for prompt, hidden in zip(prompts, got, strict=True):
        close(hidden, whole(m, prompt))
    assert sizes == [3, 1, 2, 1, 1]  # all three, then a half of one and a half of two, split again

    def short(prompts: Any, rows: Any, groups: Any = None) -> Any:
        sizes.append(len(prompts))
        raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory (00000008)")

    sizes.clear()
    m.batch = short  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        batched(m, [prompts[0].tokens], [[8]])
    assert sizes == [1]  # a lone prompt isn't tried again


@pytest.mark.parametrize("family", list(DECODERS))
def test_kept_prefixes_give_the_hidden_states_of_fresh_reads(tmp_path: Path, family: str) -> None:
    """The same state again, a longer one and one sharing only its start, after a store kept the first; a repeat and
    an extension resume from what was kept."""
    m = DECODERS[family](tmp_path)
    store = Prefixes(1 << 30)
    state = tokens(20, 5)
    for s in (state, state, state + tokens(4, 6), state[:9] + tokens(8, 7)):
        prompts = questions(s, marks=(11,))
        for prompt, got in zip(prompts, run(m, plan(prompts), chunk=4, prefixes=store), strict=True):
            close(got, whole(m, prompt))
    assert store.stats.hits >= 2


@pytest.mark.parametrize("family", list(EVERY))
def test_a_batch_matches_reading_every_prompt_alone(tmp_path: Path, family: str) -> None:
    m = EVERY[family](tmp_path)
    prompts = [tokens(n, 30 + n) for n in (9, 4, 13, 1)]
    for p, r, got in zip(prompts, ROWS, m.batch(prompts, ROWS), strict=True):
        close(got, m.prefill(p, m.new_cache(len(p)), r))


@pytest.mark.parametrize("family", list(READ_BOTH_WAYS))
def test_the_runtime_reads_a_model_that_reads_both_ways_whole(tmp_path: Path, family: str) -> None:
    """Questions about one state, each prompt read whole; a branch, which continues a prefix such a model can't
    share, is refused."""
    m = READ_BOTH_WAYS[family](tmp_path)
    prompts = questions(tokens(17, 4))
    for prompt, got in zip(prompts, run(m, plan(prompts), chunk=5), strict=True):
        close(got, whole(m, prompt))
    with pytest.raises(ValueError, match="whole"):
        m.branch([[1, 2], [3]], m.new_cache(16), [[1], [0]], head=[4, 5])


@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_set_prompts_are_read_by_their_questions_as_the_numpy_model_reads_them(
    tmp_path: Path, strategy: Strategy
) -> None:
    """Questions packed after their state (d1's set attention), prompts of different lengths: on the full attention
    layers the state reads every question and each question the state and itself, through the plan and the runtime
    (a padded batch, or each prompt alone), as the independent NumPy model reads each prompt."""
    m, config, weights = fixtures.lfm3_model(tmp_path)
    layouts = [(6, 4, 3, 4), (5, 4), (3, 2, 7)]  # a state's tokens, then each question's
    prompts = []
    for n, (state, *own) in enumerate(layouts):
        groups = [-1] * state + [k for k, size in enumerate(own, 1) for _ in range(size)]
        ends = [state + sum(own[: k + 1]) - 1 for k in range(len(own))]
        readouts = tuple(Readout(f"q{k}", end) for k, end in enumerate(ends))
        prompts.append(Prompt(tuple(decoder_tokens(len(groups), 40 + n)), readouts, groups=tuple(groups)))
    for prompt, got in zip(prompts, run(m, plan(prompts), strategy=strategy), strict=True):
        want = lfm3_forward(config, weights, list(prompt.tokens), prompt.groups)
        close(got, want[[r.position for r in prompt.readouts]])


@pytest.mark.parametrize("family", list(EVERY))
def test_the_measuring_ops_change_no_number(tmp_path: Path, family: str) -> None:
    """precision measures (PeakOps), and profile times (TimedOps), a model by running it with these Ops: every
    family's operations go through them (something is measured) and give the numbers the plain Ops give."""
    m = EVERY[family](tmp_path)
    prompt = Prompt(tuple(tokens(13, 8)), (Readout("q", 0), Readout("q", 12)))
    plain, ops = whole(m, prompt), m.ops
    for wrapper in (PeakOps, TimedOps):
        m.ops = measuring = wrapper.of(ops)
        measured = whole(m, prompt)
        close(measured, plain)
        if isinstance(measuring, PeakOps):
            assert measuring.value() > 0
            if family in ("bert", "deberta"):
                assert measuring.value() >= float(mx.max(mx.abs(measured)))
        else:
            assert sum(measuring.calls.values()) > 0


@pytest.mark.metal
@pytest.mark.parametrize("family", list(EVERY))
def test_compiled_layers_give_the_eager_layers_bits(tmp_path: Path, family: str, gpu: None) -> None:
    """With the compile switch, on the GPU: every pass the runtime makes (a prompt in chunks keeping its marks, the
    same state grown and resumed from what was kept, question branches off it, a padded batch) gives the eager
    layers' outputs bit for bit."""
    m = EVERY[family](tmp_path)

    def answers(compiled: bool) -> list[np.ndarray]:
        m.ops = replace(FUSED, compiled=compiled)
        store = Prefixes(1 << 30) if m.causal else None
        out = []
        state = tokens(20, 5)
        for s in (state, state + tokens(4, 6)):
            prompts = questions(s, marks=(11,))
            out += [np.array(h) for h in run(m, plan(prompts), chunk=4, prefixes=store)]
            out += [np.array(h) for h in run(m, plan(prompts[:1]), chunk=4, prefixes=store)]
        out += [np.array(h) for h in m.batch([tokens(n, 30 + n) for n in (9, 4, 13, 1)], ROWS)]
        return out

    eager = answers(False)
    for want, got in zip(eager, answers(True), strict=True):
        np.testing.assert_array_equal(got, want)


# Cases name only construction and inputs; the numerical driver below owns every assertion.
def reading(
    family: str,
    /,
    n: int,
    seed: int,
    rows: list[list[int]] | None = None,
    split: int = 0,
    bound: float = 2e-4,
    **options: Any,
) -> Any:
    ids = (tokens if family in ("bert", "deberta") else decoder_tokens)(n, seed)
    return pytest.param(
        family, options, ids, rows or [list(range(n))], split, bound, id=f"{family}-{options}-{n}-{seed}-{split}"
    )


@pytest.mark.parametrize(
    ("family", "options", "ids", "selections", "split", "bound"),
    [
        *[reading("dense", 19, 1, variant=v) for v in fixtures.VARIANTS],
        reading("dense", 19, 5, [list(range(8, 19)), list(range(5, 9)), [11, 4, 15], []], variant="qwen3"),
        *[
            reading("bert", 17, 1, [list(range(17)), [0], [16], [3, 9, 1], []], family=v)
            for v in ("bert", "roberta", "xlm-roberta", "distilbert")
        ],
        *[
            reading("deberta", 21, 1, [list(range(21)), [0], [20], [3, 9, 1], []], bound=3e-4, **extra)
            for extra in ({}, {"position_buckets": -1, "max_relative_positions": 16}, {"position_biased_input": True})
        ],
        *[reading("gemma3", 19, 1, rope_scaling=scaling) for scaling in (None, {"rope_type": "linear", "factor": 8.0})],
        *[
            reading("gemma4", 19, 1, [list(range(19)), list(range(12, 19)), []], layout=layout, **extra)
            for layout, extra in (
                ("fused", {}),
                ("switch_glu", {}),
                ("fused", {"enable_moe_block": False}),
                ("fused", {"attention_k_eq_v": False}),
                ("fused", {"nested": True}),
                ("fused", {"enable_moe_block": False, "per_layer": True, "bidirectional": "full"}),  # Jeff 1.1
                (
                    "fused",
                    {
                        "enable_moe_block": False,
                        "hidden_size_per_layer_input": 8,
                        "num_kv_shared_layers": 2,
                        "use_double_wide_mlp": True,
                    },
                ),  # E2B / E4B
            )
        ],
        *[
            reading(
                "lfm2",
                19,
                5,
                [list(range(19)), list(range(8, 19)), list(range(5, 9)), [11, 4, 15], []],
                layer_types=layers,
            )
            for layers in (fixtures.LFM_ATTENTION, fixtures.LFM_CONV)
        ],
        *[
            reading("lfm2", 7, 2, [[6]], layer_types=fixtures.LFM_CONV, root=root, explicit_root=explicit)
            for root, explicit in (
                ("model.language_model.", False),
                ("model.", False),
                ("model.audio_encoder.lfm.", True),
            )
        ],
        *[
            reading("lfm3", 19, 5, [list(range(19)), [11, 4, 15], []], layer_types=layers, window=window)
            for layers in (fixtures.LFM3_LAYERS, ["sliding_attention", "conv", "full_attention"])
            for window in (4, 32)  # a window shorter than the prompt, and one longer
        ],
        *[reading("modernbert", n, n) for n in (3, 7, 20)],
        *[
            reading(
                "qwen35",
                23,
                4,
                [list(range(23)), list(range(10, 23)), list(range(9, 14)), [17, 9, 21], []],
                split=split,
                layer_types=layers,
                causal=causal,
            )
            for layers in (fixtures.QWEN_ATTENTION, fixtures.QWEN_DELTANET)
            for causal, split in ((True, 0), (True, 8), (False, 0))
        ],
        *[
            reading("qwen35", 9, 2, [[8]], layer_types=fixtures.QWEN_DELTANET, root=root)
            for root in ("model.", "model.language_model.", "language_model.")
        ],
    ],
)
def test_selected_rows_match_the_independent_numpy_model(
    tmp_path: Path,
    family: str,
    options: dict[str, Any],
    ids: list[int],
    selections: list[list[int]],
    split: int,
    bound: float,
) -> None:
    build, oracle, _ = fixtures.FAMILIES[family]
    model, config, weights = build(tmp_path, **options)
    kwargs = {"causal": options.get("causal", True)} if family == "qwen35" else {}
    expected = oracle.forward(config, weights, ids, **kwargs)
    for asked in selections:
        cache = model.new_cache(len(ids))
        if split:
            model.prefill(ids[:split], cache, [])
        rows = [r for r in asked if r >= split]
        got = model.prefill(ids[split:], cache, [r - split for r in rows])
        np.testing.assert_allclose(np.array(got).reshape(-1, expected.shape[1]), expected[rows], atol=bound, rtol=bound)


@pytest.mark.parametrize(
    ("family", "config", "message"),
    [
        ("bert", {"model_type": "deberta-v2"}, "not a BERT-family encoder"),
        ("bert", {"position_embedding_type": "relative_key"}, "position_embedding_type 'relative_key'"),
        ("deberta", {"model_type": "bert"}, "not a DeBERTa"),
        ("deberta", {"conv_kernel_size": 3}, "pos_att_type, conv"),
        ("deberta", {"share_att_key": False}, "relative attention or shared keys"),
        ("dense", {"model_type": "gemma3"}, "not a dense decoder"),
        ("dense", {"rope_scaling": {"rope_type": "dynamic", "factor": 4.0}}, "rope_type 'dynamic'"),
        ("dense", {"model_type": "llama", "mlp_bias": True}, "MLP bias"),
        ("gemma3", {"final_logit_softcapping": 30.0}, "soft-capping or bidirectional"),
        ("gemma3", {"model_type": "gemma2"}, "not a Gemma 3"),
        ("gemma4", {"num_kv_shared_layers": 4}, "num_kv_shared_layers"),
        ("gemma4", {"use_bidirectional_attention": "all"}, "bidirectional"),
        ("gemma4", {"model_type": "gemma3_text"}, "not a Gemma 4"),
        ("lfm2", {"model_type": "llama"}, "not an LFM2"),
        ("modernbert", {"mlp_bias": True}, "biases"),
        ("modernbert", {"model_type": "bert"}, "not a ModernBERT"),
    ],
)
def test_unsupported_backbone_formulas_are_refused(family: str, config: dict[str, Any], message: str) -> None:
    _, oracle, parse = fixtures.FAMILIES[family]
    args: dict[str, tuple[Any, ...]] = {
        "bert": ("bert",),
        "dense": ("qwen2",),
        "lfm2": (fixtures.LFM_CONV,),
        "modernbert": (fixtures.MODERN_LAYERS,),
    }
    with pytest.raises(UnsupportedModel, match=message):
        parse(oracle.tiny_config(*args.get(family, ())) | config)


def test_a_pairs_second_text_takes_token_type_one(tmp_path: Path) -> None:
    """[CLS] a [SEP] b [SEP]: type 0 up to the first [SEP], 1 after it, as BERT's tokenizer marks a pair."""
    m, config, weights = fixtures.bert_model(tmp_path, "bert", sep=2)
    ids = [1, *tokens(5, 3), 2, *tokens(4, 4), 2]
    types = [0] * 7 + [1] * 5
    got = m.prefill(ids, m.new_cache(len(ids)), list(range(len(ids))))
    np.testing.assert_allclose(np.array(got), bert_forward(config, weights, ids, types), atol=2e-4, rtol=2e-4)
    single = m.prefill([1, *tokens(5, 3), 2], m.new_cache(len(ids)), [0])
    np.testing.assert_allclose(
        np.array(single), bert_forward(config, weights, [1, *tokens(5, 3), 2])[[0]], atol=2e-4, rtol=2e-4
    )


def test_prompts_past_the_learned_positions_are_refused(tmp_path: Path) -> None:
    m, _, _ = fixtures.bert_model(tmp_path, "roberta")  # 42 positions from offset 2: 40 tokens
    m.prefill(tokens(40, 5), m.new_cache(41), [0])
    with pytest.raises(ContextTooLong, match="41 tokens, more than the model's 40"):
        m.prefill(tokens(41, 5), m.new_cache(41), [0])


def test_the_buckets_are_transformers() -> None:
    """The tiny model's buckets and DeBERTa-v3's (256 of them up to 512 positions), at distances past its window:
    a prompt reaches only the tiny model's nearest ones."""
    rel = np.arange(-600, 601)[None, :]
    np.testing.assert_array_equal(deberta.buckets(rel, 8, 64), log_bucket(rel, 8, 64))
    np.testing.assert_array_equal(deberta.buckets(rel, 256, 512), log_bucket(rel, 256, 512))


def test_a_config_that_disagrees_with_the_weights_is_refused(tmp_path: Path) -> None:
    _, config, weights = fixtures.lfm2_model(tmp_path, fixtures.LFM_CONV)
    tensors = Weights({"model." + k: mx.array(v) for k, v in weights.items()})
    with pytest.raises(UnsupportedModel, match="layer 0"):
        lfm2.LFM2(lfm2.Config.from_hf({**config, "layer_types": ["full_attention", "full_attention", "conv"]}), tensors)


def test_the_window_includes_keys_exactly_half_of_local_attention_away(tmp_path: Path) -> None:
    m, config, weights = fixtures.modernbert_model(
        tmp_path, ["sliding_attention"], local_attention=4
    )  # sees 2 either side
    ids = decoder_tokens(9, 1)
    want = modern_forward(config, weights, ids)
    got = np.array(m.prefill(ids, m.new_cache(9), list(range(9))))
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)
    narrower = dict(config, local_attention=2)
    assert not np.allclose(modern_forward(narrower, weights, ids), want, atol=1e-3)  # the edge matters


def test_padding_far_past_a_short_prompt_stays_finite(tmp_path: Path) -> None:
    m, config, weights = fixtures.modernbert_model(
        tmp_path, ["sliding_attention", "sliding_attention"], local_attention=2
    )
    short, long = decoder_tokens(2, 7), decoder_tokens(30, 8)
    got = m.batch([short, long], [[0, 1], list(range(30))])
    assert all(bool(mx.all(mx.isfinite(g))) for g in got)
    np.testing.assert_allclose(np.array(got[0]), modern_forward(config, weights, short), atol=2e-4, rtol=2e-4)


def test_a_long_pass_reads_its_windows_in_blocks(tmp_path: Path) -> None:
    from ippon.arch import common

    m, config, weights = fixtures.modernbert_model(tmp_path)  # a 3-row window
    n = common.BAND_WINDOWS * 3 + common.BAND + 37  # blocks past the band's start, and a last one cut short
    ids = decoder_tokens(n, 3)
    got = m.prefill(ids, m.new_cache(n), list(range(n)))
    np.testing.assert_allclose(np.array(got), modern_forward(config, weights, ids), atol=2e-4, rtol=2e-4)
    short, long = (
        decoder_tokens(40, 4),
        decoder_tokens(n, 5),
    )  # a padded batch: the short prompt's padding past its window
    out = m.batch([short, long], [list(range(40)), list(range(n))])
    np.testing.assert_allclose(np.array(out[0]), modern_forward(config, weights, short), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.array(out[1]), modern_forward(config, weights, long), atol=2e-4, rtol=2e-4)
