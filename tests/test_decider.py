"""The decider end to end on the toy model: sharing the prefix, reading in chunks and branching must give the
answers of running every prompt whole; builders and System One dicts must give the same answers."""

from __future__ import annotations

import base64
import dataclasses
import itertools
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from ippon import Choice, Decider, InvalidRequest, Loaded, Score, UnsupportedModel, YesNo
from ippon.answers import Answers
from ippon.errors import ContextTooLong, DecisionError, InsufficientMemory, NonFiniteAnswer
from ippon.media import as_media
from ippon.model_dir import ModelDir
from ippon.registry import select
from ippon.rounds import Wide
from ippon.schema import parse_request
from toy import ToyBackbone, ToyFormat

NOT_LISTS: list[Any] = [None, 7, b"low", {"low", "high"}]  # options or levels as anything but a list
STATE = "The customer was charged twice for one order and wants the second charge back."
REQUEST: dict[str, Any] = {
    "state": STATE,
    "questions": {
        "refund": {"type": "noul", "instructions": "Is a refund asked for?"},
        "team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "sales", "support"]},
        "mood": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
    },
}


def decider(**options: Any) -> tuple[Decider, ToyBackbone]:
    backbone = ToyBackbone()
    return Decider(Loaded(backbone, ToyFormat(backbone)), **options), backbone


@pytest.mark.parametrize("chunk", [1, 7, 4096], ids=["token", "partial-chunk", "whole-prefix"])
def test_shared_chunks_match_whole_prompts_and_read_less(chunk: int) -> None:
    fast, fast_model = decider(chunk=chunk)
    whole, whole_model = decider(strategy="reference")
    assert fast.systemone(REQUEST)["answers"] == whole.systemone(REQUEST)["answers"]
    state = len(STATE.encode())
    assert fast_model.tokens_read == state + 3 * 2
    assert whole_model.tokens_read == 3 * (state + 2)


def test_builders_and_systemone_dicts_give_the_same_answers() -> None:
    d, _ = decider()
    result = d.decide(
        state=STATE,
        questions={
            "refund": YesNo("Is a refund asked for?"),
            "team": Choice("Which team?", ["billing", "sales", "support"]),
            "mood": Score("How upset?", ["calm", "annoyed", "furious"]),
        },
    )
    assert result.response["answers"] == d.systemone(REQUEST)["answers"]
    assert list(result.answers) == ["refund", "team", "mood"]
    assert 0.0 <= result["refund"].p_yes <= 1.0
    assert result["team"].top in {"billing", "sales", "support"}
    assert 0.0 <= result["mood"].expected_level <= 2.0
    with pytest.raises(TypeError, match="not a noul"):
        _ = result["team"].p_yes


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ([YesNo("Refund?")], "named questions"),
        ({"refund": "Refund?"}, "refund: a question"),
        ({"refund": None}, "refund: a question"),
        ({"team": Choice("Which?", "cat")}, "team:.*options.*list"),
        ({"mood": Score("How upset?", "calm")}, "mood:.*levels.*list"),
    ]
    + [
        ({"q": builder("Which?", value)}, "q:.*list")
        for builder, value in itertools.product((Choice, Score), NOT_LISTS)
    ]
    + [({"q": Score("Which?", cast(Any, {"low": "Calm", "high": "Upset"}))}, "q:.*list")],  # levels have an order
)
def test_invalid_python_questions_say_what_to_change(questions: Any, message: str) -> None:
    with decider()[0] as model:
        with pytest.raises(InvalidRequest, match=message):
            model.decide(state=STATE, questions=questions)
        with pytest.raises(InvalidRequest, match=message):
            model.decide_many([{"state": STATE, "questions": questions}])


@pytest.mark.parametrize(
    "options",
    [
        {"chunk": 0},
        {"chunk": -1},
        {"chunk": True},
        {"chunk": 1.5},
        {"window": 0},
        {"window": -1},
        {"window": "100"},
        {"prefix_budget": -1},
        {"answer_cache": -1},
        {"answer_cache": 1.5},
        {"strategy": "refernece"},
        {"truncation": "errror"},
        {"keep_awake": -1},
        {"keep_awake": float("nan")},
        {"keep_awake": float("inf")},
    ],
)
def test_invalid_load_options_fail_before_opening_a_model(tmp_path: Path, options: dict[str, Any]) -> None:
    name = next(iter(options))
    with pytest.raises(InvalidRequest, match=name):
        Decider.load(tmp_path / "missing", **options)


def test_a_model_that_hears_no_clips_refuses_a_spoken_question() -> None:
    d, _ = decider()
    clip = base64.b64encode(as_media(np.zeros(1600, np.float32), "audio").data).decode()
    said = {"type": "input_audio", "input_audio": {"data": clip}}
    with pytest.raises(InvalidRequest, match=r"spoken: .* takes text only"):
        d.systemone({"state": STATE, "questions": {"q": {"type": "noul", "instructions": said}}})


def test_the_model_not_ippon_sets_how_many_options_a_question_takes() -> None:
    labels = [f"option {i}" for i in range(300)]
    wide = {"state": STATE, "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": labels}}}
    d, _ = decider()
    assert len(d.systemone(wide)["answers"]["q"]["probabilities"]) == 300
    d.format.most_options = lambda kind: 299  # type: ignore[method-assign]
    with pytest.raises(InvalidRequest, match=r"q: 300 options; .* answers at most 299 in a question"):
        d.systemone(wide)


def test_a_choice_wider_than_a_pass_is_answered_in_rounds_as_its_maker_answers_it() -> None:
    """Torchcast's `_many_options` (torchcast_decision/model.py), ported line by line onto the toy model's asks: near-
    equal groups, the best of each in a final round (in rounds again while wider than a pass), the rest sharing the
    residual. The finalists are asked most likely first within each group, as the creator's code asks them, in rounds
    of rounds here (10 options, 8 finalists, then 6, then 4). The decider's rounds give the same probabilities, and
    answer the request's other questions too."""
    wide = Wide(per_pass=4, group=3, keep=2, residual=1e-3)
    labels = [f"team {i}" for i in range(10)]
    ask = {"type": "choice", "instructions": "Which team?"}

    def maker(keys: list[str]) -> dict[str, float]:
        count = -(-len(keys) // wide.group)
        base, extra = divmod(len(keys), count)
        bounds = [0]
        for g in range(count):
            bounds.append(bounds[-1] + base + (g < extra))
        groups = [keys[a:b] for a, b in itertools.pairwise(bounds)]
        p_one: dict[str, float] = {}
        finalists: list[str] = []
        for ps in asked(groups):  # every group in one batch, as the creator's `probabilities` asks them
            p_one.update(ps)
            finalists += [k for k, _ in sorted(ps.items(), key=lambda kp: -kp[1])[: wide.keep]]
        p_final = maker(finalists) if len(finalists) > wide.per_pass else asked([finalists])[0]
        others = [k for k in keys if k not in set(finalists)]
        others_mass = sum(p_one[k] for k in others) or 1.0
        merged = {k: p_final[k] * (1 - wide.residual) for k in finalists}
        for k in others:
            merged[k] = wide.residual * p_one[k] / others_mass
        total = sum(merged.values())
        return {k: v / total for k, v in merged.items()}

    def asked(groups: list[list[str]]) -> list[dict[str, float]]:
        body = {"state": STATE, "questions": {str(i): {**ask, "criteria": g} for i, g in enumerate(groups)}}
        answers = d.systemone(body)["answers"]
        return [dict(answers[str(i)]["probabilities"]) for i in range(len(groups))]

    d, _ = decider()
    want = maker(labels)
    d.format.wide = wide
    body = {"state": STATE, "questions": {"t": {**ask, "criteria": labels}, "refund": REQUEST["questions"]["refund"]}}
    got = d.systemone(body)["answers"]
    assert got["t"]["probabilities"] == pytest.approx(want, abs=1e-6)
    assert list(got) == ["t", "refund"]
    assert got["refund"]["type"] == "noul"
    # what is unsure names the caller's questions, never the groups a wide choice was asked in
    assert d.systemone({**body, "unsure_below": 1})["unsure"] == ["t", "refund"]


def test_a_window_past_the_creators_is_taken_and_marked_unverified() -> None:
    backbone = ToyBackbone()
    with pytest.raises(UnsupportedModel, match="sets no window"):
        Decider(Loaded(backbone, ToyFormat(backbone)), window=4096)
    fmt = ToyFormat(backbone)
    fmt.window = 1024
    with pytest.warns(UserWarning, match="past its creator's 1,024: answers on longer prompts are unverified"):
        d = Decider(Loaded(backbone, fmt), window=4096)
    assert fmt.window == 4096
    assert d.details["window"] == {"creator": 1024, "set": 4096, "verified": False}
    fmt.window = 30  # a prompt past the window is refused after rendering, before anything is read
    read = backbone.tokens_read
    with pytest.raises(ContextTooLong, match=r"refund: a prompt of \d+ tokens, past the 30"):
        d.systemone(REQUEST)
    assert backbone.tokens_read == read


def test_a_model_no_adapter_supports(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "something_else"}))
    with pytest.raises(UnsupportedModel, match="no installed adapter matches"):
        select(ModelDir.open(tmp_path))
    with pytest.raises(UnsupportedModel, match="no adapter named"):
        select(ModelDir.open(tmp_path), "nobody/nothing")


def test_many_requests_give_each_requests_own_answers() -> None:
    other = {"state": "A different ticket entirely.", "questions": {"refund": {"type": "noul", "instructions": "?"}}}
    fewer = {"state": STATE, "questions": {"refund": REQUEST["questions"]["refund"]}}
    bodies = [REQUEST, other, fewer, REQUEST]
    alone = [decider(prefix_budget=0)[0].systemone(b) for b in bodies]
    for options in ({}, {"prefix_budget": 0}, {"strategy": "reference"}):
        assert decider(**options)[0].systemone_many(bodies) == alone
    many = decider()[0].decide_many([{"state": STATE, "questions": {"refund": YesNo("Is a refund asked for?")}}])
    assert many[0]["refund"].p_yes == pytest.approx(alone[2]["answers"]["refund"]["noul"], abs=1e-4)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("ticket", "request must be a mapping"),
        ({"state": STATE}, "named questions"),
        ({"staet": STATE, "questions": REQUEST["questions"]}, "unknown field 'staet'"),
        ({**REQUEST, "unsure_below": 2}, "unsure_below"),
    ],
)
def test_invalid_batches_name_the_request_before_reading_any(bad: Any, message: str) -> None:
    model, backbone = decider()
    with model, pytest.raises(InvalidRequest, match=rf"requests\[1\]: .*{message}"):
        model.decide_many([REQUEST, bad])
    assert backbone.tokens_read == 0


def test_requests_about_the_same_state_read_it_once() -> None:
    d, backbone = decider(prefix_budget=0)  # no kept prefixes: the sharing is within the batch
    d.systemone_many([REQUEST, REQUEST, REQUEST])
    assert backbone.tokens_read == len(STATE.encode()) + 3 * 3 * 2


def test_a_non_causal_model_answers_many_requests_as_it_answers_each() -> None:
    backbone = ToyBackbone()
    backbone.causal = False
    d = Decider(Loaded(backbone, ToyFormat(backbone)))
    other = {"state": "Another state.", "questions": {"x": {"type": "noul", "instructions": "?"}}}
    assert d.systemone_many([REQUEST, other]) == [d.systemone(REQUEST), d.systemone(other)]


def test_the_warm_up_reads_every_question_type_the_model_takes_and_keeps_nothing() -> None:
    class YesNoOnly(ToyFormat):
        def render(self, request: Any) -> Any:
            if any(q.type != "noul" for q in request.questions):
                raise InvalidRequest("yes/no questions only")
            return super().render(request)

    backbone = ToyBackbone()
    d = Decider(Loaded(backbone, YesNoOnly(backbone)), prefix_budget=1 << 20)
    d.warm_up()
    assert backbone.tokens_read > 0
    assert d.prefixes is not None
    assert len(d.prefixes) == 0
    assert d.prefixes.stats.misses == 0


def test_lone_prompts_about_different_states_share_one_batched_pass() -> None:
    bodies = [{"state": s, "questions": {"q": {"type": "noul", "instructions": "?"}}} for s in ("one", "two", "three")]
    alone = [decider(prefix_budget=0)[0].systemone(b) for b in bodies]
    d, backbone = decider(prefix_budget=1 << 20)
    assert d.systemone_many(bodies) == alone
    assert backbone.batches == 1
    d.systemone(bodies[0] | {"questions": {"q": {"type": "noul", "instructions": "!"}}})  # its state is kept now
    backbone.batches = 0
    assert d.systemone_many(bodies) == alone
    assert backbone.batches == 1  # the other two, batched; the kept one resumed from the store
    assert d.prefixes is not None
    assert d.prefixes.stats.hits >= 1


def test_prompts_longer_than_a_chunk_are_not_batched_across_requests() -> None:
    """A batch reads its prompts whole; ones longer than a chunk are read chunk by chunk, as alone."""
    bodies = [{"state": s * 40, "questions": {"q": {"type": "noul", "instructions": "?"}}} for s in ("one", "two")]
    alone = [decider(prefix_budget=0, chunk=64)[0].systemone(b) for b in bodies]
    d, backbone = decider(prefix_budget=0, chunk=64)
    assert d.systemone_many(bodies) == alone
    assert backbone.batches == 0


def test_an_answer_that_overflows_fp16_is_answered_again_in_bf16() -> None:
    """The bf16 build answers from then on, with its kept readings and fingerprint, and the result says why; with
    no fallback (fp16 asked for) the error stands."""
    import mlx.core as mx

    class Overflowing(ToyFormat):
        def scores(self, rendering: Any, hidden: Any) -> dict[str, mx.array]:
            return {k: v * mx.array(float("nan")) for k, v in super().scores(rendering, hidden).items()}

    backbone = ToyBackbone()
    d = Decider(Loaded(backbone, Overflowing(backbone)), prefix_budget=1 << 20)
    d.details["precision"] = "fp16"
    bf16, _ = decider(prefix_budget=1 << 20)
    d.fallback = lambda: bf16
    with pytest.warns(UserWarning, match="overflowed fp16"):
        (result,) = d.decide_many([REQUEST])
    assert result.response["answers"] == decider()[0].systemone(REQUEST)["answers"]
    assert result.diagnostics["precision"] == "bf16"  # the result says which path answered, and why
    assert "overflowed fp16" in (result.fallback or "")
    assert d.precision == "bf16"
    assert d.fallback is None
    assert d.prefixes is bf16.prefixes  # kept readings are the bf16 build's from now on
    plain = Decider(Loaded(backbone, Overflowing(backbone)))  # no fallback (fp16 asked for): the error stands
    with pytest.raises(NonFiniteAnswer, match="finite"):
        plain.systemone(REQUEST)
    failing = Decider(Loaded(backbone, Overflowing(backbone)))  # the bf16 build can't load: closed, never half-built

    def no_room() -> Decider:
        raise InsufficientMemory("no room for bf16")

    failing.fallback = no_room
    with pytest.warns(UserWarning, match="overflowed fp16"), pytest.raises(InsufficientMemory):
        failing.systemone(REQUEST)
    assert failing.closed
    failing.close()  # closing again does nothing
    with pytest.raises(DecisionError, match="closed"):
        failing.systemone(REQUEST)


def test_after_the_bf16_fallback_kept_prefixes_still_fit_the_room_for_readings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bf16 build that takes over from an overflowing fp16 one holds a kept reading past its budget to half the
    room this Mac's GPU has for readings, as the fp16 build did, never without a bound."""
    import sys

    import mlx.core as mx

    from test_generic_adapters import LABELS, qwen3_classifier

    qwen3_classifier(tmp_path)
    # auto: fp16, with bf16 to fall back to; the same request is read again, not answered from the kept answers
    d = Decider.load(tmp_path, keep_awake=None, pin_memory=False, answer_cache=0)
    request = {"state": "w1 w2", "questions": {"team": {"type": "choice", "criteria": list(LABELS.values())}}}
    d.systemone(request)
    assert d.fallback is not None
    assert d.prefixes is not None
    assert d.profile.device is not None
    cap = d.prefixes.most_alone
    assert cap < sys.maxsize
    overflowing, scores = d.format, type(d.format).scores

    def overflow(self: Any, rendering: Any, hidden: Any) -> Any:
        got = scores(self, rendering, hidden)
        return {k: v * mx.array(float("nan")) for k, v in got.items()} if self is overflowing else got

    monkeypatch.setattr(type(d.format), "scores", overflow)
    with pytest.warns(UserWarning, match="overflowed fp16"):
        d.systemone(request)
    assert d.precision == "bf16"
    assert d.prefixes is not None
    assert d.prefixes.most_alone == cap


def test_closing_a_decider_unloads_it_and_later_calls_say_so() -> None:
    import gc
    import weakref

    d, backbone = decider(prefix_budget=1 << 20)
    d.systemone(REQUEST)
    gone = weakref.ref(backbone)
    backbone.compiled = [backbone]  # type: ignore[attr-defined]  # a cycle, as its compiled functions make
    del backbone
    gc.collect()
    gc.freeze()  # as Decider.load leaves what loading made
    try:
        with d:
            pass  # leaving the block closes it
        assert gone() is None  # the weights are released by closing alone, frozen cycle included
    finally:
        gc.unfreeze()
    assert d.closed
    assert d.prefixes is None
    with pytest.raises(DecisionError, match="closed"):
        d.systemone(REQUEST)
    d.close()  # again: nothing


def test_unsure_answers_are_listed_for_sending_on() -> None:
    """A request with unsure_below lists the questions whose most likely option is under it, as Result.unsure does,
    so a caller sends only those on to a person or a larger model; without it the response has no such list."""
    d, _ = decider()
    questions: dict[str, Any] = {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "sales", "other"]},
    }
    plain = d.systemone({"state": "I was charged twice.", "questions": questions})
    assert "unsure" not in plain
    result = d.decide("I was charged twice.", questions)
    sure = {q: a.confidence for q, a in result.answers.items()}
    assert sure == {q: max(a.probabilities) for q, a in result.answers.items()}
    cut = sorted(sure.values())[0] + 1e-9  # just above the least sure: exactly that one is unsure
    asked = d.systemone({"state": "I was charged twice.", "questions": questions, "unsure_below": cut})
    assert asked["unsure"] == result.unsure(cut) == [min(sure, key=sure.__getitem__)]
    batched = d.decide_many([{"state": "I was charged twice.", "questions": questions, "unsure_below": cut}])
    assert batched[0].response == asked
    assert d.systemone({"state": "I was charged twice.", "questions": questions, "unsure_below": 1})["unsure"] == list(
        questions
    )


def test_an_identical_request_is_answered_again_without_reading_it() -> None:
    d, model = decider(answer_cache=1024)  # off by default: every request read
    first = d.systemone(REQUEST)
    read = model.tokens_read
    again = d.systemone(REQUEST)
    assert again == first
    assert model.tokens_read == read  # the kept answer: nothing read
    again["answers"]["refund"]["noul"] = -1.0  # the caller's copy: the kept answer stays as it was
    assert d.systemone(REQUEST) == first
    d.systemone({**REQUEST, "questions": {"refund": REQUEST["questions"]["refund"]}})  # other questions: read
    assert model.tokens_read > read
    # a classifier's yes/no instructions pick which label answers, with the same tokens read: they key the answer too
    rendering = d.render(parse_request(REQUEST))
    asked = dataclasses.replace(
        rendering.request,
        questions=tuple(
            dataclasses.replace(q, instructions="Is this about sales?") for q in rendering.request.questions
        ),
    )
    assert Answers.key(rendering, d.chunk, d.strategy) != Answers.key(
        dataclasses.replace(rendering, request=asked), d.chunk, d.strategy
    )
    off, off_model = decider()
    off.systemone(REQUEST)
    read = off_model.tokens_read
    off.systemone(REQUEST)
    assert off_model.tokens_read > read


def test_a_reading_nobody_waits_for_stops_before_its_next_pass() -> None:
    """A server's request past its deadline (or whose caller hung up) stops on the GPU at the next pass, never runs to
    its end ahead of the next request; with someone waiting, the same request is answered."""
    from ippon.errors import Abandoned
    from ippon.runtime import STOP

    d, _ = decider()
    token = STOP.set(lambda: Abandoned("nobody waits"))
    try:
        with pytest.raises(Abandoned, match="nobody waits"):
            d.systemone(REQUEST)
    finally:
        STOP.reset(token)
    assert d.systemone(REQUEST)["answers"]


def test_a_prompt_whose_reading_this_mac_cannot_hold_is_refused_before_it_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside the window is not enough: a prompt whose keys and values don't fit the GPU's working set beside the
    model's weights and kept prefixes is refused (context_too_long), before any pass. And the largest model admission
    lets onto a Mac still has a reading's room beside it: an 8 GB Mac admitted Kettei Core at 8 bits, then held 1 of
    its tokens and refused every request."""
    import dataclasses
    import types

    import mlx.core as mx

    from ippon.admission import MIN_READING, READING_MARGIN, budget
    from ippon.arch.cache import Attention
    from ippon.device import Device
    from ippon.errors import ContextTooLong
    from ippon.profiles import resolve

    layer = Attention.empty(1, 1, 64, mx.float16)  # 256 bytes of keys and values a token
    plain = ToyBackbone.new_cache
    monkeypatch.setattr(
        ToyBackbone, "new_cache", lambda self, capacity, start=None: types.SimpleNamespace(layers=[layer])
    )
    kept = 1 << 20
    mac = Device("Apple M1", "applegpu_g13g", 8 << 30, (1 << 30) + kept + READING_MARGIN + 256 * 40, (26,), True)
    d, backbone = decider(prefix_budget=kept, profile=dataclasses.replace(resolve(), device=mac), resident=1 << 30)
    monkeypatch.setattr(ToyBackbone, "new_cache", plain)
    assert d.tokens_here == 40
    read = backbone.tokens_read
    with pytest.raises(ContextTooLong, match="about 40 of this model's tokens"):
        d.systemone({**REQUEST, "state": "x" * 200})
    assert backbone.tokens_read == read  # refused before anything was read
    gpu = 5_726_623_744  # what macOS gives an 8 GB Mac's GPU, 5.33 GiB
    eight = Device("Apple M1", "applegpu_g13g", 8 << 30, gpu, (26,), True)
    monkeypatch.setattr(
        ToyBackbone, "new_cache", lambda self, capacity, start=None: types.SimpleNamespace(layers=[layer])
    )
    largest, _ = decider(profile=dataclasses.replace(resolve(), device=eight), resident=budget(gpu))
    monkeypatch.setattr(ToyBackbone, "new_cache", plain)
    assert largest.tokens_here is not None
    assert largest.tokens_here * 256 >= MIN_READING  # with the default kept prefixes, as a load makes them
