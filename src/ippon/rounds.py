"""A choice wider than a model reads in one pass, answered in rounds where its creator answers it so: the options
split into groups of at most `group` (`Wide.split`; by default near-equal, the first ones an option larger), each group
asked in the request's own pass, then the best `keep` of every group asked in a final round, most likely first within
each group (in rounds again while they are more than one pass takes), and the two rounds' probabilities put together
(`Wide.combine`; by default the finalists share 1 - `residual` of the probability as the final round gives it, and the
others share `residual` in proportion to their first-round probabilities). A creator that splits or combines otherwise
passes its own functions.

A format that answers this way names its `Wide` (`FormatBase.wide`); the decider renders the first round with the
request's other questions (`first_round`), and `answered` reads it, then the final round as a request of its own, and
puts the caller's answers together from both.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar

from .schema import Answer, Question, Request, Result

SEPARATOR = "\x1f"  # between a wide question's id and its group's number: no caller's id holds it


def group_id(qid: str, group: int) -> str:
    """The id a wide question's group is asked under in the first round."""
    return f"{qid}{SEPARATOR}{group}"


def near_equal(options: Sequence[str], group: int) -> tuple[tuple[str, ...], ...]:
    """Near-equal groups of at most `group` options, in order, the first ones one option larger."""
    count = -(-len(options) // group)
    base, extra = divmod(len(options), count)
    out, at = [], 0
    for g in range(count):
        size = base + (g < extra)
        out.append(tuple(options[at : at + size]))
        at += size
    return tuple(out)


def with_residual(
    q: Question,
    groups: Sequence[Sequence[str]],
    first: Sequence[Sequence[float]],
    final: Mapping[str, float],
    wide: Wide,
) -> list[float]:
    """The finalists' final probabilities scaled to 1 - `wide.residual`, the others sharing `residual` as their
    first-round probabilities do, normalized, in the question's options' order."""
    residual = wide.residual
    one = {k: p for g, ps in zip(groups, first, strict=True) for k, p in zip(g, ps, strict=True)}
    others = [k for k in q.options if k not in final]
    mass = sum(one[k] for k in others) or 1.0
    out = {k: p * (1 - residual) for k, p in final.items()} | {k: residual * one[k] / mass for k in others}
    total = sum(out.values())
    return [out[k] / total for k in q.options]


Split = Callable[[Sequence[str], int], tuple[tuple[str, ...], ...]]  # options and the most in a group -> groups
# the question, its groups, their first-round probabilities, the finalists' final ones and the Wide -> probabilities
Combine = Callable[
    [Question, Sequence[Sequence[str]], Sequence[Sequence[float]], Mapping[str, float], "Wide"], list[float]
]


@dataclass(frozen=True)
class Wide:
    """How a model's creator answers a choice of more than `per_pass` options: in groups of at most `group`, as
    `split` makes them, the best `keep` of each asked again, and the rounds put together by `combine`."""

    per_pass: int
    group: int
    keep: int
    residual: float
    split: Split = near_equal
    combine: Combine = with_residual


@dataclass(frozen=True)
class Round:
    """A request whose wide choices are asked in groups: `request` is the caller's, `groups` each wide question's
    groups of options, in order."""

    request: Request
    groups: Mapping[str, tuple[tuple[str, ...], ...]]


def first_round(request: Request, wide: Wide) -> tuple[Request, Round] | None:
    """The request with each choice wider than one pass replaced by its groups; None when it has none."""
    asked: list[Question] = []
    groups: dict[str, tuple[tuple[str, ...], ...]] = {}
    for q in request.questions:
        if q.type != "choice" or len(q.options) <= wide.per_pass:
            asked.append(q)
            continue
        groups[q.id] = wide.split(q.options, wide.group)
        asked += [subset(q, group_id(q.id, i), g) for i, g in enumerate(groups[q.id])]
    if not groups:
        return None
    return replace(request, questions=tuple(asked)), Round(request, groups)


def subset(q: Question, qid: str, options: Sequence[str]) -> Question:
    """`q` asked under `qid` of `options` alone, in their order: a group in the question's, the finalists most likely
    first within each group, as Torchcast's `_many_options` asks them (`only(finalists)`)."""
    described = dict(q.criteria)
    return replace(q, id=qid, criteria=tuple((k, described[k]) for k in options))


def finalists(round_: Round, probabilities: Mapping[str, Sequence[float]], wide: Wide) -> dict[str, list[str]]:
    """Each wide question's finalists: the best `keep` of each group, most likely first, group by group."""
    out: dict[str, list[str]] = {}
    for qid, groups in round_.groups.items():
        out[qid] = []
        for i, g in enumerate(groups):
            ranked = sorted(zip(g, probabilities[group_id(qid, i)], strict=True), key=lambda kp: -kp[1])
            out[qid] += [k for k, _ in ranked[: wide.keep]]
    return out


class Rendered(Protocol):
    """A rendered request (adapter.Rendering), as far as its rounds go: the first round of its wide choices, if it is
    one."""

    @property
    def round(self) -> Round | None: ...


R = TypeVar("R", bound=Rendered)
Shape = Callable[[Question, Sequence[float]], tuple[tuple[float, ...], dict[str, Any]]]  # Format.answer


def answered(
    renderings: Sequence[R],
    read: Callable[[Sequence[R]], list[Result]],
    render: Callable[[Request], R],
    wide: Wide | None,
    shape: Shape,
) -> list[Result]:
    """The answers to `renderings` (`read` reads them), each one that is a first round finished with its final round:
    all such requests' final rounds (`render` renders them) read together, in rounds again while they are wider than a
    pass. `shape` is the format's answer from a question's probabilities."""
    results = read(renderings)
    waiting = [i for i, r in enumerate(renderings) if r.round is not None]
    if not waiting:
        return results
    assert wide is not None
    finals = []
    for i in waiting:
        round_ = renderings[i].round
        assert round_ is not None
        got = {qid: answer.probabilities for qid, answer in results[i].answers.items()}
        picks = finalists(round_, got, wide)
        asked = tuple(subset(q, q.id, picks[q.id]) for q in round_.request.questions if q.id in picks)
        finals.append(render(replace(round_.request, questions=asked)))
    for i, final in zip(waiting, answered(finals, read, render, wide, shape), strict=True):
        round_ = renderings[i].round
        assert round_ is not None
        results[i] = finished(round_, results[i], final, wide, shape)
    return results


def finished(round_: Round, first: Result, final: Result, wide: Wide, shape: Shape) -> Result:
    """The caller's request answered: its wide choices from their two rounds (`merged`), its other questions from the
    first; usage and cuts of both rounds."""
    answers: dict[str, Answer] = {}
    shaped: dict[str, Any] = {}
    for q in round_.request.questions:
        if q.id not in round_.groups:
            answers[q.id], shaped[q.id] = first.answers[q.id], first.response["answers"][q.id]
            continue
        groups = round_.groups[q.id]
        ones = [first.answers[group_id(q.id, i)].probabilities for i in range(len(groups))]
        p = wide.combine(q, groups, ones, final.answers[q.id].by_option, wide)
        probabilities, shaped[q.id] = shape(q, p)
        answers[q.id] = Answer(q, probabilities)
    usage = {k: v + final.response["usage"].get(k, 0) for k, v in first.response["usage"].items()}
    response = {**first.response, "answers": shaped, "usage": usage}
    if cuts := [*first.response.get("truncated", ()), *final.response.get("truncated", ())]:
        response["truncated"] = cuts
    merged = replace(first, answers=answers, response=response)
    if round_.request.unsure_below is not None:  # the first round's list names its group questions, not the caller's
        response["unsure"] = merged.unsure(round_.request.unsure_below)
    return merged
