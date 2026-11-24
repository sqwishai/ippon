"""Many requests answered through the batched path, each one's error its own: what `serve`'s model thread,
`ippon decide --states` and `ippon eval` answer with.

Each request is rendered alone, so one that a model can't be asked (InvalidRequest, ContextTooLong) is that request's
error and no other's; the rest are read together in shared passes (`Decider.answer_rendered`). Should a batch's pass
fail, its requests are answered again one by one, so that the failure stays with the request that caused it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Any

from .adapter import Rendering
from .decider import Decider
from .errors import DecisionError, as_decision_error
from .policy import MAX_BATCH
from .schema import Request, Result, parse_request


def answer_each(decider: Decider, requests: Sequence[Request]) -> list[Result | Exception]:
    """The answer to each request, or the exception it raised, in order: the requests read together, and one by one
    should that pass fail (one request alone: its error). Each answer carries the milliseconds its own rendering took
    and the pass that scored it (`Result.timing_ms`)."""
    out: list[Result | Exception | None] = [None] * len(requests)
    rendered: list[tuple[int, Rendering, float]] = []
    for i, request in enumerate(requests):
        started = time.perf_counter()
        try:
            rendered.append((i, decider.render(request), time.perf_counter() - started))
        except Exception as error:
            out[i] = error
    try:
        if rendered:
            started = time.perf_counter()
            answered = decider.answer_rendered([r for _, r, _ in rendered])
            score = time.perf_counter() - started
            for (i, _, render), result in zip(rendered, answered, strict=True):
                out[i] = result.timed(render, score)
    except Exception as error:
        if len(rendered) == 1:
            out[rendered[0][0]] = error
        else:
            for i, rendering, render in rendered:
                started = time.perf_counter()
                try:
                    out[i] = decider.answer_rendered([rendering])[0].timed(render, time.perf_counter() - started)
                except Exception as failed:
                    out[i] = failed
    return [r for r in out if r is not None]


VIDEOS_TOGETHER = 4  # videos read in one batch at most (`answer_all`): d3-mini's 36 clips peaked at 13 GB


def answer_all(
    decider: Decider, bodies: Sequence[Any], batch: int | None = None
) -> Iterator[tuple[int, Result | DecisionError]]:
    """(index, the answer to `bodies[index]` or its request's error as a DecisionError: `errors.as_decision_error`),
    in order, `batch` requests (default MAX_BATCH) read together at most, and at most VIDEOS_TOGETHER videos: a batch
    holds every video's decoded frames at once (36 short clips at the default batch passed 31 GB on d3-mini)."""
    size = batch or MAX_BATCH
    start = 0
    while start < len(bodies):
        out: dict[int, Result | DecisionError] = {}
        parsed: list[tuple[int, Request]] = []
        videos, i = 0, start
        while i < len(bodies) and i - start < size and (videos < VIDEOS_TOGETHER or not parsed):
            try:
                request = parse_request(bodies[i])
                parsed.append((i, request))
                videos += len(request.videos)
            except DecisionError as error:
                out[i] = error
            i += 1
        start = i
        answered = answer_each(decider, [request for _, request in parsed])
        for (i, _), got in zip(parsed, answered, strict=True):
            out[i] = as_decision_error(got) if isinstance(got, Exception) else got
        yield from sorted(out.items())
