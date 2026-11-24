"""Whether a model's answers depend on how ippon runs it: `ippon verify MODEL --invariance`.

Every request is answered the plain way (alone, in the default chunks, nothing kept from before) and then:

- **chunks**: in chunks of 512 and of 2,048 tokens;
- **whole**: every prompt read whole, one after another (no shared prefix, no branches: the reference path);
- **kept**: right after the same request, its state resumed from a kept prefix;
- **grown**: right after a request about the state without its last part, resumed from that prefix (an agent's
  next turn; text states only);
- **order**: with its questions in the reverse order, where the format reads each question from a prompt of its
  own (where questions share a prompt, their order is part of it);
- **together**: answered in one call with a request about another state and one about the same state.

For each, the largest difference of any probability from the plain answers and the top answers that changed (near
ties aside, as `verify` counts them), against a tolerance: a reference's bound, else 0.01. They are not exactly
equal in fp16 or bf16: another chunking or batch adds and rounds in another order, and an M5's fused recurrence
rounds about as TF32 does; the tolerance says how far that may go.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._portable import record
from .decider import Decider
from .errors import DecisionError
from .schema import Result, parse_request
from .verify import BOUND, Comparison, compare

CHUNKS = (512, 2048)


@dataclass
class Check:
    name: str
    worst: float = 0.0
    changed: list[str] = field(default_factory=list)  # requests whose top answer changed
    near_ties: int = 0
    requests: int = 0

    def add(self, request: str, c: Comparison) -> None:
        self.requests += 1
        self.worst = max(self.worst, c.max_abs_difference)
        self.near_ties += c.near_ties
        if not c.top_agrees:
            self.changed.append(request)

    def passed(self, tolerance: float) -> bool:
        return self.worst <= tolerance and not self.changed


@dataclass(frozen=True)
class Report:
    model: str
    tolerance: float
    checks: tuple[Check, ...]
    skipped: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed(self.tolerance) for c in self.checks)

    def text(self) -> str:
        lines = [f"{self.model}: invariance, tolerance {self.tolerance}"]
        for c in self.checks:
            mark = "ok" if c.passed(self.tolerance) else "FAILS"
            extra = f"; top answer changed: {', '.join(c.changed)}" if c.changed else ""
            ties = f"; {c.near_ties} near ties" if c.near_ties else ""
            lines.append(f"  {c.name:9} {c.requests:3} requests, worst {c.worst:.5f} {mark}{extra}{ties}")
        if self.skipped:
            lines.append(f"  skipped (the model doesn't take them): {', '.join(self.skipped)}")
        lines.append("passed" if self.passed else "failed")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return record(
            {
                "model": self.model,
                "tolerance": self.tolerance,
                "passed": self.passed,
                "skipped": list(self.skipped),
                "checks": [
                    {
                        "name": c.name,
                        "requests": c.requests,
                        "worst": c.worst,
                        "top_changed": c.changed,
                        "near_ties": c.near_ties,
                        "passed": c.passed(self.tolerance),
                    }
                    for c in self.checks
                ],
            }
        )


def probabilities(result: Result) -> dict[str, list[float]]:
    """Per question, its probabilities in its options' order."""
    return {q: list(a.probabilities) for q, a in result.answers.items()}


def prompts_by_question(decider: Decider, body: Mapping[str, Any]) -> dict[str, set[tuple[int, ...]]]:
    """The prompts each question is read from: equal for two orders of the questions unless the format renders a
    question by its place (several questions in one prompt): then their order is part of the prompt."""
    out: dict[str, set[tuple[int, ...]]] = {}
    for prompt in decider.render(parse_request(body)).prompts:
        for r in prompt.readouts:
            out.setdefault(r.question, set()).add(prompt.tokens)
    return out


def shortened(body: Mapping[str, Any]) -> dict[str, Any] | None:
    """The request about its state without the last fifth (cut at a space), or None for a short or non-text one."""
    state = body.get("state")
    if not isinstance(state, str) or len(state) < 200:
        return None
    cut = state.rfind(" ", 0, len(state) * 4 // 5)
    return {**copy.deepcopy(dict(body)), "state": state[: max(cut, 1)]}


def run(decider: Decider, requests: Sequence[Mapping[str, Any]], tolerance: float = BOUND, model: str = "") -> Report:
    """Every invariance check over `requests` ([{"name", "request"}, ...], as reference files hold them)."""
    checks = {n: Check(n) for n in ("chunks", "whole", "kept", "grown", "order", "together")}

    def answer(*bodies: Mapping[str, Any], keep: bool = False) -> list[dict[str, list[float]]]:
        if decider.prefixes is not None and not keep:
            decider.prefixes.clear()
        rendered = [decider.render(parse_request(b)) for b in bodies]
        with decider.uncached():  # every check reads its request: a kept answer would agree with itself
            return [probabilities(r) for r in decider.answer_rendered(rendered)]

    def against(check: str, name: str, base: dict[str, list[float]], got: dict[str, list[float]]) -> None:
        ids = list(base)
        checks[check].add(name, compare([got[q] for q in ids], [base[q] for q in ids], tie=tolerance))

    bases, skipped = {}, []
    for case in requests:
        try:
            bases[case["name"]] = answer(case["request"])[0]
        except DecisionError:
            skipped.append(case["name"])
    taken = [case for case in requests if case["name"] in bases]
    plain, strategy = decider.chunk, decider.strategy
    for i, case in enumerate(taken):
        name, body, base = case["name"], case["request"], bases[case["name"]]
        try:
            for size in CHUNKS:
                decider.chunk = size
                against("chunks", name, base, answer(body)[0])
            decider.chunk, decider.strategy = plain, "reference"
            against("whole", name, base, answer(body)[0])
        finally:
            decider.chunk, decider.strategy = plain, strategy
        if decider.prefixes is not None:
            answer(body)
            against("kept", name, base, answer(body, keep=True)[0])
            earlier = shortened(body)
            if earlier is not None:
                answer(earlier)
                against("grown", name, base, answer(body, keep=True)[0])
        reverse = {**body, "questions": dict(reversed(list(body["questions"].items())))}
        if len(body["questions"]) > 1 and prompts_by_question(decider, body) == prompts_by_question(decider, reverse):
            against("order", name, base, answer(reverse)[0])
        other = next(
            (o["request"] for o in taken[i + 1 :] + taken[:i] if o["request"].get("state") != body.get("state")), None
        )
        same = {**body, "questions": dict(list(body["questions"].items())[:1])}
        together = answer(*([other] if other is not None else []), body, same)
        against("together", name, base, together[1 if other is not None else 0])
    return Report(model, tolerance, tuple(c for c in checks.values() if c.requests), tuple(skipped))
