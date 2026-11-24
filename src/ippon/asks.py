"""The questions ippon asks a model on its own (warm-up, bench, compare): the standard sets, or the model's
own when its adapter says what it can be asked (`bench_questions(model, count, variant)`: a classifier's labels, a
label-token model's answers, a format that takes only yes/no questions).

    ask = asker(adapter, model)   # (count, variant) -> System One questions
    ask(3, 0)
    warm_up(own(adapter, model))  # the warm-up requests, in the order they are tried
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapter import Adapter
    from .model_dir import ModelDir

Asker = Callable[[int, int], dict[str, dict[str, Any]]]  # (count, variant) -> questions a model can answer

# A short request of every question type, which loads the weights in, compiles the kernels requests use and measures
# the activations auto precision chooses fp16 or bf16 by (`Decider.warm_up`)
WARM_UP: dict[str, Any] = {
    "state": "A short state to warm the model up with.",
    "questions": {
        "yes": {"type": "noul", "instructions": "Is this a warm-up?"},
        "pick": {"type": "choice", "instructions": "Which one?", "criteria": {"a": "The first", "b": "The second"}},
        "rate": {"type": "score", "instructions": "How warm is it?", "criteria": ["cold", "warm", "hot"]},
    },
}

QUESTIONS: tuple[dict[str, Any], ...] = (
    {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": ["billing", "shipping", "support"],
    },
    {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]},
    {"type": "noul", "instructions": "Is a deadline mentioned?"},
    {
        "type": "choice",
        "instructions": "What is the main topic?",
        "criteria": ["payment", "delivery", "account", "other"],
    },
)


def standard(count: int, variant: int = 0) -> dict[str, dict[str, Any]]:
    """The first `count` standard questions; another `variant` asks them about another case, so their tokens
    differ."""
    out = {}
    for i in range(count):
        q = dict(QUESTIONS[(i + variant) % len(QUESTIONS)])
        q["instructions"] = instructions(i, variant)
        out[f"q{i}"] = q
    return out


def yes_no(count: int, variant: int = 0) -> dict[str, dict[str, Any]]:
    """The first `count` standard questions' instructions as yes/no questions, for a model that is asked those."""
    return {f"q{i}": {"type": "noul", "instructions": instructions(i, variant)} for i in range(count)}


def instructions(i: int, variant: int = 0) -> str:
    """The i-th standard question's instructions, of another case for another `variant`."""
    text = str(QUESTIONS[(i + variant) % len(QUESTIONS)]["instructions"])
    return f"{text} (case {variant})" if variant else text


def own(adapter: Adapter, model: ModelDir) -> Asker | None:
    """The questions `model` can be asked, its adapter's `bench_questions`, or None for a model that is asked any
    System One question (an adapter whose `bench_questions` gives None)."""
    if adapter.bench_questions(model, 1) is None:
        return None

    def ask(count: int, variant: int = 0) -> dict[str, dict[str, Any]]:
        return dict(adapter.bench_questions(model, count, variant) or {})

    return ask


def asker(adapter: Adapter, model: ModelDir) -> Asker:
    """The questions `model` can be asked: its adapter's own, else the standard ones."""
    return own(adapter, model) or standard


def warm_up(asked: Asker | None) -> list[dict[str, Any]]:
    """The warm-up requests, in the order a decider tries them: one of every question type together (WARM_UP), then
    each of them alone, for a model that refuses one; or, for a model asked its own questions (`own`), two of
    those."""
    if asked is not None:
        return [{"state": WARM_UP["state"], "questions": asked(2, 0)}]
    alone = ({"state": WARM_UP["state"], "questions": {key: q}} for key, q in WARM_UP["questions"].items())
    return [WARM_UP, *alone]
