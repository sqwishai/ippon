"""Fastino's GLiDE decision models (2B, 9B), one prompt per question, as the creator's `glide` package
(prompt.py, model.py, encoder.py, recurrent.py, shipped in each model repository) builds and reads it:

    <|im_start|>user
    Context:
    <state>

    Question: <instructions>

    Options:
    (A) key: description
    (B) key

    Think it through briefly, then answer with the option letter only.<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

    Answer: (

(decision_config.json "prompt_format": "quick-empty-think-medium-v1", Qwen's chat template at reasoning effort medium,
which adds no system text). A state that is not text is written as JSON, instructions that are an object as
"key: value" lines. Options as the creator shows them: a choice's keys with their descriptions (the key alone when the
description is the key or absent), a score's levels "0" to "n-1", a noul's true then false ("Yes" and "No" without
descriptions). The 255-code readout (readout.safetensors) reads the final "(": a question of n options takes the first
n codes' logits, divided by decision_config.json's temperature, softmaxed. A prompt of more than 40,000 tokens less 8
is refused, with its request.

Both are Qwen3.5 whose full attention reads the whole prompt both ways (the Gated DeltaNet layers stay causal). The 9B
reads two blocks of middle layers twice (decision_config.json "midloop", `arch.common.Recurrence`).

GLiDE reads text only here. A choice given as a list of labels is accepted (the creator's code takes only a map) and
shown as the map of those labels without descriptions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .. import asks
from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import qwen35
from ..arch.common import Recurrence
from ..errors import UnsupportedModel
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout
from ..readout import CODE_ROWS, CodeReadout
from ..schema import Question, QuestionType, Request, as_text, systemone_answer
from ..tokenizer import Tokenizer
from .qwen import ANSWER_TURN

PROMPT_FORMAT = "quick-empty-think-medium-v1"
USER_TURN = "<|im_start|>user\n"
ANSWER_CUE = "Answer: ("
BODY_TAIL = "\n\nThink it through briefly, then answer with the option letter only."
MAX_TOKENS = 40_000 - 8  # model.py: a prompt of more than DEFAULT_MAX_LEN less 8 tokens is refused


def instructions_text(value: Any) -> str:
    """The creator's `as_text` for instructions: text as it is, an object as "key: value" lines, else JSON; none as
    nothing."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return "\n".join(
            f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}" for k, v in value.items()
        )
    return as_text(value)


def options(question: Question) -> list[tuple[str, Any]]:
    """(key, description) in prompt order, as the creator's `options_for` writes them."""
    if question.type == "noul":
        given = dict(question.criteria)
        return [("true", given.get("true") or "Yes"), ("false", given.get("false") or "No")]
    if question.type == "score":
        return [(key, str(level)) for key, level in question.criteria]
    return [(key, key if value is None or value == "" else as_text(value)) for key, value in question.criteria]


def margin(probabilities: Sequence[float]) -> float:
    """The creator's confidence for a choice and a score: the most likely option's probability less the next one's."""
    ranked = sorted(probabilities, reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]


@dataclass
class GlideFormat(FormatBase):
    tokenizer: Tokenizer
    readout: CodeReadout
    window: int | None = MAX_TOKENS

    def most_options(self, kind: QuestionType) -> int | None:
        return len(self.readout.codes)

    def text(self, request: Request, question: Question) -> str:
        """The whole prompt of one question (the module docstring)."""
        shown = options(question)
        self.readout.check(question.id, len(shown))
        lines = [
            f"({code}) {key}" if text == key else f"({code}) {key}: {text}"
            for code, (key, text) in zip(self.readout.codes, shown, strict=False)
        ]
        body = (
            f"Context:\n{as_text(request.state)}\n\nQuestion: {instructions_text(question.instructions)}"
            "\n\nOptions:\n" + "\n".join(lines)
        )
        return USER_TURN + body.rstrip() + BODY_TAIL + ANSWER_TURN + ANSWER_CUE

    def render(self, request: Request) -> Rendering:
        texts = [self.text(request, question) for question in request.questions]
        prompts = []
        for question, ids in zip(request.questions, self.tokenizer.encode_batch(texts), strict=True):
            prompts.append(Prompt(tuple(ids), (Readout(question.id, len(ids) - 1),), ()))
        return Rendering(request, tuple(prompts))

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        return {
            question.id: self.readout(hidden.rows(question.id)[-1], len(options(question)))
            for question in rendering.request.questions
        }

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        probabilities = tuple(float(x) for x in scores)  # prompt order: a noul's true first, as question.options
        return probabilities, systemone_answer(
            question, probabilities, choice_confidence=margin, score_confidence=margin
        )


class Glide(AdapterBase):
    api = API_VERSION
    name = "fastino/glide"
    compare_arms = ("glide-package",)  # the creator's glide package from the checkpoint (`compare.py`)
    files = (
        "config.json",
        "decision_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "readout.safetensors",
        "model*.safetensors",
        "model.safetensors.index.json",
    )

    def bench_questions(self, model: ModelDir, count: int, variant: int = 0) -> dict[str, dict[str, Any]]:
        return asks.yes_no(count, variant)

    def matches(self, model: ModelDir) -> bool:
        config = model.json("decision_config.json") if model.has("decision_config.json") else None
        return isinstance(config, dict) and config.get("prompt_format") == PROMPT_FORMAT

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        config = decision_config(model)
        tokenizer = ctx.tokenizer(model)
        readout = CodeReadout.load(ctx, model, codes(model, config, tokenizer), float(config.get("temperature", 1.0)))
        fmt = GlideFormat(tokenizer, readout)
        weights = ctx.weights(model, exclude=("readout.safetensors",))
        details: dict[str, Any] = {
            "attention_mode": config["attention_mode"],
            "temperature": readout.temperature,
            "prompt_format": PROMPT_FORMAT,
            "max_length": fmt.window,
            "images": "none (text only)",
        }
        depth = recurrence(config["midloop"]) if config.get("midloop") else None
        backbone = ctx.backbone(qwen35, model, weights=weights, causal=False, recurrence=depth)
        details["recurrent_depth"] = (
            f"layers {', '.join(f'{a}-{b}' for a, b in depth.blocks)} read {depth.passes} times"
            if depth is not None and depth.passes > 1
            else "none"
        )
        return Loaded(backbone, fmt, details)


def decision_config(model: ModelDir) -> Mapping[str, Any]:
    """decision_config.json, checked to be a layout the creator's code reads and this adapter runs: version 1, read
    at the last token, a Qwen3.5 with full attention both ways, no "loop"."""
    config: Mapping[str, Any] = model.json("decision_config.json")
    if (
        config.get("format_version", 1) != 1
        or config.get("pooling", "last") != "last"
        or config.get("attention_mode") != "noncausal_full_attention"
        or config.get("loop")
    ):
        raise UnsupportedModel(f"{model.path}: a GLiDE decision_config.json this adapter doesn't read")
    return config


def codes(model: ModelDir, config: Mapping[str, Any], tokenizer: Tokenizer) -> list[str]:
    """The readout's 255 codes, each the single token decision_config.json names, as the creator's `_check_codes`
    checks them against the tokenizer that renders the prompt."""
    listed, ids = list(config.get("codes") or ()), list(config.get("token_ids") or ())
    if len(listed) != CODE_ROWS or len(ids) != CODE_ROWS:
        raise UnsupportedModel(f"{model.path}: decision_config.json must list {CODE_ROWS} codes and token ids")
    if any(tokenizer.encode(code) != [i] for code, i in zip(listed, ids, strict=True)):
        raise UnsupportedModel(f"{model.path}: the codes' token ids differ from decision_config.json's")
    return listed


def recurrence(midloop: Mapping[str, Any]) -> Recurrence:
    """decision_config.json's "midloop" as the creator's recurrent.py `resolve` reads it: the parameter-free
    damped Euler form only (strategy "euler", conv "A", no gates or LoRA), its blocks [a, b] in order of b."""
    if midloop.get("strategy", "euler") != "euler" or midloop.get("conv", "A") != "A":
        raise UnsupportedModel(f"recurrent depth {dict(midloop)}: only the damped Euler form is supported")
    if int(midloop.get("n_gate") or 0) or int(midloop.get("n_lora") or 0):
        raise UnsupportedModel("recurrent depth with gates or LoRA is not supported")
    given = midloop.get("blocks") or [[midloop["a"], midloop["b"]]]
    blocks = tuple(sorted(((int(a), int(b)) for a, b in given), key=lambda block: block[1]))
    return Recurrence(blocks, int(midloop.get("passes") or 1))
