"""AutoTrust Decider (AutoTrust-Decider-Mini-9B; Apache-2.0): Qwen3.5-9B with a LoRA and a 24-slot decision head, read
as the model card's System 1 code reads it (its "Plain transformers + peft" example), with a state rendered as the
creator's server renders one (vl/serve_decide.py, `_parts`), checked token for token against that code:

- every question is its own prompt, the text `[kind] KIND\\n[state] STATE\\n[question] QUESTION\\n[options]\\n` with one
  line per option and `\\n[decision]:`, tokenized as one string without special tokens and without a chat template;
- a noul's option lines are `false` and `true`, a score's `0` to `5` (six levels, the head's score slots), a choice's
  `A) option` to `P) option` (2 to 16, the head's choice slots), an option being its name, or `name: description`;
- a state that is text is written as it is, a JSON object as `json.dumps(state, ensure_ascii=False)`, a list as its
  items one after the other (text as it is, anything else as JSON), as the creator's server writes them;
- the head reads the last token's hidden state (after the final norm) in fp32: `head.safetensors`' linear layer to 24
  slots, the question's kind's slots (judge_config.json's ranges) divided by calibration.json's per-kind temperature,
  softmaxed over its options.

The repository holds Qwen3.5-9B's own weights and the LoRA in `adapter/`, merged into them at load (W + (alpha / r) B A
in fp32, `Weights.with_lora`), as the card's code merges it. The server's other paths (more than 16 options read from
the language-model head, its thinking mode, pictures through `vl/`) are not this checkpoint's System 1 head and are
not reimplemented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import qwen35
from ..errors import InvalidRequest, UnsupportedModel
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout, one_mark, state_end_marks
from ..readout import softmax
from ..schema import Question, QuestionType, Request
from ..tokenizer import Tokenizer

LETTERS = "ABCDEFGHIJKLMNOP"  # the head's choice slots, in order
LEVELS = ("0", "1", "2", "3", "4", "5")  # its score slots
ADAPTER = "adapter"  # judge_config.json's adapter_subfolder, where the card's code reads the LoRA


def state_text(state: Any) -> str:
    """The state as the creator's server writes it into the prompt (`_parts`)."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "".join(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False) for p in state)
    return json.dumps(state, ensure_ascii=False)


def option_lines(question: Question) -> list[str]:
    """The option lines of a question, in the order the head's slots score them."""
    if question.type == "noul":
        return ["false", "true"]
    if question.type == "score":
        if question.options != LEVELS:
            raise InvalidRequest(
                f"{question.id}: the decider scores on six levels, 0 to 5, and this score has {len(question.options)}"
            )
        return list(LEVELS)
    if not 2 <= len(question.criteria) <= len(LETTERS):
        raise InvalidRequest(
            f"{question.id}: the decider chooses among 2 to {len(LETTERS)} options, "
            f"and this choice has {len(question.criteria)}"
        )
    return [
        f"{LETTERS[i]}) {name if description in (None, '') else f'{name}: {state_text(description)}'}"
        for i, (name, description) in enumerate(question.criteria)
    ]


def prompt_text(kind: str, state: str, question: str, lines: list[str]) -> str:
    return f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


@dataclass
class AutoTrustFormat(FormatBase):
    tokenizer: Tokenizer
    weight: mx.array  # head.safetensors' proj.weight, [24, hidden], fp32
    bias: mx.array  # its proj.bias, [24]
    ranges: dict[str, tuple[int, int]]  # each kind's slots
    temperature: dict[str, float]  # calibration.json's per_kind

    def most_options(self, kind: QuestionType) -> int | None:
        return len(LETTERS) if kind == "choice" else None

    def render(self, request: Request) -> Rendering:
        state = state_text(request.state)
        prompts = []
        for question in request.questions:
            if not isinstance(question.instructions, str):
                raise InvalidRequest(f"{question.id}: the decider asks every question in words, and this one has none")
            head = f"[kind] {question.type}\n[state] "
            text = prompt_text(question.type, state, question.instructions, option_lines(question))
            ((ids, offsets),) = self.tokenizer.encode_offsets([text])
            end = len(head) + len(state)  # where the state ends: the tokens before it are shared by its next version
            marks = one_mark(state_end_marks(text, offsets, end), sum(1 for _, stop in offsets if stop <= end))
            prompts.append(Prompt(tuple(ids), (Readout(question.id, len(ids) - 1),), marks))
        return Rendering(request, tuple(prompts))

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        """Each question's probabilities in its options' order (a noul: true, then false)."""
        out = {}
        for question in rendering.request.questions:
            h = hidden.rows(question.id)[-1].astype(mx.float32)
            start, _ = self.ranges[question.type]
            n = 2 if question.type == "noul" else len(question.options)
            z = (self.weight @ h + self.bias) / self.temperature[question.type]
            p = softmax(z[start : start + n], 1.0)
            out[question.id] = p[mx.array([1, 0])] if question.type == "noul" else p
        return out


class AutoTrust(AdapterBase):
    api = API_VERSION
    name = "autotrust/decider"
    files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "head.safetensors",
        "judge_config.json",
        "calibration.json",
        f"{ADAPTER}/adapter_config.json",
        f"{ADAPTER}/adapter_model.safetensors",
        "README.md",
    )

    def matches(self, model: ModelDir) -> bool:
        judge = model.json("judge_config.json")
        return isinstance(judge, dict) and "slots" in judge and model.has("head.safetensors")

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        judge = model.json("judge_config.json")
        if judge.get("softcap") is not None:
            raise UnsupportedModel(f"{model.path}: a soft-capped decision head is not supported")
        if judge.get("weights_mode", "unmerged") != "unmerged" or judge.get("adapter_subfolder", ADAPTER) != ADAPTER:
            raise UnsupportedModel(f"{model.path}: the decider's LoRA must sit unmerged in {ADAPTER}/")
        slots = judge["slots"]
        if slots.get("verbalizers", [])[8:] != list(LETTERS) or slots.get("verbalizers", [])[2:8] != list(LEVELS):
            raise UnsupportedModel(f"{model.path}: the decider's head slots are not the 24 this adapter reads")
        lora = ctx.weights(model, [f"{ADAPTER}/adapter_model.safetensors"])
        weights = ctx.weights(model, exclude=("head.safetensors",)).with_lora(
            lora, model.json(f"{ADAPTER}/adapter_config.json")
        )
        backbone = ctx.backbone(qwen35, model, weights=weights)
        head = ctx.weights(model, ["head.safetensors"]).tensors
        calibration = model.json("calibration.json") or {}
        temperature = {k: float(calibration.get("per_kind", {}).get(k, 1.0)) for k in ("noul", "choice", "score")}
        fmt = AutoTrustFormat(
            ctx.tokenizer(model),
            head["proj.weight"].astype(mx.float32),
            head["proj.bias"].astype(mx.float32),
            {k: (int(a), int(b)) for k, (a, b) in slots["ranges"].items()},
            temperature,
        )
        return Loaded(backbone, fmt, {"temperature": temperature, "model_version": judge.get("model_version")})
