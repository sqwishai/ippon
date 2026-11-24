"""AWS Strands Decider v2 (strands-decider-{E2B,E4B,12B,26B-A4B,31B}-gemma4-v2-2611; Apache-2.0): Gemma 4 with
prefix-bidirectional attention across every layer (no sliding window) and a 255-way option-code readout
(`readout.safetensors`, Gemma 4 soft-capping at 30.0, and `decision_config.json`'s calibration temperature).

Reimplements `strands_decider.core` (`prompt.py`, `readout.py`, `request.py`, `config.py`, `confidence.py`):

- every question shares the state prefix `<bos><|turn>user\\n<state>\\n...\\n</state>\\n`, followed by its own
  `<question type="...">...</question><turn|>\\n<|turn>model\\n` turn;
- control tokens (`<bos>`, `<|turn>`, `<turn|>`) are emitted by id and all request text is encoded with
  special-token matching off, so text spelling `<turn|>` in a state stays literal text;
- every layer (both sliding and full) attends in prefix-bidirectional mode with no sliding window (`gemma4`,
  `block=True`): state tokens attend to state tokens both ways, and each question attends to the shared state and to
  itself both ways without seeing other questions;
- a noul lists `A. false` then `B. true` (with Strands' default descriptions when omitted); a choice takes 2 to 255
  options (`A`..`IV` without `GZ`); a score takes 2 to 10 levels lowest first;
- the readout computes `bf16(h) @ bf16(W[:k]).T`, applies `tanh(z / softcap) * softcap / T` and softmax in fp32,
  and formats answers with Strands' confidence formulas (`above_chance` for choice, `spread` for score) rounded to
  4 decimals.

Text only: Gemma 4's vision tower is not loaded here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from string import ascii_uppercase
from typing import Any

import mlx.core as mx

from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import gemma4
from ..errors import ContextTooLong, InvalidRequest, UnsupportedModel
from ..formulas import above_chance
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout
from ..schema import Question, QuestionType, Request, false_first, rounded, systemone_answer
from ..tokenizer import Tokenizer

INTERFACE = "strands-decider"
FORMAT_VERSION = 1
PROMPT_NAME = "strands-chat"
READOUT_FILE = "readout.safetensors"

# A..IV without GZ (Gemma 4 splits "GZ" into two tokens): 255 single-token option codes.
_NAMES = [*ascii_uppercase, *(a + b for a in ascii_uppercase for b in ascii_uppercase)][:256]
CODES: tuple[str, ...] = tuple(c for c in _NAMES if c != "GZ")
MAX_CODES = len(CODES)
MAX_SCORE_LEVELS = 10

BOS_ID, TURN_OPEN_ID, TURN_CLOSE_ID = 2, 105, 106
CONTROL_TOKENS = (("<bos>", BOS_ID), ("<|turn>", TURN_OPEN_ID), ("<turn|>", TURN_CLOSE_ID))

HEADERS: dict[QuestionType, str] = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}
NOUL_KEYS = ("false", "true")
NOUL_DEFAULT_CRITERIA = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}


def render_content(value: Any) -> str:
    """A string stripped; structured content as indented JSON in its own key order; None as empty."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, indent=2, ensure_ascii=False)


def score_confidence(probabilities: Sequence[float]) -> float:
    """Strands' ordinal confidence: `1 - sigma / ((L - 1) / 2)` clamped to `[0, 1]`."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    p = [x / total for x in probabilities]
    mean = sum(i * pi for i, pi in enumerate(p))
    sigma = sum(pi * (i - mean) ** 2 for i, pi in enumerate(p)) ** 0.5
    sigma_max = (n - 1) / 2.0
    return float(max(0.0, min(1.0, (sigma_max - sigma) / sigma_max)))


def option_line(code: str, name: str, description: str) -> str:
    """`A. name — description\\n`, with whitespace collapsed to single spaces."""
    name, description = " ".join(name.split()), " ".join(description.split())
    return f"{code}." + (f" {name}" if name else "") + (f" — {description}" if description else "") + "\n"


def question_options(q: Question) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Option names and rendered descriptions in Strands' prompt order (noul: false, true; score: 0..L-1)."""
    if q.type == "noul":
        given = dict(q.criteria)
        descs = tuple(render_content(given.get(k, NOUL_DEFAULT_CRITERIA[k])) for k in NOUL_KEYS)
        return NOUL_KEYS, descs
    if q.type == "score":
        if not 2 <= len(q.criteria) <= MAX_SCORE_LEVELS:
            raise InvalidRequest(f"{q.id}: score takes 2 to {MAX_SCORE_LEVELS} levels, got {len(q.criteria)}")
        return tuple(k for k, _ in q.criteria), tuple(render_content(d) for _, d in q.criteria)
    if not 2 <= len(q.criteria) <= MAX_CODES:
        raise InvalidRequest(f"{q.id}: choice takes 2 to {MAX_CODES} options, got {len(q.criteria)}")
    return tuple(k for k, _ in q.criteria), tuple(render_content(d) for _, d in q.criteria)


def state_tokens(tokenizer: Tokenizer, state: Any) -> tuple[int, ...]:
    """`<bos><|turn>user\\n<state>\\n...\\n</state>\\n`, with user text encoded literally."""
    rendered = render_content(state)
    body = "user\n<state>\n" + (f"{rendered}\n" if rendered else "") + "</state>\n"
    return (BOS_ID, TURN_OPEN_ID, *tokenizer.encode_literal(body))


def question_tokens(tokenizer: Tokenizer, q: Question, codes: Sequence[str]) -> tuple[int, ...]:
    """One question turn, ending at `<|turn>model\\n`, with request text encoded literally."""
    names, descs = question_options(q)
    lines = [option_line(code, name, desc) for code, name, desc in zip(codes, names, descs, strict=False)]
    inst = render_content(q.instructions)
    head = (
        f'<question type="{q.type}">\n{HEADERS[q.type]}\n'
        + (f"{inst}\n" if inst else "")
        + "<options>\n"
        + "".join(lines)
        + "</options>\nAnswer with the code of one option.\n</question>"
    )
    return (
        *tokenizer.encode_literal(head),
        TURN_CLOSE_ID,
        *tokenizer.encode_literal("\n"),
        TURN_OPEN_ID,
        *tokenizer.encode_literal("model\n"),
    )


@dataclass(frozen=True)
class StrandsReadout:
    """Strands Decider v2's 255-row readout (`readout.safetensors`), evaluated in training precision:
    `bf16(h) @ bf16(W[:k]).T`, then `softmax(tanh(z / softcap) * softcap / temperature)` in fp32."""

    weight: mx.array  # [255, hidden] bfloat16
    codes: tuple[str, ...]
    temperature: float
    softcap: float

    @classmethod
    def load(
        cls,
        ctx: LoadContext,
        model: ModelDir,
        codes: Sequence[str],
        temperature: float,
        softcap: float,
        hidden_size: int,
    ) -> StrandsReadout:
        weight = ctx.weights(model, [READOUT_FILE]).get("weight", mx.bfloat16)
        if weight.ndim != 2 or weight.shape != (MAX_CODES, hidden_size):
            raise UnsupportedModel(
                f"{model.path}: readout weight must be [{MAX_CODES}, {hidden_size}], not {list(weight.shape)}"
            )
        if tuple(codes) != CODES:
            raise UnsupportedModel(f"{model.path}: decision_config.json codes do not match Gemma 4's 255 codes")
        if not (math.isfinite(temperature) and temperature > 0):
            raise UnsupportedModel(f"{model.path}: temperature must be positive and finite")
        if not (math.isfinite(softcap) and softcap > 0):
            raise UnsupportedModel(f"{model.path}: final_logit_softcapping must be positive and finite")
        return cls(weight, tuple(codes), float(temperature), float(softcap))

    def __call__(self, row: mx.array, options: int) -> mx.array:
        z = (row.astype(mx.bfloat16) @ self.weight[:options].T).astype(mx.float32)
        s = mx.tanh(z / self.softcap) * self.softcap / self.temperature
        return mx.softmax(s, axis=-1)


@dataclass
class StrandsFormat(FormatBase):
    tokenizer: Tokenizer
    readout: StrandsReadout
    max_length: int

    def most_options(self, kind: QuestionType) -> int | None:
        return MAX_SCORE_LEVELS if kind == "score" else MAX_CODES

    def render(self, request: Request) -> Rendering:
        s_ids = state_tokens(self.tokenizer, request.state)
        prompts = []
        for q in request.questions:
            q_ids = question_tokens(self.tokenizer, q, self.readout.codes)
            total = len(s_ids) + len(q_ids)
            if total > self.max_length:
                raise ContextTooLong(
                    f"{q.id}: prompt of {total} tokens ({len(s_ids)} state + {len(q_ids)} question) "
                    f"exceeds the limit of {self.max_length} tokens"
                )
            ids = s_ids + q_ids
            prompts.append(Prompt(ids, (Readout(q.id, len(ids) - 1),), (len(s_ids),)))
        return Rendering(request, tuple(prompts))

    def input_tokens(self, rendering: Rendering) -> int:
        """The shared state once, then every question's own tokens (`RenderedRequest.input_tokens`)."""
        state = rendering.prompts[0].marks[0]
        return state + sum(len(p.tokens) - state for p in rendering.prompts)

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        return {
            q.id: self.readout(hidden.rows(q.id)[-1], len(question_options(q)[0])) for q in rendering.request.questions
        }

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        probabilities = false_first(question, scores)
        legend = {k: render_content(d) for k, d in question.criteria} if question.type == "score" else None
        out = systemone_answer(
            question, probabilities, choice_confidence=above_chance, score_confidence=score_confidence, legend=legend
        )
        return probabilities, rounded(out, 4)


class Strands(AdapterBase):
    api = API_VERSION
    name = "strands/decider"
    files = (
        "config.json",
        "decision_config.json",
        "readout.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "model*.safetensors",
        "model.safetensors.index.json",
    )

    def matches(self, model: ModelDir) -> bool:
        config = model.json("decision_config.json")
        return isinstance(config, dict) and config.get("interface") == INTERFACE

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        config = model.json("decision_config.json")
        if not isinstance(config, dict):
            raise UnsupportedModel(f"{model.path}: missing or invalid decision_config.json")
        if config.get("format_version") != FORMAT_VERSION:
            raise UnsupportedModel(f"{model.path}: format_version {config.get('format_version')!r} is not supported")
        if config.get("prompt") != PROMPT_NAME or config.get("pooling") != "last":
            raise UnsupportedModel(f"{model.path}: prompt {config.get('prompt')!r} / pooling is not supported")
        if config.get("attention_mode") != "prefix_bidir":
            raise UnsupportedModel(f"{model.path}: attention_mode {config.get('attention_mode')!r} is not supported")
        max_length = config.get("max_length_validated")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise UnsupportedModel(f"{model.path}: max_length_validated must be a positive integer")
        weights = ctx.weights(model, exclude=(READOUT_FILE,))
        backbone = ctx.backbone(gemma4, model, weights=weights, block=True)
        readout = StrandsReadout.load(
            ctx,
            model,
            config.get("codes") or (),
            float(config["temperature"]),
            float(config["final_logit_softcapping"]),
            backbone.hidden_size,
        )
        tokenizer = ctx.tokenizer(model)
        for token, expected in CONTROL_TOKENS:
            if tokenizer.token_id(token) != expected:
                raise UnsupportedModel(
                    f"{model.path}: control token {token!r} is {tokenizer.token_id(token)!r}, expected {expected}"
                )
        fmt = StrandsFormat(tokenizer, readout, max_length)
        raw_prov = config.get("provenance")
        provenance: dict[str, Any] = raw_prov if isinstance(raw_prov, dict) else {}
        return Loaded(
            backbone,
            fmt,
            {
                "attention_mode": "prefix_bidir",
                "temperature": readout.temperature,
                "softcap": readout.softcap,
                "max_length": max_length,
                "base_model": provenance.get("base_model"),
                "release": provenance.get("release"),
            },
        )
