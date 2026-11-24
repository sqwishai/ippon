"""Jeff (Jonathan Geuter; CC BY-NC 4.0): Jeff-1.1-Medium (Gemma 4 12B) and Jeff-1.1-Qwen-4B (Qwen3.5 4B), each with
a merged LoRA and a decision head, whose full-attention layers read the prompt both ways (Qwen3.5's Gated DeltaNet
layers stay causal, as bidir.py leaves them).

A reimplementation of the one-pass path of its serving code (github.com/j-geuter/Jeff v1.3.0: prompting.py, head.py,
bidir.py, model.py and typesafe.py), checked token for token against it:

- every question is its own prompt: the base's chat template (thinking off) around SemIf's system message and the user
  turn, the JSON object {"evidence": state, "criterion": instructions, "options": [{"letter": "A", "description":
  ...}, ...]} written with Python's default separators and non-ASCII kept;
- options follow JevBench's mapping (typesafe.question_options): a noul's "true" then "false" (undescribed: "The
  proposition is true."), a choice's keys in the request's order (undescribed: the key), a score's levels
  "0".."n-1"; every description is shown as "<id>: <description>"; instructions that aren't text are JSON, none are
  "(no instructions were given)"; an empty state is "(no state was given)" and a bare number its JSON
  (longinput.plain_state);
- the full-attention layers read every position, the sliding ones stay causal (jeff_config.json "bidirectional":
  "full"); the head reads the last position's final hidden state, rounded to bf16 as the creator's code rounds it, in
  fp32: cap(h W0) + A (h - mu) / sigma + b over the first k of its 16 slots (Qwen has no cap), softmax;
- the served answer applies serving_rule.json: one temperature per question type, then a yes/no answer inside
  (0.2, 0.8) moves to 0.801 or 0.199, as the server commits it by default (the probabilities `verify` compares stay
  the head's);
- a choice of more than 16 options is asked as decide_wide asks it: groups of at most 16 with bounds at
  round(i * n / groups), then the groups' winners, an option getting its winner's final probability times its own in
  its group (rounds.py).

Not covered: the server's filtering of prompts over 8,192 tokens (longinput.shorten keeps the units that mention the
question): such a prompt is refused here. Reasoning (off by default in the creator's server) is not offered.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import gemma4, qwen35
from ..errors import UnsupportedModel
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout
from ..readout import softmax
from ..rounds import Wide
from ..schema import Question, QuestionType, Request, as_text
from ..tokenizer import Tokenizer
from ..weights import Weights
from . import qwen

PROMPT_VERSION = "direct-options-v1"
LETTERS = "ABCDEFGHIJKLMNOP"  # the head's 16 slots
MAX_WIDE = len(LETTERS) ** 2  # decide_wide's most options
MAX_TOKENS = 8192
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
BAND = (0.2, 0.8)  # JevBench's yes/no abstention band
EDGE = (0.199, 0.801)  # where a committed answer moves to


def split_groups(options: Sequence[str], group: int) -> tuple[tuple[str, ...], ...]:
    """Near-equal groups of at most `group` options, in order, with bounds at round(i * n / groups), as Jeff's
    split_groups makes them (17 options: 8 then 9)."""
    count = -(-len(options) // group)
    bounds = [round(i * len(options) / count) for i in range(count + 1)]
    return tuple(tuple(options[bounds[i] : bounds[i + 1]]) for i in range(count))


def group_product(
    q: Question,
    groups: Sequence[Sequence[str]],
    first: Sequence[Sequence[float]],
    final: Mapping[str, float],
    wide: Wide,
) -> list[float]:
    """decide_wide's probabilities: each option's within its group times its group's one finalist's final one."""
    out: dict[str, float] = {}
    for g, ps in zip(groups, first, strict=True):
        won = next(final[k] for k in g if k in final)
        out |= {k: won * p for k, p in zip(g, ps, strict=True)}
    return [out[k] for k in q.options]


WIDE = Wide(len(LETTERS), len(LETTERS), 1, 0.0, split_groups, group_product)  # decide_wide's rounds


def gemma_chat(system: str, user: str) -> str:
    """Gemma 4's chat template over a system and a user message, the model's turn opened with thinking off."""
    return (
        f"<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


def plain_state(state: Any) -> Any:
    """The state as the prompt holds it (longinput.plain_state): text, an object or a list as given, an empty one as a
    note, anything else (a number) as its JSON."""
    if isinstance(state, (str, dict, list)) and state:
        return state
    if state is None or state in ("", [], {}):
        return "(no state was given)"
    return json.dumps(state, ensure_ascii=False)


def instructions(q: Question) -> str:
    text = q.instructions
    if isinstance(text, str) and text:
        return text
    if text is None or text in ("", {}, []):
        return "(no instructions were given)"
    return json.dumps(text, ensure_ascii=False)


def options(q: Question) -> list[tuple[str, str]]:
    """(id, description as shown) of each option in prompt order (typesafe.question_options)."""
    if q.type == "noul":
        given = dict(q.criteria)
        shown = [
            (k, as_text(given[k]) if given.get(k) is not None else f"The proposition is {k}.")
            for k in ("true", "false")
        ]
    elif q.type == "choice":
        shown = [(k, as_text(d) if d not in (None, "", [], {}) else k) for k, d in q.criteria]
    else:
        shown = [(str(i), as_text(level)) for i, (_, level) in enumerate(q.criteria)]
    return [(k, f"{k}: {d}") for k, d in shown]


def prompt_text(state: Any, q: Question, chat: Callable[[str, str], str]) -> str:
    payload = {
        "evidence": state,
        "criterion": instructions(q),
        "options": [{"letter": letter, "description": d} for letter, (_, d) in zip(LETTERS, options(q), strict=False)],
    }
    return chat(SYSTEM, json.dumps(payload, ensure_ascii=False))


def tempered(p: list[float], t: float) -> list[float]:
    """score_temperature: log probabilities divided by t, renormalised."""
    logs = [math.log(max(x, 1e-12)) / t for x in p]
    top = max(logs)
    e = [math.exp(v - top) for v in logs]
    return [v / sum(e) for v in e]


def committed(p_yes: float, rule: dict[str, Any]) -> float:
    """A yes/no probability under the serving rule: its temperature, then inside the band moved to its edge."""
    t = float(rule.get("noul_t", 1.0))
    if t != 1.0:
        x = min(max(p_yes, 1e-9), 1 - 1e-9)
        p_yes = 1 / (1 + math.exp(-math.log(x / (1 - x)) / t))
    margin = rule.get("noul_edge")
    if margin is not None and BAND[0] < p_yes < BAND[1] and abs(p_yes - 0.5) >= margin and p_yes != 0.5:
        return EDGE[1] if p_yes > 0.5 else EDGE[0]
    return p_yes


@dataclass(frozen=True)
class Head:
    """head.py's DecisionHead, in fp32: z = cap(h W0) + A (h - mu) / sigma + b."""

    rows: mx.array  # [16, hidden], the output matrix's rows of A..P
    a: mx.array
    b: mx.array
    mu: mx.array
    sigma: mx.array
    softcap: float | None

    @classmethod
    def load(cls, weights: Weights, softcap: float | None) -> Head:
        def w(name: str) -> mx.array:
            return weights.get(name, mx.float32)

        head = cls(w("base_rows"), w("delta.weight"), w("delta.bias"), w("mu"), w("sigma"), softcap)
        mx.eval(head.rows, head.a, head.b, head.mu, head.sigma)
        return head

    def __call__(self, h: mx.array, k: int) -> mx.array:
        h = h.astype(mx.bfloat16).astype(mx.float32)  # the creator rounds the hidden state to bf16 first
        z = h @ self.rows[:k].T
        if self.softcap:
            z = self.softcap * mx.tanh(z / self.softcap)
        return z + ((h - self.mu) / self.sigma) @ self.a[:k].T + self.b[:k]


@dataclass
class JeffFormat(FormatBase):
    tokenizer: Tokenizer
    head: Head
    rule: dict[str, Any]  # serving_rule.json
    chat: Callable[[str, str], str]
    window: int = MAX_TOKENS
    wide: Wide | None = WIDE

    def most_options(self, kind: QuestionType) -> int | None:
        return MAX_WIDE if kind == "choice" else len(LETTERS)

    def render(self, request: Request) -> Rendering:
        state = plain_state(request.state)
        prompts = []
        for q in request.questions:
            # a prompt past the window is refused (Decider.render): Jeff's server keeps the parts of a longer state
            # that mention the question, which ippon does not do
            ids = self.tokenizer.encode(prompt_text(state, q, self.chat))
            prompts.append(Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), ()))
        return Rendering(request, tuple(prompts))

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        return {
            q.id: softmax(self.head(hidden.rows(q.id)[-1], len(options(q))), 1.0) for q in rendering.request.questions
        }

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        """The head's probabilities (a noul's true, then false) and the answer as the server sends it."""
        p = [float(x) for x in scores]
        ids = [k for k, _ in options(question)]
        if question.type == "noul":
            return (p[0], p[1]), {"type": "noul", "noul": committed(p[0], self.rule)}
        t = float(self.rule.get(f"{question.type}_t", 1.0))
        served = tempered(p, t) if t != 1.0 else p
        probabilities = dict(zip(ids, served, strict=True))
        best = max(ids, key=probabilities.__getitem__)
        value: Any = best if question.type == "choice" else int(best)
        return tuple(p), {"type": question.type, question.type: value, "probabilities": probabilities}


class Jeff(AdapterBase):
    api = API_VERSION
    name = "jgeuter/jeff"
    files = (
        "config.json",
        "jeff_config.json",
        "head_config.json",
        "head.safetensors",
        "serving_rule.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-*.safetensors",
        "model.safetensors.index.json",
    )

    def matches(self, model: ModelDir) -> bool:
        config = model.json("jeff_config.json") if model.has("jeff_config.json") else None
        return isinstance(config, dict) and config.get("prompt_version") == PROMPT_VERSION

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        config = model.json("jeff_config.json")
        for key, supported in (("noul_mapping", "tf-true-first"), ("max_options", len(LETTERS))):
            if config.get(key, supported) != supported:
                raise UnsupportedModel(f"{model.path}: {key} {config[key]!r} is not supported")
        mode = config.get("bidirectional")
        if mode not in (None, "full"):
            raise UnsupportedModel(f"{model.path}: bidirectional {mode!r} is not supported (only 'full')")
        head_config = model.json("head_config.json")
        if head_config.get("kind") != "residual" or head_config.get("extra_layers") or head_config.get("k_slots") != 16:
            raise UnsupportedModel(f"{model.path}: decision head {head_config} is not supported")
        weights = ctx.weights(model, exclude=("head.safetensors",))
        chat: Callable[[str, str], str]
        if str(model.config.get("model_type")).startswith("qwen3_5"):  # Jeff-1.1-Qwen-4B
            backbone: Any = ctx.backbone(qwen35, model, weights=weights, causal=mode != "full")
            chat = qwen.chat
        else:  # Gemma 4, which gemma4.load checks
            backbone = ctx.backbone(gemma4, model, weights=weights, causal=mode != "full")
            chat = gemma_chat
        head = Head.load(ctx.weights(model, ["head.safetensors"]), head_config.get("softcap"))
        rule = model.json("serving_rule.json") or {}
        fmt = JeffFormat(ctx.tokenizer(model), head, dict(rule), chat, int(config.get("max_input_tokens", MAX_TOKENS)))
        return Loaded(backbone, fmt, {"bidirectional": mode, "serving_rule": rule, "name": config.get("name")})
