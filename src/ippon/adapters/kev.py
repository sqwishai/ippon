"""Kev (Jared Palmer; Apache-2.0): a LoRA adapter on a Qwen3.5 base (Qwen/Qwen3.5-{0.8B,4B,9B}-Base) with a pointer
head, read as the creator's server reads it (github.com/jaredpalmer/kev: model.py, api.py and checkpoint.py), checked
token for token against it:

- a request maps onto Kev's record as api.to_record maps it: the state and each question's instructions flattened
  to text (render: an object's fields as `name: value` lines, a list's items as `- ` lines, nothing for null), a noul's
  options "no" then "yes", a choice's "name" or "name: description" in the request's order, a score's levels;
- the state is read once, `<|fim_prefix|>` and its tokens, and every question is its own causal row after it (the
  form Kev runs on hybrid backbones): `<|fim_middle|>` and the instructions, each option between `<|box_start|>` and
  `<|box_end|>`, then `<|fim_suffix|>` where the question is decided. Caller text is tokenized with every `<|name|>`
  rewritten to `<¦name¦>`, so a state or option can never write a delimiter;
- the head reads the final hidden states (after the final norm) in fp32: option k's logit is
  (k(h at its `<|box_end|>`) . q(h at `<|fim_suffix|>`)) / sqrt(256), divided by the calibration temperature
  head.pt carries, softmaxed over the question's options.

The adapter repository holds the LoRA, the head and Kev's tokenizer; the base is downloaded at the revision head.pt
pins and the LoRA merged into it at load (W + (alpha / r) B A in fp32, `Weights.with_lora`), as Kev's own server
merges it by default. head.pt is read with ptfile, which runs nothing from the pickle. The server's opt-in date facts
(KEV_DATE_FACTS) are not applied, as they are off by default.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .. import ptfile
from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import qwen35
from ..errors import ContextTooLong, UnsupportedModel
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout
from ..readout import softmax
from ..schema import Question, QuestionType, Request
from ..tokenizer import Tokenizer

STATE, QUESTION, OPEN, CLOSE, DECIDE = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
MAX_STATE = 65536  # the serving context: a state's tokens, its <|fim_prefix|> included (model.SERVE_MAX_STATE)
MAX_ROW = MAX_STATE + 8192  # a question's row: the state and its branch (SERVE_MAX_BRANCH)
MAX_OPTIONS = 255
SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def render(value: Any, indent: int = 0) -> str:
    """api.render: str | object | array flattened into the text the model reads, field names kept as labels."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in value)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
        for k, x in value.items()
    )


def option_text(name: str, description: Any) -> str:
    return name if description is None or description == "" else f"{name}: {render(description)}"


def options(q: Question) -> list[str]:
    """api.to_record's options in the order the head scores them (a noul: no, then yes)."""
    if q.type == "noul":
        given = dict(q.criteria)
        return [option_text("no", given.get("false")), option_text("yes", given.get("true"))]
    if q.type == "choice":
        return [option_text(k, d) for k, d in q.criteria]
    return [render(level) for _, level in q.criteria]


@dataclass(frozen=True)
class Head:
    """model.PointerHead in fp32, with the temperature head.pt carries."""

    q_weight: mx.array
    q_bias: mx.array
    k_weight: mx.array
    k_bias: mx.array
    temperature: float

    def __call__(self, decide: mx.array, opts: mx.array) -> mx.array:
        q = decide @ self.q_weight.T + self.q_bias
        k = opts @ self.k_weight.T + self.k_bias
        return (k @ q) * (1 / math.sqrt(self.q_weight.shape[0])) / self.temperature


@dataclass
class KevFormat(FormatBase):
    tokenizer: Tokenizer
    head: Head
    marks: dict[str, int]  # the delimiters' token ids

    def most_options(self, kind: QuestionType) -> int | None:
        return MAX_OPTIONS

    def text(self, value: str) -> list[int]:
        """model.user_tokens: caller text, with any `<|name|>` rewritten so it can never be a delimiter."""
        return self.tokenizer.encode(SPECIAL.sub(r"<¦\1¦>", value))

    def render(self, request: Request) -> Rendering:
        m = self.marks
        state = [m[STATE], *self.text(render(request.state))]
        if len(state) > MAX_STATE:
            raise ContextTooLong(f"the state is {len(state)} tokens, over Kev's {MAX_STATE}")
        prompts = []
        for q in request.questions:
            branch = [m[QUESTION], *self.text(render(q.instructions))]
            ends = []
            for o in options(q):
                branch += [m[OPEN], *self.text(o), m[CLOSE]]
                ends.append(len(state) + len(branch) - 1)
            branch.append(m[DECIDE])
            ids = state + branch
            if len(ids) > MAX_ROW:
                raise ContextTooLong(f"{q.id}: {len(ids)} tokens, over Kev's {MAX_ROW} for a state and its question")
            readouts = (*(Readout(q.id, e) for e in ends), Readout(q.id, len(ids) - 1))
            prompts.append(Prompt(tuple(ids), readouts, (len(state),)))
        return Rendering(request, tuple(prompts))

    def input_tokens(self, rendering: Rendering) -> int:
        """As Kev's server counts a request: the state once, then every question's branch (the packed record's ids)."""
        state = rendering.prompts[0].marks[0]
        return state + sum(len(p.tokens) - state for p in rendering.prompts)

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        """Each question's probabilities in its options' order (a noul: true, then false)."""
        out = {}
        for q in rendering.request.questions:
            h = hidden.rows(q.id).astype(mx.float32)
            p = softmax(self.head(h[-1], h[:-1]), 1.0)
            out[q.id] = p[mx.array([1, 0])] if q.type == "noul" else p
        return out


def meta(model: ModelDir) -> dict[str, Any]:
    return ptfile.read(model.path / "head.pt")


class Kev(AdapterBase):
    api = API_VERSION
    name = "jaredpalmer/kev"
    read_safely = ("head.pt",)  # its head and the base it pins, which `matches` and `base` read
    files = (
        "head.pt",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
    )

    def matches(self, model: ModelDir) -> bool:
        if not (model.has("head.pt") and model.has("adapter_config.json")):
            return False
        base = model.json("adapter_config.json").get("base_model_name_or_path", "")
        return isinstance(base, str) and base.startswith("Qwen/Qwen3.5-")

    def base(self, model: ModelDir) -> tuple[str, str]:
        """The base and revision head.pt pins, which must be the base the adapter names."""
        m = meta(model)
        if m.get("base") != model.json("adapter_config.json").get("base_model_name_or_path") or not m.get(
            "base_revision"
        ):
            raise UnsupportedModel(f"{model.path}: head.pt does not pin the adapter's base at a revision")
        return str(m["base"]), str(m["base_revision"])

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        m = meta(model)
        for key, supported in (("weights", "lora"), ("option_isolation", False), ("special_embeddings", False)):
            if m.get(key, supported) != supported:
                raise UnsupportedModel(f"{model.path}: Kev's {key} {m[key]!r} is not supported")
        if ctx.base is None:
            raise UnsupportedModel(f"{model.path}: a LoRA adapter loads with its base (ippon fetches it)")
        lora = ctx.weights(model, ["adapter_model.safetensors"])
        weights = ctx.weights(ctx.base).with_lora(lora, model.json("adapter_config.json"))
        backbone = ctx.backbone(qwen35, ctx.base, weights=weights)
        tokenizer = ctx.tokenizer(model)
        marks = {t: tokenizer.token_id(t) for t in (STATE, QUESTION, OPEN, CLOSE, DECIDE)}
        if any(i is None for i in marks.values()):
            raise UnsupportedModel(f"{model.path}: the tokenizer lacks one of Kev's delimiters")
        h = m["head"]
        head = Head(
            mx.array(h["q.weight"]),
            mx.array(h["q.bias"]),
            mx.array(h["k.weight"]),
            mx.array(h["k.bias"]),
            float(m.get("temperature", 1.0)),
        )
        fmt = KevFormat(tokenizer, head, {k: int(v) for k, v in marks.items() if v is not None})
        return Loaded(
            backbone,
            fmt,
            {"base_model": f"{m['base']}@{m['base_revision']}", "temperature": head.temperature},
        )
