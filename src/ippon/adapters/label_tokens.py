"""Bring your own causal decision model: a language model that answers with a token (a reranker's "yes", a
classifier's label, an end-of-turn token) loads with no adapter code, given a `ippon.json` beside its weights
that says how it is asked:

    {
      "format": "label_tokens",
      "prompt": "<|im_start|>user\\n<Query>: {instructions}\\n<Document>: {state}<|im_end|>\\n<|im_start|>assistant\\n",
      "noul": {"true": "yes", "false": "no"},
      "choice": "{label}",
      "score": "{index}",
      "option": "{label}: {description}",
      "over": "answers"
    }

- `prompt` is the whole prompt (chat markers written out, as the model was trained), over {state}, the question's
  {instructions} and its {options} (each option as `option`, one per line). The tokenizer's own special tokens are
  added as `tokenizer(prompt)` adds them.
- The answer is read at the prompt's last token: each option's answer text's first token, its logit from the
  model's output matrix. A `noul` answers "true"/"false" with `noul`'s texts, a `choice` each label with `choice`
  (over {label}), a `score` each level with `score` (over {index}, from 0, and {level}).
- `over: "answers"` (the default) softmaxes over the options' tokens; `over: "vocabulary"` reads each option's
  token's probability among every token (an end-of-turn detector's P(<|im_end|>)): a noul's "false" may then be
  left out (1 - P(true)), and a choice's or a score's probabilities are renormalized over its options.

Questions asked about one state share its reading (the prompts' common start), and an agent's next request about the
same, or a grown, state resumes from the kept reading. Architectures: the dense decoders (Llama, Mistral, Qwen2,
Qwen3), Gemma 3, Qwen3.5 and LFM2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter
from typing import Any

import mlx.core as mx

from .. import asks
from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import DECODERS
from ..errors import InvalidRequest, UnsupportedModel
from ..host_rows import HostRows
from ..model_dir import API_VERSION, SETTINGS, ModelDir
from ..plan import Prompt, Readout, state_end_marks
from ..quant import Table, mm
from ..readout import last_label_probabilities
from ..schema import Question, Request, as_text
from ..tokenizer import Tokenizer

FORMAT = "label_tokens"
KEYS = {"format", "prompt", "noul", "choice", "score", "option", "over"}


@dataclass(frozen=True)
class Settings:
    prompt: str
    noul: Mapping[str, str]
    choice: str = "{label}"
    score: str = "{index}"
    option: str = "{label}: {description}"
    over: str = "answers"

    @classmethod
    def read(cls, found: Mapping[str, Any]) -> Settings:
        if found.get("format") != FORMAT or not set(found) <= KEYS:
            raise UnsupportedModel(f'{SETTINGS}: "format": "{FORMAT}" and only {", ".join(sorted(KEYS))}')
        prompt = found.get("prompt")
        try:
            if not isinstance(prompt, str):
                raise ValueError
            split_state(prompt)
        except ValueError:
            raise UnsupportedModel(f'{SETTINGS}: "prompt" is a template with {{state}} at most once') from None
        noul = dict(found.get("noul") or {"true": "yes", "false": "no"})
        if "true" not in noul or not set(noul) <= {"true", "false"}:
            raise UnsupportedModel(f'{SETTINGS}: "noul" maps "true" (and "false") to their answer texts')
        over = found.get("over", "answers")
        if over not in ("answers", "vocabulary"):
            raise UnsupportedModel(f'{SETTINGS}: "over" is "answers" or "vocabulary"')
        if over == "answers" and "false" not in noul:
            raise UnsupportedModel(f'{SETTINGS}: with "over": "answers", a noul needs a "false" answer too')
        return cls(
            prompt,
            noul,
            str(found.get("choice", "{label}")),
            str(found.get("score", "{index}")),
            str(found.get("option", "{label}: {description}")),
            over,
        )


def split_state(prompt: str) -> tuple[str, str] | None:
    """The templates before and after `prompt`'s {state} field, as str.format reads its fields ({{state}} is text);
    None without one. ValueError for a malformed template, a second {state}, or one with a conversion or spec."""
    before: list[str] = []
    after: list[str] = []
    seen = False
    for literal, name, spec, conversion in Formatter().parse(prompt):
        side = after if seen else before
        side.append(literal.replace("{", "{{").replace("}", "}}"))
        if name is None:
            continue
        if name == "state":
            if seen or spec or conversion:
                raise ValueError(prompt)
            seen = True
            continue
        side.append("{" + name + (f"!{conversion}" if conversion else "") + (f":{spec}" if spec else "") + "}")
    return ("".join(before), "".join(after)) if seen else None


@dataclass
class LabelTokenFormat(FormatBase):
    tokenizer: Tokenizer
    output: Table  # the model's output matrix [vocabulary, hidden]
    settings: Settings

    def answers(self, q: Question) -> list[str]:
        """Each option's answer text, in the question's option order (a noul: true, then false when given)."""
        s = self.settings
        if q.type == "noul":
            return [s.noul["true"]] + ([s.noul["false"]] if "false" in s.noul else [])
        if q.type == "score":
            return [s.score.format(index=i, level=as_text(level, "")) for i, (_, level) in enumerate(q.criteria)]
        return [s.choice.format(label=label) for label in q.options]

    def tokens(self, q: Question) -> list[int]:
        """The first token of each option's answer text; they must differ."""
        firsts = []
        for text in self.answers(q):
            ids = self.tokenizer.encode(text)
            if not ids:
                raise InvalidRequest(f"{q.id}: the answer text {text!r} has no tokens")
            firsts.append(ids[0])
        if len(set(firsts)) != len(firsts):
            raise InvalidRequest(f"{q.id}: two options' answers start with the same token, which can't tell them apart")
        return firsts

    def render(self, request: Request) -> Rendering:
        s, state = self.settings, as_text(request.state, "")
        parts = split_state(s.prompt)
        before, after = parts or (s.prompt, "")
        prompts = []
        for q in request.questions:
            self.tokens(q)  # every question's answers are readable before anything runs
            fields = {"instructions": as_text(q.instructions, ""), "options": self.options(q)}
            head = before.format(**fields)
            text = head + (state + after.format(**fields) if parts else "")
            ((ids, offsets),) = self.tokenizer.encode_offsets([text], special=True)
            marks = state_end_marks(text, offsets, len(head) + len(state)) if parts and state else ()
            prompts.append(Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), tuple(m for m in marks if m < len(ids))))
        return Rendering(request, tuple(prompts))

    def options(self, q: Question) -> str:
        shown = []
        for i, (label, description) in enumerate(q.criteria):
            text = label if q.type != "score" else as_text(description, "")
            shown.append(self.settings.option.format(label=text, description=as_text(description, ""), index=i))
        return "\n".join(shown)

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        out = {}
        for q in rendering.request.questions:
            tokens = self.tokens(q)
            if self.settings.over == "answers":
                out[q.id] = last_label_probabilities(hidden.rows(q.id), self.output, tokens)
                continue
            h, ids = hidden.rows(q.id)[-1].astype(mx.float32), mx.array(tokens)
            # every token's logit: the output matrix as it is stored (in the compute dtype), then fp32
            assert not isinstance(self.output, HostRows)  # loaded whole for this (`load`: host_tables)
            dtype = self.output.dtype
            every = mx.softmax(mm(h.astype(dtype)[None], self.output)[0].astype(mx.float32))
            picked = every[ids]
            if q.type == "noul" and ids.size == 1:
                out[q.id] = mx.concatenate([picked, 1 - picked])
            else:
                out[q.id] = picked / mx.sum(picked)
        return out


class LabelTokens(AdapterBase):
    api = API_VERSION
    compare_arms = ("mlx-lm-model", "transformers")  # the plain ways to run one (`compare.py`)
    name = "generic/label-tokens"
    generic = True  # matched only when no model's own adapter matches
    files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        SETTINGS,
        "*.safetensors",
        "model.safetensors.index.json",
    )

    def bench_questions(self, model: ModelDir, count: int, variant: int = 0) -> dict[str, dict[str, Any]]:
        return asks.yes_no(count, variant)

    def matches(self, model: ModelDir) -> bool:
        config = model.config.get("text_config", model.config)
        if config.get("model_type") not in DECODERS or not model.has(SETTINGS) or not model.has("tokenizer.json"):
            return False
        found = model.json(SETTINGS)
        return isinstance(found, Mapping) and found.get("format") == FORMAT

    def validate(self, model: ModelDir) -> None:
        """Validate published request settings before any weights are downloaded."""
        Settings.read(model.settings())

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        settings = Settings.read(model.settings())
        config = model.config.get("text_config", model.config)
        kind = config.get("model_type")
        if kind not in DECODERS:
            raise UnsupportedModel(f"{model.path}: model_type {kind!r} is not a causal architecture ippon covers")
        weights = ctx.weights(model)
        weights.host_tables = settings.over == "answers"  # over every token, the output matrix multiplies whole
        backbone = ctx.backbone(DECODERS[kind], model, weights=weights)
        fmt = LabelTokenFormat(ctx.tokenizer(model), backbone.output_matrix, settings)
        return Loaded(backbone, fmt, {"architecture": kind, "over": settings.over})
