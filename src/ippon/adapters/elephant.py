"""Base Labs' elephant-1 (MIT): a Qwen3.5-9B decision model with block attention, answering each question with the
label of an option, read as its creator's serving runtime reads it (OneJev's qev at 81ce62f with the checkpoint's
serve.patch: prompt.py, engine.py, readout.py, answers.py), checked token for token against it:

- every question is its own prompt: Qwen's chat template around qev's system prompt and the user turn
  "<state>\\n<state>\\n</state>\\n\\n<header> <instructions>\\n\\nOptions:\\nA. <name>: <text>\\n...\\n\\nAnswer with
  one letter: A, B.", ending where the assistant's answer starts (thinking off); the state is used as written when it
  is text, nothing when there is none, else indented JSON; a noul shows "yes" then "no" (with qev's default texts), a
  choice its keys in the caller's order, a score "level 0", "level 1", ... with the level texts;
- block attention (config.json's modeling_overrides, Backbone.block): the system turn and the state, up to and
  including "</state>\\n\\n", are one block every question shares, read once and both ways within itself; each
  question's part reads that block and itself both ways, never another question; the Gated DeltaNet layers stay
  causal. A kept reading of the block is reused only for the same block;
- the answer is read at the last token: the final hidden state times the output matrix's rows of the slot labels
  (A..Z, then two-letter codes: 256), in fp32, divided by the calibration's temperature (1 without a
  calibration.json, as b3f8a93 ships), softmax; the confidences are System One's, every number rounded to 6 decimals
  as qev's answers are;
- qev refuses a request whose block and longest question pass 32,768 tokens, or whose block and every question pass
  65,536; nothing is cut.

Images (`Request.images`) are read where the state names them, `<image:N>` (1-based, each image once, in order: OneJev's
qev/media.py), each placeholder becoming the chat template's `<|vision_start|><|image_pad|><|vision_end|>` and its
image's rows, so they are part of the state's block; a state that names none has them at its start, a line before the
rest (ippon's convention: the creator's code takes only placeholders). Placeholders that name the images otherwise (out
of order, one missing, a video's) are refused, as the creator's code refuses them. An image is decoded as the creator's
train/mm.py loads it (no EXIF turn, alpha dropped) and resized within its processor's own range, 65,536 to 16,777,216
pixels, and read by the checkpoint's vision tower (adapters/qwen.py). Checked against the path the checkpoint was
trained and evaluated on, train.common.batch_scores with the readout attached as train/eval_mm.py attaches it (qev's
served multimodal engine is not touched by serve.patch and reads one way), in references/elephant.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..adapter import AdapterBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import qwen35
from ..errors import ContextTooLong, InvalidRequest, UnsupportedModel
from ..formulas import above_chance, near_best
from ..media.qwen import QwenImages
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout
from ..readout import answer_tokens, last_label_probabilities
from ..schema import Question, QuestionType, Request, rounded, systemone_answer
from ..tokenizer import Tokenizer
from .qwen import VISION, QwenImageFormat, chat, chat_head

SYSTEM = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# the two-letter codes qev skips because they are not one token, and the slots it reads at most
TWO_TOKEN_CODES = frozenset(
    {
        "BQ",
        "BZ",
        "CJ",
        "CQ",
        "CZ",
        "DQ",
        "DZ",
        "EJ",
        "EY",
        "FJ",
        "FQ",
        "FV",
        "FZ",
        "GJ",
        "GK",
        "GQ",
        "GZ",
        "HJ",
        "IY",
        "JF",
        "JG",
        "JH",
        "JL",
        "JN",
        "JQ",
    }
)
SLOTS = 256
LABELS = tuple(list(LETTERS) + [a + b for a in LETTERS for b in LETTERS if a + b not in TWO_TOKEN_CODES])[:SLOTS]
NOUL_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_TEXTS = {"true": "the statement is true / the answer is yes", "false": "the statement is false / the answer is no"}
DEFAULT_INSTRUCTIONS = {"choice": "Which option applies to the state?", "score": "Which level describes the state?"}
MAX_BRANCH, MAX_REQUEST = 32768, 65536  # qev's DecisionEngine defaults, as the checkpoint's engine.py uses them
PLACEHOLDER = re.compile(r"<(image|video):(\d+)>")  # qev/media.py's and train/mm.py's


def entry(value: Any) -> str:
    """Instructions or an option's text as qev writes them: text stripped, anything else as indented JSON."""
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def state_text(state: Any) -> str:
    if state is None:
        return ""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def placed(text: str, count: int) -> str:
    """The state's text with its `count` images placed: at its placeholders, each the chat template's image; at its
    start, a line before the rest, when it names none. InvalidRequest for placeholders the creator refuses."""
    found = list(PLACEHOLDER.finditer(text))
    if not found:
        return VISION * count + ("\n" + text if text else "")
    named = [int(m.group(2)) for m in found]
    if any(m.group(1) != "image" for m in found) or named != list(range(1, count + 1)):
        raise InvalidRequest(
            f"images: the state names {', '.join(m.group(0) for m in found)}; it must name each of its {count} "
            "image(s) once, in order, as <image:1>, <image:2>, ..., or none of them"
        )
    return PLACEHOLDER.sub(VISION, text)


def suffix(q: Question) -> str:
    """The question's part of the user turn, after the state's block."""
    if q.type == "noul":
        given = dict(q.criteria)
        options = [
            (name, entry(given[key]) if given.get(key) is not None else NOUL_TEXTS[key])
            for name, key in (("yes", "true"), ("no", "false"))
        ]
        instructions = entry(q.instructions) or NOUL_INSTRUCTIONS
    elif q.type == "score":
        options = [(f"level {i}", entry(level)) for i, (_, level) in enumerate(q.criteria)]
        instructions = entry(q.instructions) or DEFAULT_INSTRUCTIONS["score"]
    else:
        options = [(key, entry(description)) for key, description in q.criteria]
        instructions = entry(q.instructions) or DEFAULT_INSTRUCTIONS["choice"]
    labels = LABELS[: len(options)]
    lines = [
        f"{label}. {name}: {text}" if text else f"{label}. {name}"
        for label, (name, text) in zip(labels, options, strict=True)
    ]
    header = "Rate the state:" if q.type == "score" else "Question:"
    answer = "letter" if len(labels) <= len(LETTERS) else "label"
    return (
        f"{header} {instructions}\n\nOptions:\n"
        + "\n".join(lines)
        + f"\n\nAnswer with one {answer}: {', '.join(labels)}."
    )


@dataclass
class ElephantFormat(QwenImageFormat):
    tokenizer: Tokenizer
    output: Any  # the output matrix, whose slot-label rows score the options
    slots: tuple[int, ...]  # each label's token where the answer is read
    temperatures: dict[str, float]  # calibration.json's, by "type:options" or "type"
    images: QwenImages | None = None  # None: a build without the vision tower, text only

    @property
    def anchors(self) -> frozenset[str]:
        return frozenset(("images",)) if self.images is not None else frozenset()

    def most_options(self, kind: QuestionType) -> int | None:
        return len(self.slots)

    def temperature(self, q: Question) -> float:
        return self.temperatures.get(f"{q.type}:{len(q.options)}", self.temperatures.get(q.type, 1.0))

    def render(self, request: Request) -> Rendering:
        for q in request.questions:
            if len(q.options) > len(self.slots):
                raise InvalidRequest(f"{q.id}: {len(q.options)} options; this model reads at most {len(self.slots)}")
        text = state_text(request.state)
        if request.images:
            if any(PLACEHOLDER.search(suffix(q)) for q in request.questions):
                raise InvalidRequest("images: name them in the state, not in a question")
            text = placed(text, len(request.images))
        rows, pixels = self.place(request)
        block_text = chat_head(SYSTEM) + f"<state>\n{text}\n</state>\n\n"
        block = self.tokenizer.encode(block_text)
        state = block_text[len(chat_head(SYSTEM)) :]
        size = len(self.expand_checked(block, rows))
        prompts = []
        for q in request.questions:
            ids = self.tokenizer.encode(chat(SYSTEM, state + suffix(q)))
            if ids[: len(block)] != block:  # qev refuses a block its tokenizer doesn't end where the question starts
                raise InvalidRequest(f"{q.id}: the state's block does not end at a token boundary")
            ids = self.expand_checked(ids, rows)  # a text request's as they are, as qev reads them
            prompts.append(Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), (size,)))
        longest = max(len(p.tokens) for p in prompts)
        total = size + sum(len(p.tokens) - size for p in prompts)  # the block once, its images expanded
        if longest > MAX_BRANCH or total > MAX_REQUEST:
            raise ContextTooLong(
                f"the state and its longest question are {longest} tokens (at most {MAX_BRANCH}), the request "
                f"{total} (at most {MAX_REQUEST}): qev refuses longer requests"
            )
        return Rendering(request, tuple(prompts), pixels=pixels)

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        out = {}
        for q in rendering.request.questions:
            out[q.id] = last_label_probabilities(
                hidden.rows(q.id), self.output, self.slots[: len(q.options)], self.temperature(q)
            )
        return out

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        p = tuple(float(x) for x in scores)  # prompt order, which is q.options order: a noul's yes, then no
        shaped = systemone_answer(question, p, choice_confidence=above_chance, score_confidence=near_best)
        return p, rounded(shaped, 6)

    def input_tokens(self, rendering: Rendering) -> int:
        """The block once and every question's part, as qev counts them."""
        block = rendering.prompts[0].marks[0]
        return block + sum(len(p.tokens) - block for p in rendering.prompts)


class Elephant(AdapterBase):
    api = API_VERSION
    name = "baselabs/elephant"
    files = (
        "config.json",
        "readout.json",
        "calibration.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "model*.safetensors",
        "model.safetensors.index.json",
    )

    def matches(self, model: ModelDir) -> bool:
        overrides = model.config.get("modeling_overrides") or {}
        attention = overrides.get("attention") or {}
        return overrides.get("readout") == "slot_lm" and attention.get("full_attention") == "block_bidirectional"

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        overrides = model.config["modeling_overrides"]
        if overrides["attention"].get("linear_attention", "causal") != "causal":
            raise UnsupportedModel(f"{model.path}: expected causal linear-attention layers")
        slots = int(overrides.get("n_slots", SLOTS))
        if slots > SLOTS:
            raise UnsupportedModel(f"{model.path}: {slots} slots; qev's labels give {SLOTS}")
        weights = ctx.weights(model)
        backbone = ctx.backbone(qwen35, model, weights=weights, block=True)
        tokenizer = ctx.tokenizer(model)
        images = QwenImages.attach(model, ctx, backbone, weights, tokenizer, exif=False)  # the processor's own range
        probe = chat(SYSTEM, "<state>\n\n</state>\n\nQuestion: q")
        calibration = model.json("calibration.json") if model.has("calibration.json") else {}
        temperatures = {k: float(t) for k, t in (calibration or {}).get("temperatures", {}).items()}
        fmt = ElephantFormat(
            tokenizer, backbone.output_matrix, answer_tokens(tokenizer, probe, LABELS[:slots]), temperatures, images
        )
        return Loaded(
            backbone,
            fmt,
            {
                "attention_mode": "block_bidirectional",
                "readout": "slot_lm",
                "slots": slots,
                "calibration": "calibration.json" if temperatures else "none",
                "images": fmt.images_detail(),
            },
        )
