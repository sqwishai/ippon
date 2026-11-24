"""Torchcast Decision (Torchcast AI; Apache-2.0): torchcast-decision-4b, Qwen3.5-4B whose full-attention layers
read the whole prompt both ways, answering with the letter of an option.

A reimplementation of the text path of its `torchcast_decision` serving package (prompt format "jev_render_v3",
render.py and model.py of v1.3.1), checked token for token against it:

- every question is its own prompt: Qwen's chat template around a fixed system prompt and the user turn
  "Evidence:\\n<state or (none)>\\n\\nQuestion: <instructions>\\nOptions:\\nA) ...\\nB) ...", ending where the
  assistant's answer starts (thinking off);
- the state is used as written when it is text, else as compact JSON in which every list of 8 or more items gives
  each item its position ("_index"); options follow the request's order, a noul "yes" then "no", a score its levels
  "0".."n-1" with their texts; an option line shows only its description when every choice is a bare label;
- the answer is read at the last token: the final hidden state times the output matrix's rows of the letters, in
  fp32, divided by the question type's temperature, softmax; a yes/no answer between 0.199 and 0.801 is moved to
  the nearer of the two, as the creator's server answers;
- a prompt over 16,384 tokens keeps the head and the tail of its state around a "\\n...\\n" line, as the creator's
  code (and its training) cut it.

The full-attention layers see later tokens, so questions can't share the state's reading: every prompt is read
whole. A choice of more than 26 options is asked as the creator asks it: in groups of up to 25 options, then their
finalists (rounds.py).

Images (`Request.images`): the card calls the model text-based, but its package reads images (model.py, `decide` with
`images`) and the checkpoint has a vision tower. Each image goes where the state (as text, JSON compact) says
"<image>", one mark per image, or, with no mark, ahead of the state: the image alone, or "Image 1: ...", "Image 2: ..."
lines, then a blank line and the state; marks that do not match the images are refused, as the package refuses them.
Each becomes `<|vision_start|><|image_pad|><|vision_end|>` and its rows; an image is decoded as `read_image` decodes it
(no EXIF turn, alpha dropped), resized within 65,536 and 1,048,576 pixels (the server's --max-pixels default) and read
by the checkpoint's vision tower (adapters/qwen.py). A prompt with images is never cut: over 16,384 tokens it is
refused, as the package refuses it.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..adapter import AdapterBase, Cut, Hidden, LoadContext, Loaded, Rendering
from ..arch import qwen35
from ..errors import InvalidRequest, UnsupportedModel
from ..formulas import above_chance, floored, near_best
from ..media.qwen import QwenImages
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout
from ..readout import answer_tokens, last_label_probabilities
from ..rounds import Wide
from ..schema import Question, QuestionType, Request, as_text, systemone_answer
from ..tokenizer import Tokenizer
from .qwen import MIN_PIXELS, VISION, QwenImageFormat, chat

PROMPT_FORMAT = "jev_render_v3"
SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
LETTERS = string.ascii_uppercase  # one per option: 26 at most in a pass
MAX_LEVELS = 10
POSITIONS_FROM = 8  # JSON lists at least this long give each item its "_index"
BUDGET = 16384  # tokens of a prompt, the training budget; past it the state keeps its head and tail
ELLIPSIS = "\n...\n"
NO_STATE = "(none)"
DEFAULT_NOUL_QUESTION = "Which answer fits the evidence?"
IMAGE_MARK = "<image>"  # where a state puts an image (model.py)
MAX_PIXELS = 1 << 20  # the server's --max-pixels default, an image's pixels after resizing
LABEL_ONLY = re.compile(r"\(?[a-z][).]?|\(?\d{1,3}[).]?|opt(?:ion)?[_ ]?\d{1,3}", re.IGNORECASE)


def _positions(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _positions(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) < POSITIONS_FROM:
            return [_positions(v) for v in value]
        return [
            {"_index": n, **_positions(v)} if isinstance(v, dict) else {"_index": n, "value": _positions(v)}
            for n, v in enumerate(value)
        ]
    return value


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return "" if state in (None, {}, []) else json.dumps(_positions(state), ensure_ascii=False)


def pictured(text: str, count: int) -> str:
    """The state's text with its `count` images, as the package's _state_with_images places them: at its marks, else
    ahead of it. InvalidRequest when its marks are not one per image."""
    marks = text.count(IMAGE_MARK)
    if marks == count:
        return text.replace(IMAGE_MARK, VISION)
    if marks:
        raise InvalidRequest(f"images: {count} images but {marks} {IMAGE_MARK} marks in the state")
    lead = VISION if count == 1 else "\n".join(f"Image {k}: {VISION}" for k in range(1, count + 1))
    return lead if not text.strip() else lead + "\n\n" + text


def _text(value: Any) -> str | None:
    """A description or instruction as the creator shows it: text stripped, other JSON written out; None when blank."""
    return (as_text(value).strip() or None) if value is not None else None


def _squash(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text).strip().lower()


def options(q: Question) -> list[tuple[str, str | None]]:
    """(id, description) of each option in prompt order: a noul true then false, a choice the caller's order, a
    score its levels "0".."n-1"."""
    if q.type == "noul":
        given = dict(q.criteria)
        return [("true", _text(given.get("true"))), ("false", _text(given.get("false")))]
    return [(key, _text(description)) for key, description in q.criteria]


def instructions(q: Question) -> str:
    text = _text(q.instructions)
    if text:
        return text
    if q.type == "noul":
        return DEFAULT_NOUL_QUESTION
    raise InvalidRequest(f"{q.id}: a {q.type} question needs instructions")


def option_lines(q: Question) -> list[str]:
    shown_options = options(q)
    labels_only = q.type == "choice" and all(LABEL_ONLY.fullmatch(k) and d for k, d in shown_options)
    lines = []
    for letter, (key, description) in zip(LETTERS, shown_options, strict=False):
        shown = ("yes" if key == "true" else "no") if q.type == "noul" else key
        if labels_only:
            body = description
        elif description is None or _squash(description) == _squash(shown):
            body = shown
        else:
            body = f"{shown}: {description}"
        lines.append(f"{letter}) {body}")
    return lines


def prompt_text(state: str, q: Question) -> str:
    """The prompt: the chat template over the system message and the question's user message."""
    user = f"Evidence:\n{state if state.strip() else NO_STATE}\n\nQuestion: {instructions(q)}\nOptions:\n"
    return chat(SYSTEM, user + "\n".join(option_lines(q)))


@dataclass
class TorchcastFormat(QwenImageFormat):
    tokenizer: Tokenizer
    output: Any  # the backbone's output matrix, whose letter rows score the options
    letters: tuple[int, ...]  # the token ids of "A".."Z" as read after the prompt
    temperatures: dict[str, float]
    noul_floor: float
    images: QwenImages | None = None  # None: a build without the vision tower, text only
    window: int = BUDGET  # the creator's window: a prompt's tokens at most (Decider.load's `window` replaces it)
    wide: Wide | None = None  # a choice past 26 letters, in rounds (decision_config.json's wide_choice)

    def most_options(self, kind: QuestionType) -> int | None:
        return MAX_LEVELS if kind == "score" else None  # a choice past 26 letters is asked in rounds (`wide`)

    def prompt(self, state: str, q: Question) -> tuple[list[int], Cut | None]:
        """The question's prompt tokens; over the window, its state's head and tail around ELLIPSIS (render.py)."""
        ids = self.tokenizer.encode(prompt_text(state, q))
        if len(ids) <= self.window:
            return ids, None
        tokens = self.tokenizer.encode(state)
        keep = len(tokens) - (len(ids) - self.window) - 16  # room for the marker and for re-tokenisation at the joins
        while keep > 0:
            head = keep // 2
            cut = self.tokenizer.decode(tokens[:head]) + ELLIPSIS + self.tokenizer.decode(tokens[-(keep - head) :])
            ids = self.tokenizer.encode(prompt_text(cut, q))
            if len(ids) <= self.window:
                return ids, Cut(q.id, "state", len(tokens) - keep, keep)
            keep -= max(16, len(ids) - self.window + 16)
        raise InvalidRequest(f"{q.id}: the prompt does not fit in {self.window} tokens even without its state")

    def render(self, request: Request) -> Rendering:
        if request.images:
            return self.render_images(request)
        state = state_text(request.state)
        prompts, cuts = [], []
        for q in request.questions:
            ids, cut = self.prompt(state, q)
            prompts.append(Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), ()))
            if cut is not None:
                cuts.append(cut)
        return Rendering(request, tuple(prompts), cuts=tuple(cuts))

    def render_images(self, request: Request) -> Rendering:
        """A request with images: each question's prompt with them, as the package's _image_prompt writes it."""
        rows, pixels = self.place(request)
        state = pictured(state_text(request.state), len(rows))
        prompts = []
        for q in request.questions:
            # the creator's code cuts no state with images: a prompt past the window is refused (Decider.render)
            ids = self.expand_checked(self.tokenizer.encode(prompt_text(state, q)), rows)
            prompts.append(Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), ()))
        return Rendering(request, tuple(prompts), pixels=pixels)

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        out = {}
        for q in rendering.request.questions:
            letters = self.letters[: len(options(q))]
            out[q.id] = last_label_probabilities(hidden.rows(q.id), self.output, letters, self.temperatures[q.type])
        return out

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        """The model's probabilities, and the answer as the creator's server shapes it: a yes/no past its floor."""
        p = tuple(float(x) for x in scores)  # prompt order: a noul's yes, then no
        shaped = systemone_answer(question, p, choice_confidence=above_chance, score_confidence=near_best)
        if question.type == "noul":
            shaped["noul"] = floored(p[0], self.noul_floor)
        return p, shaped


def letter_ids(tokenizer: Tokenizer) -> tuple[int, ...]:
    """The token ids of "A".."Z" as read after a prompt (render.py probes them after a yes/no question's)."""
    return answer_tokens(tokenizer, prompt_text("s", Question("q", "noul", "q", ())), LETTERS)


class Torchcast(AdapterBase):
    api = API_VERSION
    name = "torchcast/decision"
    files = (
        "config.json",
        "decision_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model*.safetensors",
        "model.safetensors.index.json",
    )

    def matches(self, model: ModelDir) -> bool:
        config = model.json("decision_config.json") if model.has("decision_config.json") else None
        return isinstance(config, dict) and config.get("prompt_format") == PROMPT_FORMAT

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        config = model.json("decision_config.json")
        if config.get("attention_mode") != "noncausal_full_attention":
            raise UnsupportedModel(f"{model.path}: expected noncausal full attention")
        weights = ctx.weights(model)
        backbone = ctx.backbone(qwen35, model, weights=weights, causal=False)
        tokenizer = ctx.tokenizer(model)
        images = QwenImages.attach(
            model, ctx, backbone, weights, tokenizer, exif=False, pixels=(MIN_PIXELS, MAX_PIXELS)
        )
        letters = letter_ids(tokenizer)
        if list(letters) != list(config["letter_token_ids"]):
            raise UnsupportedModel(f"{model.path}: the letters' token ids differ from decision_config.json's")
        temperatures = {kind: float(t) for kind, t in config["temperature_by_type"].items()}
        many = config.get("wide_choice", {})  # model.py's defaults where the file names none
        per_pass = int(config.get("max_options_per_pass", len(LETTERS)))
        wide = Wide(per_pass, int(many.get("group", 25)), int(many.get("keep", 3)), float(many.get("residual", 1e-3)))
        fmt = TorchcastFormat(
            tokenizer, backbone.output_matrix, letters, temperatures, float(config["noul_floor"]), images, wide=wide
        )
        return Loaded(
            backbone,
            fmt,
            {
                "attention_mode": "noncausal_full_attention",
                "temperature": temperatures,
                "noul_floor": fmt.noul_floor,
                "model_name": config.get("model_name"),
                "images": fmt.images_detail(),
            },
        )
