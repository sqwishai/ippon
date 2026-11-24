"""H2O-Lightning-4B (H2O.ai, Apache-2.0): a Qwen3.5-4B decision model answering each question with the label of an
option, read as the checkpoint's own server reads it (h2o_lightning_shim.py with serve_config.json), checked token for
token against it:

- every question is its own prompt: Qwen's chat template around serve_config.json's system prompt and the user turn
  "record: <state>\\nquestion: <instructions>\\noptions:\\nA) <key>: <text>\\n...", the assistant's turn opened with
  thinking off and the prefill "Answer:";
- the record is a text state as written, any other state as compact JSON; instructions as written, or as JSON with
  `indent=1` when they are not text, without leading whitespace ("Answer the question below." when there are none);
  a noul shows "true" then "false" ("the statement holds", "it does not" when undescribed), a choice its keys with
  their descriptions in the caller's order (a key alone repeats itself), a score its levels "0".."n-1";
- the answer is read at the last token (the prefill's ":"): the final hidden state times the output matrix's rows
  of the options' labels (" A", " B", ...; 255 of them, letters, Greek letters, then common subwords), in fp32,
  divided by the question type's temperature, softmax; a yes/no answer whose max(P, 1 - P) is under
  serve_config.json's noul_floor moves to the floor on its side, as the server answers (0.801 at the verified
  revisions; 0, off, in H2O's later revisions); a score's confidence is its most likely level's probability.

The full-attention layers read the whole prompt both ways by default (the server's `two_way.default`), so every
prompt is read whole; a `ippon.json` setting `{"attention_mode": "causal"}` reads them one way, the questions sharing
the record. The creator's Q8_0 GGUF build (h2oai/h2o-lightning-4b-gguf) is read one way, as its card says llama.cpp
reads it, with the server's prompt, tokenizer, floor and confidences; llama.cpp's own `h2o-lightning` type leaves
the floor to the client, scores the confidence otherwise, quotes special-token text and skips the tokenizer's NFC.

A record too long for a question's prompt is fitted as the server fits it, never refused: cut to the passages the
creator's BM25 selection (h2o_select.py) finds most relevant to the question and its options, else to its head and
tail, and under 64 tokens of room left out with the question's own text cut (`H2OFormat.fit`); `truncated` says so.

Images (`Request.images`, as many as the context holds; serve_config.json's image.max_images, 4, is the server's) go
ahead of the user turn's text, one
`<|vision_start|><|image_pad|><|vision_end|>` each, the image token expanded to the image's rows, as the server sends
them to vLLM's chat completions (the record and the question as without images). An image is decoded as the h2oai/vllm
build serve.sh pins loads it (b05cc29, vLLM 0.30: turned by its EXIF orientation, transparency composited on white),
resized within image.max_pixels (1,048,576 at the verified revision, 1,638,400 in later revisions) and the processor's
floor of 65,536 pixels, and read by the checkpoint's
vision tower (adapters/qwen.py); every type of question is then answered at image.temperature (two_way's
image_temperature two-way), the yes/no floor as without images. The GGUF build's vision tower is its mmproj file
(mmproj-lightning-f16.gguf), read as llama.cpp's --mmproj reads it; without it beside the GGUF file, text only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx

from ..adapter import AdapterBase, Cut, Hidden, LoadContext, Loaded, Rendering
from ..arch import qwen35
from ..errors import UnsupportedModel
from ..formulas import above_chance, floored, top_probability
from ..gguf.header import Header, close
from ..gguf.names import Reads, Rule, qwen3vl_media_rules, qwen35_rules
from ..gguf.source import qwen35_differences
from ..media.qwen import QwenImages
from ..model_dir import API_VERSION, ModelDir
from ..plan import Prompt, Readout, state_end_marks
from ..readout import answer_tokens, last_label_probabilities
from ..schema import Question, QuestionType, Request, as_text, systemone_answer
from ..tokenizer import Tokenizer
from . import h2o_select
from .qwen import MIN_PIXELS, VISION, QwenImageFormat, chat

MODEL = "h2oai/h2o-lightning-4b"
DEFAULT_INSTRUCTIONS = "Answer the question below."
NOUL_TEXTS = {"true": "the statement holds", "false": "it does not"}
MAX_LENGTH = 40960 - 16  # the server's vLLM context (serve.sh's MAX_LEN) less its CONTEXT_SLACK
MIN_STATE_BUDGET = 64  # the server's: under this many tokens of room a question is asked without the record
PIXELS_PER_IMAGE_TOKEN = 32 * 32  # the server's bound on an image's tokens when it fits a record: 16-px patches, 2 x 2
ATTENTION_MODES = ("bidirectional", "causal")
WHITE = (255, 255, 255)  # vLLM's background for a transparent image (ImageMediaIO's rgba_background_color default)


def record(state: Any) -> str:
    """The state as the server writes it after "record: ": text as it is, anything else as compact JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def instructions(q: Question) -> str:
    """The instructions as the server writes them: text as it is, anything else as JSON with indent=1, without
    leading whitespace; the default when that leaves nothing."""
    value = q.instructions if q.instructions is not None else ""
    text = (value if isinstance(value, str) else json.dumps(value, indent=1)).lstrip()
    return text or DEFAULT_INSTRUCTIONS


def options(q: Question) -> list[tuple[str, str]]:
    """(key, text) of each option in prompt order, which is q.options order: a noul's true and false, a choice's
    keys, a score's levels "0".."n-1"."""
    if q.type == "noul":
        given = dict(q.criteria)
        return [
            (key, as_text(given[key]) if given.get(key) not in (None, "") else NOUL_TEXTS[key]) for key in NOUL_TEXTS
        ]
    if q.type == "score":
        return [(key, as_text(level)) for key, level in q.criteria]
    return [(key, key if description is None else as_text(description)) for key, description in q.criteria]


def prompt_text(
    system: str,
    prefill: str,
    labels: Sequence[str],
    state: str,
    q: Question,
    images: int = 0,
    parts: tuple[str, list[tuple[str, str]]] | None = None,
) -> str:
    """The prompt as serve_config.json's `system`, `prefill` and `labels` write it, or with `parts` its instructions
    and options as `H2OFormat.fit_question` cut them."""
    said, shown = parts if parts is not None else (instructions(q), options(q))
    lines = [f"{label}) {key}: {text}" for label, (key, text) in zip(labels, shown, strict=False)]
    user = f"record: {state}\nquestion: {said}\noptions:\n" + "\n".join(lines)
    return chat(system, user, images) + prefill


def label_ids(tokenizer: Tokenizer, system: str, prefill: str, labels: Sequence[str]) -> tuple[int, ...]:
    """Each label's token where the answer is read (" A", " B", ...), as the server probes them after a prompt."""
    probe = Question("q", "choice", "q", tuple((f"o{i}", None) for i in range(len(labels))))
    return answer_tokens(tokenizer, prompt_text(system, prefill, labels, "s", probe), [" " + label for label in labels])


@dataclass
class H2OFormat(QwenImageFormat):
    tokenizer: Tokenizer
    output: Any  # the output matrix (tied to the input embedding), whose label rows score the options
    system: str
    prefill: str
    labels: tuple[str, ...]  # each option's label in its line ("A", "B", ...)
    label_ids: tuple[int, ...]  # each label's token where the answer is read (" A", " B", ...)
    temperatures: dict[str, float]
    noul_floor: float
    causal: bool = False
    window: int = MAX_LENGTH  # the creator's window: a prompt's tokens at most (Decider.load's `window` replaces it)
    images: QwenImages | None = None  # None: a build without the vision tower (the GGUF), text only
    image_temperature: float = 1.0
    image_bound: int = (1 << 24) // PIXELS_PER_IMAGE_TOKEN + 2  # an image's tokens at most, as the server counts them

    def most_options(self, kind: QuestionType) -> int | None:
        return len(self.labels)

    def prompt_text(
        self, state: str, q: Question, images: int = 0, parts: tuple[str, list[tuple[str, str]]] | None = None
    ) -> str:
        """The prompt (`prompt_text`), or with `parts` its instructions and options as `fit_question` cut them."""
        return prompt_text(self.system, self.prefill, self.labels, state, q, images, parts)

    def fit(
        self, request: Request, q: Question, full: str
    ) -> tuple[str, tuple[str, list[tuple[str, str]]] | None, list[Cut]]:
        """A question whose prompt is over the context, fitted as the server fits it (h2o_lightning_shim.py,
        `fit_states`): the record cut to the room the question's own prompt leaves (the context less the prompt
        without the record and its images' tokens at most) by the creator's BM25 selection of its passages
        (h2o_select.py), or its head and tail where no passage matches; under 64 tokens of room the record left out
        and the question's instructions, then its option descriptions, then its option names cut (`fit_question`).
        Returns the record, the cut parts (None when they are whole) and what was cut."""
        img = len(request.images) * self.image_bound
        n_state = len(self.tokenizer.encode(full))
        room = self.window - len(self.tokenizer.encode(self.prompt_text("", q))) - img
        if n_state <= room:
            return full, None, []
        if room < MIN_STATE_BUDGET:
            parts, cuts = self.fit_question(q, img)
            return "", parts, [Cut(q.id, "state", n_state, 0), *cuts]
        budget, record = room, self.select(request.state, full, q, room)
        for _ in range(3):  # the exact count: the record's seams with the prompt can add a token
            n = len(self.tokenizer.encode(self.prompt_text(record, q))) + img
            if n <= self.window:
                break
            budget -= (n - self.window) + 8
            if budget < MIN_STATE_BUDGET:
                record = ""
                break
            record = self.select(request.state, full, q, budget)
        kept = len(self.tokenizer.encode(record)) if record else 0
        return record, None, [Cut(q.id, "state", n_state - kept, kept)]

    def select(self, state: Any, full: str, q: Question, budget: int) -> str:
        """The record cut to `budget` tokens for `q`: the creator's selection on the state as sent, else (no passage
        matches, or the record can't be split to fit) its head and tail."""
        said = q.instructions if q.instructions is not None else ""
        given = dict(q.criteria)
        if q.type == "score":
            criteria: Any = [level for _, level in q.criteria]
        elif q.type == "choice" and all(d is None for d in given.values()):
            criteria = list(given)  # sent as a list of labels
        else:
            criteria = given or None
        try:
            return str(
                h2o_select.select_state(
                    state,
                    said if isinstance(said, str) else json.dumps(said, indent=1),
                    criteria,
                    budget,
                    self.tokenizer.encode,
                )
            )
        except h2o_select.EvidenceSelectionUnsupported:
            return self.cut(full, budget)

    def cut(self, text: str, most: int) -> str:
        """`text` cut to `most` tokens by its head and tail around a "\n...\n" line (the server's `cut_text`)."""
        if not most or len(text.encode("utf-8", "ignore")) <= most:
            return text
        ids = self.tokenizer.encode(text)
        if len(ids) <= most:
            return text
        half = max(1, most // 2)
        return self.tokenizer.decode(ids[:half]) + "\n...\n" + self.tokenizer.decode(ids[-half:])

    def fit_question(self, q: Question, img: int) -> tuple[tuple[str, list[tuple[str, str]]] | None, list[Cut]]:
        """A question too long for the context without its record: its instructions, then each option description,
        then each option name (in the prompt only) cut by head and tail until it fits (the server's `fit_question`)."""
        said, shown = instructions(q), options(q)
        names, texts = [k for k, _ in shown], [t for _, t in shown]
        cuts: list[Cut] = []

        def over() -> int:
            parts = (said, list(zip(names, texts, strict=True)))
            return len(self.tokenizer.encode(self.prompt_text("", q, parts=parts))) + img - self.window

        if over() <= 0:
            return None, cuts
        for what in ("instructions", "option descriptions", "option names"):
            excess = over()
            if excess <= 0:
                break
            if what == "instructions":
                k = len(self.tokenizer.encode(said))
                said = self.cut(said, max(16, k - excess - 8))
                cuts.append(
                    Cut(q.id, "instructions", k - len(self.tokenizer.encode(said)), len(self.tokenizer.encode(said)))
                )
                continue
            group = texts if what == "option descriptions" else names
            total = sum(len(self.tokenizer.encode(x)) for x in group)
            per = max(4, (total - excess - 8) // max(1, len(group)))
            group[:] = [self.cut(x, per) for x in group]
            cuts.append(
                Cut(
                    q.id,
                    what,
                    total - sum(len(self.tokenizer.encode(x)) for x in group),
                    sum(len(self.tokenizer.encode(x)) for x in group),
                )
            )
        return (said, list(zip(names, texts, strict=True))), cuts

    def render(self, request: Request) -> Rendering:
        rows, pixels = self.place(request)
        # the images' rows beyond their one token each: they come before the record, so its mark moves on by them
        extra = sum(len(r) for r in rows) - len(rows)
        full = record(request.state)
        prompts, cuts = [], []
        for q in request.questions:
            state, parts = full, None
            text = self.prompt_text(state, q, len(rows))
            if len(text.encode("utf-8", "ignore")) + len(request.images) * self.image_bound > self.window:
                state, parts, cut = self.fit(request, q, full)  # bytes bound tokens: under it, it fits as it is
                cuts += cut
                text = self.prompt_text(state, q, len(rows), parts)
            ((ids, offsets),) = self.tokenizer.encode_offsets([text])
            ids = self.expand_checked(ids, rows, "record")
            marks = state_end_marks(text, offsets, text.index("record: ") + len("record: ") + len(state))
            marks = tuple(m + extra for m in marks)
            if rows:  # the images' end too, where a request with one more image (after them) parts from this one
                end = text.rindex(VISION) + len(VISION)
                marks = (sum(1 for _, stop in offsets if stop <= end) + extra, *marks)
            prompts.append(Prompt(tuple(ids), (Readout(q.id, len(ids) - 1),), marks if self.causal else ()))
        return Rendering(request, tuple(prompts), cuts=tuple(cuts), pixels=pixels)

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        out = {}
        pictured = bool(rendering.request.images)
        for q in rendering.request.questions:
            temperature = self.image_temperature if pictured else self.temperatures[q.type]
            out[q.id] = last_label_probabilities(
                hidden.rows(q.id), self.output, self.label_ids[: len(q.options)], temperature
            )
        return out

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        p = tuple(float(x) for x in scores)  # q.options order: a noul's true, then false
        legend = {key: as_text(level) for key, level in question.criteria} if question.type == "score" else None
        shaped = systemone_answer(
            question, p, choice_confidence=above_chance, score_confidence=top_probability, legend=legend
        )
        if question.type == "noul":  # the server's floor shapes the answer, not the model's probabilities
            shaped["noul"] = floored(p[0], self.noul_floor)
        return p, shaped

    def input_tokens(self, rendering: Rendering) -> int:
        """Every question's prompt, their shared start counted once, as the server bills them."""
        prompts = [p.tokens for p in rendering.prompts]
        if len(prompts) < 2:
            return sum(len(t) for t in prompts)
        shared = 0
        for column in zip(*prompts, strict=False):
            if any(t != column[0] for t in column[1:]):
                break
            shared += 1
        return sum(len(t) for t in prompts) - shared * (len(prompts) - 1)


def vision_named(model: ModelDir) -> ModelDir:
    """The model with its vision_config as transformers reads it: H2O's config.json names its vision tower's
    model_type "qwen3_5", the whole model's, and Qwen3_5Config builds that block as a Qwen3_5VisionConfig whatever it
    names (as the creator's vLLM and transformers load it)."""
    v = model.config.get("vision_config")
    if model.config.get("model_type") != "qwen3_5" or not isinstance(v, dict) or v.get("model_type") != "qwen3_5":
        return model
    return replace(model, config={**model.config, "vision_config": {**v, "model_type": "qwen3_5_vision"}})


def attention_mode(model: ModelDir, default: str) -> str:
    mode = model.settings().get("attention_mode", default)
    if mode not in ATTENTION_MODES:
        raise UnsupportedModel(f"{model.path}: attention_mode is one of {', '.join(ATTENTION_MODES)}")
    return str(mode)


def gguf_tensors(header: Header) -> tuple[list[Rule], int]:
    """A GGUF build's Qwen3.5, the output tied to the embedding (no output tensor), and its blocks, without the
    speculative decoding layer its block count includes (nextn_predict_layers) and the file leaves out."""
    return qwen35_rules(header), int(header.get("block_count")) - int(header.get("nextn_predict_layers", 0))


class H2OLightning(AdapterBase):
    api = API_VERSION
    name = "h2oai/h2o-lightning"
    gguf_rules = Reads(gguf_tensors, qwen3vl_media_rules)
    files = (
        "config.json",
        "serve_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model*.safetensors",
        "model.safetensors.index.json",
    )

    def matches(self, model: ModelDir) -> bool:
        config = model.json("serve_config.json") if model.has("serve_config.json") else None
        return isinstance(config, dict) and config.get("model") == MODEL

    def gguf_check(self, header: Header, model: ModelDir) -> list[str]:
        """A GGUF build's Qwen3.5 against the config, and its attention (one way, as its card says llama.cpp reads
        it), decision type, labels and temperatures against serve_config.json."""
        serve = model.json("serve_config.json") or {}
        differ = qwen35_differences(header, model.config)
        if not header.get("attention.causal", True):
            differ.append("non-causal attention against the creator's one-way GGUF")
        if header.get("decision.type") != "h2o-lightning":
            differ.append(f"decision type {header.get('decision.type')!r}, not h2o-lightning")
        if header.get("decision.labels") != [" " + label for label in serve.get("labels", [])]:
            differ.append("its labels differ from serve_config.json's")
        for kind, t in (serve.get("temperature") or {}).items():
            if not close(header.get(f"decision.temperature.{kind}", 0.0), t):
                differ.append(f"{kind} temperature {header.get(f'decision.temperature.{kind}')} against {t}")
        return differ

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        config = model.json("serve_config.json")
        prompt = config["prompt"]
        if (prompt.get("layout"), prompt.get("chat_template"), prompt.get("enable_thinking")) != ("plain", True, False):
            raise UnsupportedModel(f"{model.path}: expected the plain layout in the chat template, thinking off")
        # a GGUF build one-way, as its card says llama.cpp reads it and its metadata declares (`gguf_check`)
        mode = (
            "causal"
            if model.source.kind == "gguf"
            else attention_mode(model, config.get("two_way", {}).get("default", "causal"))
        )
        calibration = config["two_way"] if mode == "bidirectional" else config
        weights = ctx.weights(model)
        backbone = ctx.backbone(qwen35, model, weights=weights, causal=mode == "causal")
        tokenizer = ctx.tokenizer(model)
        image = config.get("image") or {}
        temperature = float(image.get("temperature", 1.0))  # one-way; two-way's own when its block names one
        images = None
        ceiling = int(image.get("max_pixels") or 0) or None
        if image:  # the server reads images only when serve_config.json has an image block
            images = QwenImages.attach(
                vision_named(model),
                ctx,
                backbone,
                weights,
                tokenizer,
                exif=True,
                pixels=(MIN_PIXELS, ceiling) if ceiling else None,
                matte=WHITE,
            )
        labels = tuple(str(label) for label in config["labels"][: int(config["max_options"])])
        system, prefill = str(prompt["system"]), str(prompt["prefill"])
        fmt = H2OFormat(
            tokenizer,
            backbone.output_matrix,
            system,
            prefill,
            labels,
            label_ids(tokenizer, system, prefill, labels),
            {kind: float(t) for kind, t in calibration["temperature"].items()},
            float(calibration["noul_floor"]),
            causal=mode == "causal",
            images=images,
            image_temperature=temperature
            if mode == "causal"
            else float(config["two_way"].get("image_temperature", temperature)),
            image_bound=(ceiling or 1 << 24) // PIXELS_PER_IMAGE_TOKEN + 2,
        )
        return Loaded(
            backbone,
            fmt,
            {
                "attention_mode": mode,
                "temperature": fmt.temperatures,
                "noul_floor": fmt.noul_floor,
                "labels": len(labels),
                "images": fmt.images_detail(),
            },
        )
