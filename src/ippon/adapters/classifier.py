"""Bring your own classifier: a Hugging Face `*ForSequenceClassification` checkpoint of an architecture ippon
covers loads with no adapter code, and runs as transformers runs it.

- **Head.** A decoder's (Llama, Mistral, Qwen2, Qwen3, Gemma 3) is `score` [labels, hidden] at the last token;
  ModernBERT's is its prediction head (dense, activation, LayerNorm) then `classifier`, over the [CLS] token or the
  mean of every token (`classifier_pooling`); BERT's its pooler (dense, tanh) then `classifier`, RoBERTa's and
  XLM-RoBERTa's `classifier.dense`, tanh, `classifier.out_proj`, DistilBERT's `pre_classifier`, ReLU, `classifier`,
  DeBERTa's `pooler.dense` and its activation then `classifier`, over the first token. In fp32, as transformers'
  fp32 path computes it.
- **Labels.** `id2label`. A single-label model's probabilities are the softmax of its logits, a multi-label one's
  (`problem_type: "multi_label_classification"`) or a one-logit one's (a reranker or a scorer) each logit's sigmoid.
- **Input.** The request's state is the text (JSON states as JSON), with the tokenizer's own special tokens, as
  `tokenizer(text)` gives it; a `ippon.json` beside the weights can change that:

      {"text": "{state}", "pair": "{instructions}"}

  `text` (and, for a cross-encoder or an NLI model, a second text `pair`) are templates over {state} and the
  question's {instructions}; questions whose texts are the same share one reading.
- **Questions.** A `choice` gets its labels' probabilities (any of the model's labels, renormalized over those
  asked); a `noul` the probability of the label its instructions name (or of `ippon.json`'s "true_label"), or
  of a one-logit model's sigmoid; a `score` the probabilities of its levels, which are the model's labels in order
  (or `ippon.json`'s "levels"). A multi-label model's labels are independent, so it answers a `noul` per label
  only.

`Decider.classify(texts)` asks the plain question: every label's probability for each text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from string import Formatter
from typing import Any

import mlx.core as mx

from ..adapter import AdapterBase, FormatBase, Hidden, LoadContext, Loaded, Rendering
from ..arch import DECODERS, ENCODERS, Architecture, bert, deberta, dense, gemma3, modernbert
from ..arch.ops import activation
from ..asks import instructions
from ..errors import InvalidRequest, UnsupportedModel
from ..model_dir import API_VERSION, SETTINGS, ModelDir
from ..plan import Prompt, Readout
from ..quant import Matrix, QuantizedMatrix, rows
from ..schema import Question, Request, as_text
from ..tokenizer import Tokenizer

# the architectures whose sequence classifiers' heads `Classifier.load` reads, as arch's tables name them
COVERED = (dense, gemma3, bert, deberta, modernbert)


def classifier_arch(config: Mapping[str, Any]) -> Architecture[Any] | None:
    """The `arch` module of a sequence classifier ippon covers, else None."""
    architectures = config.get("architectures") or []
    if not architectures or not str(architectures[0]).endswith("ForSequenceClassification"):
        return None
    module = {**DECODERS, **ENCODERS}.get(str(config.get("model_type")))
    return module if module in COVERED else None


@dataclass(frozen=True)
class Settings:
    """`ippon.json`: how a request becomes the classifier's input, and what its questions mean."""

    text: str = "{state}"
    pair: str | None = None
    true_label: str | None = None
    levels: tuple[str, ...] | None = None

    @classmethod
    def read(cls, found: Mapping[str, Any]) -> Settings:
        keys = {"format", "text", "pair", "true_label", "levels"}
        if not set(found) <= keys or found.get("format", "classifier") != "classifier":
            raise UnsupportedModel(f'{SETTINGS}: an object of "text", "pair", "true_label" and "levels" (a classifier)')
        levels = found.get("levels")
        return cls(
            text=str(found.get("text", "{state}")),
            pair=None if found.get("pair") is None else str(found["pair"]),
            true_label=found.get("true_label"),
            levels=None if levels is None else tuple(str(x) for x in levels),
        )

    def fill(self, template: str, request: Request, q: Question) -> str:
        try:
            return template.format(state=as_text(request.state, ""), instructions=as_text(q.instructions, ""))
        except (KeyError, IndexError) as e:
            raise UnsupportedModel(f"{SETTINGS}: {template!r} uses {e}, not {{state}} or {{instructions}}") from e


@dataclass
class Head:
    """logits [labels] from the final hidden states a question read: a decoder's last row, or an encoder's pooled
    rows through its prediction head."""

    classifier: mx.array  # [labels, hidden] fp32
    bias: mx.array | None = None
    pooling: str = "last"  # "last", "cls" or "mean"
    dense: mx.array | None = None  # an encoder's prediction head: dense, activation (then a LayerNorm, if any)
    dense_bias: mx.array | None = None
    norm: mx.array | None = None
    norm_bias: mx.array | None = None
    eps: float = 1e-5
    activation: str = "gelu"

    def logits(self, rows: mx.array) -> mx.array:
        h = rows.astype(mx.float32)
        x = h[-1] if self.pooling == "last" else h[0] if self.pooling == "cls" else h.mean(axis=0)
        if self.dense is not None:
            x = x @ self.dense.T
            if self.dense_bias is not None:
                x = x + self.dense_bias
            x = activate(x, self.activation)
            if self.norm is not None:
                x = mx.fast.layer_norm(x, self.norm, self.norm_bias, self.eps)
        y = x @ self.classifier.T
        return y + self.bias if self.bias is not None else y


def activate(x: mx.array, name: str) -> mx.array:
    try:
        return activation(x, name)
    except ValueError as error:
        raise UnsupportedModel(f"classifier activation {name!r} is not supported") from error


@dataclass
class ClassifierFormat(FormatBase):
    tokenizer: Tokenizer
    head: Head
    labels: tuple[str, ...]  # by id
    multi_label: bool = False
    settings: Settings = field(default_factory=Settings)

    def __post_init__(self) -> None:
        self._index = {label: i for i, label in enumerate(self.labels)}

    # -- prompts ------------------------------------------------------------------------------------------------

    def render(self, request: Request) -> Rendering:
        for q in request.questions:
            self._label_rows(q)  # every question asks about labels this model has, before anything is read
        texts: dict[tuple[str, str | None], list[str]] = {}  # (text, pair) -> its questions
        for q in request.questions:
            s = self.settings
            key = (s.fill(s.text, request, q), None if s.pair is None else s.fill(s.pair, request, q))
            texts.setdefault(key, []).append(q.id)
        prompts = []
        for (text, pair), questions in texts.items():
            ids = self.tokenizer.encode_special(text) if pair is None else self.tokenizer.encode_pair(text, pair)
            if not ids:
                raise InvalidRequest("the classifier's input is empty")
            positions = range(len(ids)) if self.head.pooling == "mean" else [0 if self.head.pooling == "cls" else -1]
            prompts.append(Prompt(tuple(ids), tuple(Readout(q, p % len(ids)) for q in questions for p in positions)))
        return Rendering(request, tuple(prompts))

    # -- answers ------------------------------------------------------------------------------------------------

    def _label_rows(self, q: Question) -> list[int]:
        """The model's label ids a question's probabilities come from, in the question's option order."""
        if q.type == "noul":
            if len(self.labels) == 1:
                return [0]
            named = q.instructions if isinstance(q.instructions, str) and q.instructions in self._index else None
            label = named or self.settings.true_label
            if label is None or label not in self._index:
                raise InvalidRequest(
                    f"{q.id}: a yes/no question names one of the model's labels in its instructions "
                    f"({', '.join(self.labels)}), or {SETTINGS} sets true_label"
                )
            return [self._index[label]]
        if q.type == "score":
            levels = self.settings.levels or self.labels
            if len(q.criteria) != len(levels):
                raise InvalidRequest(f"{q.id}: this model scores on {len(levels)} levels: {', '.join(levels)}")
            return [self._index[level] for level in levels]
        if self.multi_label:
            raise InvalidRequest(f"{q.id}: a multi-label model answers a yes/no question per label, naming it")
        missing = [o for o in q.options if o not in self._index]
        if missing:
            raise InvalidRequest(f"{q.id}: not labels of this model: {', '.join(missing)} ({', '.join(self.labels)})")
        return [self._index[o] for o in q.options]

    def scores(self, rendering: Rendering, hidden: Hidden) -> dict[str, mx.array]:
        out = {}
        for q in rendering.request.questions:
            logits = self.head.logits(hidden.rows(q.id))
            rows = mx.array(self._label_rows(q))
            if q.type == "noul":
                independent = self.multi_label or len(self.labels) == 1  # each logit its own probability
                p = mx.sigmoid(logits[rows]) if independent else mx.softmax(logits)[rows]
                out[q.id] = mx.concatenate([p, 1 - p])  # true, false: the question's option order
            else:
                out[q.id] = mx.softmax(logits[rows])
        return out

    def classify_request(self, text: str) -> dict[str, Any]:
        """The System One request that asks for every label's probability of `text`: one choice among the labels,
        or a yes/no question per label for a multi-label or one-logit model."""
        if self.multi_label or len(self.labels) == 1:
            return {
                "state": text,
                "questions": {label: {"type": "noul", "instructions": label} for label in self.labels},
            }
        return {"state": text, "questions": {"label": {"type": "choice", "criteria": list(self.labels)}}}

    def classify_result(self, response: Mapping[str, Any]) -> dict[str, float]:
        """Every label's probability from the response to `classify_request`."""
        answers = response["answers"]
        if self.multi_label or len(self.labels) == 1:
            return {label: float(answers[label]["noul"]) for label in self.labels}
        return {label: float(p) for label, p in answers["label"]["probabilities"].items()}


class Classifier(AdapterBase):
    api = API_VERSION
    name = "generic/classifier"
    generic = True  # matched only when no model's own adapter matches
    files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        SETTINGS,
        "*.safetensors",
        "model.safetensors.index.json",
    )

    compare_arms = ("mlx-lm-model", "mlx-embeddings-model", "transformers")  # the plain ways to run one (`compare.py`)

    def bench_questions(self, model: ModelDir, count: int, variant: int = 0) -> dict[str, dict[str, Any]]:
        """`bench_questions` from the checkpoint's files alone (no weights read)."""
        multi = model.config.get("problem_type") == "multi_label_classification"
        settings = Settings.read(model.settings())
        return bench_questions(labels_of(model.config), multi, settings, count, variant)

    def matches(self, model: ModelDir) -> bool:
        has_tokenizer = model.has("tokenizer.json") or (
            model.has("vocab.txt") and model.config.get("model_type") in bert.FAMILIES
        )
        return classifier_arch(model.config) is not None and has_tokenizer

    def validate(self, model: ModelDir) -> None:
        """Validate labels and input settings from configuration alone."""
        Settings.read(model.settings())
        labels_of(model.config)

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
        config = model.config
        module = classifier_arch(config)
        kind = str(config.get("model_type"))
        if module is None:
            raise UnsupportedModel(f"{model.path}: not a sequence classifier ippon covers")
        weights = ctx.weights(model)
        labels = labels_of(config)

        def head_matrix(name: str) -> mx.array:  # in full precision, even from a quantized checkpoint
            m: Matrix = weights.matrix(name, mx.float32)
            if isinstance(m, QuantizedMatrix):
                return rows(m, mx.arange(m.w.shape[0])).astype(mx.float32)
            assert isinstance(m, mx.array)
            return m.astype(mx.float32)

        def vector(name: str) -> mx.array | None:
            return weights.get(name, mx.float32) if name in weights else None

        settings = Settings.read(model.settings())
        tokenizer = ctx.tokenizer(model)
        if kind in DECODERS:
            backbone: Any = ctx.backbone(module, model, config=config, weights=weights)
            head = Head(head_matrix("score.weight"))
        elif module is deberta:
            sep = tokenizer.token_id("[SEP]") if settings.pair is not None else None
            backbone = ctx.backbone(deberta, model, config=config, weights=weights, sep=sep)
            head = Head(
                head_matrix("classifier.weight"),
                vector("classifier.bias"),
                pooling="cls",
                dense=head_matrix("pooler.dense.weight"),
                dense_bias=vector("pooler.dense.bias"),
                activation=str(config.get("pooler_hidden_act", "gelu")),
            )
        elif module is bert:
            sep = (tokenizer.token_id("[SEP]") or tokenizer.token_id("</s>")) if settings.pair is not None else None
            backbone = ctx.backbone(bert, model, config=config, weights=weights, sep=sep)
            first, second, activation = {
                "bert": ("bert.pooler.dense", "classifier", "tanh"),
                "distilbert": ("pre_classifier", "classifier", "relu"),
            }.get(kind, ("classifier.dense", "classifier.out_proj", "tanh"))
            head = Head(
                head_matrix(second + ".weight"),
                vector(second + ".bias"),
                pooling="cls",
                dense=head_matrix(first + ".weight"),
                dense_bias=vector(first + ".bias"),
                activation=activation,
            )
        else:
            backbone = ctx.backbone(modernbert, model, config=config, weights=weights, root="model.")
            pooling = config.get("classifier_pooling", "cls")
            if pooling not in ("cls", "mean"):
                raise UnsupportedModel(f"classifier_pooling {pooling!r} is not supported")
            head = Head(
                head_matrix("classifier.weight"),
                vector("classifier.bias"),
                pooling=pooling,
                dense=head_matrix("head.dense.weight"),
                dense_bias=vector("head.dense.bias"),
                norm=weights.get("head.norm.weight", mx.float32),
                norm_bias=vector("head.norm.bias"),
                eps=float(config.get("norm_eps", 1e-5)),
                activation=str(config.get("classifier_activation", "gelu")),
            )
        if head.classifier.shape[0] != len(labels):
            raise UnsupportedModel(f"{model.path}: {head.classifier.shape[0]} logits for {len(labels)} labels")
        for label in [settings.true_label, *(settings.levels or ())]:
            if label is not None and label not in labels:
                raise UnsupportedModel(f"{SETTINGS}: {label!r} is not one of the model's labels ({', '.join(labels)})")
        multi = config.get("problem_type") == "multi_label_classification"
        fmt = ClassifierFormat(tokenizer, head, labels, multi, settings)
        return Loaded(backbone, fmt, {"architecture": kind, "labels": len(labels), "pooling": head.pooling})


def bench_questions(
    labels: tuple[str, ...], multi: bool, settings: Settings, count: int, variant: int = 0
) -> dict[str, dict[str, Any]]:
    """`count` questions a classifier answers, for `ippon bench` (another `variant`: other instructions where
    the input reads them): a multi-label model's labels, each a yes/no; a one-logit model's yes/no; else a choice
    among every label."""
    reads = any(
        name == "instructions" for t in (settings.text, settings.pair or "") for _, name, _, _ in Formatter().parse(t)
    )
    out: dict[str, dict[str, Any]] = {}
    for i in range(count):
        q: dict[str, Any]
        if multi:
            q = {"type": "noul", "instructions": labels[(i + variant) % len(labels)]}
        else:
            q = {"type": "noul"} if len(labels) == 1 else {"type": "choice", "criteria": list(labels)}
            if reads:
                q["instructions"] = instructions(i, variant)
        out[f"q{i}"] = q
    return out


def labels_of(config: Mapping[str, Any]) -> tuple[str, ...]:
    """The labels by id, from `id2label` (LABEL_i when it has none)."""
    found = config.get("id2label") or {}
    n = int(config.get("num_labels") or len(found) or 2)
    labels = tuple(str(found.get(str(i), found.get(i, f"LABEL_{i}"))) for i in range(n))
    if len(set(labels)) != n:
        raise UnsupportedModel(f"the model's labels are not distinct: {', '.join(labels)}")
    return labels
