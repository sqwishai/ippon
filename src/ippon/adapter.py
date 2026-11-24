"""The API a decision model plugs in through.

An adapter teaches ippon one model family: which files it needs, how to recognise a downloaded model,
and how to load it into a backbone (its layers, usually one of `ippon.arch`'s) and a format (how its
creator's code turns a System One request into prompts and reads the answers). Adapters register through the
`ippon.adapters` entry-point group, so `pip install` of an adapter package is all a user does; built-in
adapters register the same way.

    class Acme(AdapterBase):
        api, name = 1, "acme/decider"
        files = ("config.json", "tokenizer.json", "*.safetensors", "model.safetensors.index.json")

        def matches(self, model: ModelDir) -> bool:
            return model.json("acme_decider.json") is not None

        def load(self, model: ModelDir, ctx: LoadContext) -> Loaded:
            backbone = ctx.backbone(qwen35, model)  # its dtype and quantization are the load's
            return Loaded(backbone, AcmeFormat(ctx.tokenizer(model), backbone.output_matrix))

    # pyproject.toml of the adapter package:
    # [project.entry-points."ippon.adapters"]
    # "acme/decider" = "acme_ippon:Acme"

The `ModelDir` an adapter is handed (model_dir.py) reads the model's files the same way whatever it was published as,
a folder, a GGUF build or an Ollama build, so an adapter reads its config, JSON and weights without asking which.

How the request runs (one prompt, a shared prefix read once then each question, or every prompt whole) follows
from the prompts the format renders and from the backbone; an adapter never chooses it.

Optional members, for what only some models have, each with its default on `AdapterBase` (subclass it, as the
built-in adapters do, so a new member never breaks an adapter written before it):

- `generic` (True for a fallback that matches whole families, tried after the specific adapters);
- `read_safely` (pickles it names exactly among its `files`, which only ptfile reads: fetched with the first, small
  download, so `matches` can read them; every other pickle is never downloaded);
- `validate(model)` (refuses a model it matched but can't run, before anything else is downloaded);
- `base(model)` (a LoRA adapter's base model, (repo, revision): ippon downloads its BASE_FILES at that revision
  before checking the model fits, so the base's size counts, and hands it to `load` as `ctx.base`, where
  `Weights.with_lora` merges the adapter into it);
- `gguf_rules` and `gguf_check(header, model)` (for a family with GGUF builds: how a build's tensors read under the
  creator's names, `gguf.names.Reads`, and how its metadata differs from the creator's configuration, a list of the
  differences, which `gguf.source.bound` refuses);
- `compare_arms` (the `bench --compare` arms that run its models, `compare_arms.ARMS`) and
  `bench_questions(model, count, variant)` (what its models can be asked, when not any System One question:
  `asks.asker`);
- a format's `tokenizer` (for `literal_state`), and `classify_request(text)` and `classify_result(response)` (for
  `Decider.classify`), with their defaults on `FormatBase`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import mlx.core as mx
import numpy as np

from .arch.base import Backbone
from .errors import UnsupportedModel
from .model_dir import ModelDir
from .plan import Prompt, Readout
from .rounds import Round, Wide
from .schema import Answer, Question, QuestionType, Request
from .tokenizer import Tokenizer
from .weights import Weights

if TYPE_CHECKING:
    from .arch import Architecture
    from .gguf.header import Header
    from .gguf.names import Reads
    from .media.video import Clip

B = TypeVar("B")  # a backbone, as an arch module's `load` returns it


@dataclass(frozen=True)
class LoadContext:
    """What an adapter's `load` gets from ippon: the compute dtype, the bits to quantize weight matrices to
    (None: as they are; pass it to the `arch` loader), and helpers that read only safe files."""

    dtype: mx.Dtype = mx.bfloat16
    bits: int | None = None
    base: ModelDir | None = None  # a LoRA adapter's base model, downloaded as its adapter's `base` names it

    def weights(self, model: ModelDir, files: Sequence[str] | None = None, *, exclude: Sequence[str] = ()) -> Weights:
        """The model's weights under the creator's names, as its source reads them (`model_dir.Source.weights`):
        `files` only, or the model's weight files (its index's, else every *.safetensors) but those in `exclude` (a
        head stored beside the backbone)."""
        return model.source.weights(model, files, exclude)

    def media_weights(self, model: ModelDir) -> Weights:
        """The weights a picture, video or audio path reads: a GGUF build's mmproj file under the creator's names
        (none without one), else the model's own (`weights`)."""
        return model.source.media_weights(model)

    def has_media(self, model: ModelDir, weights: Weights, name: str) -> bool:
        """Whether tensor `name` of a picture or audio path is there to load: in `weights` (the model's), or for a
        GGUF build in its mmproj file, found from the file's header without reading it."""
        return name in model.source.media_names(model, weights)

    def backbone(
        self,
        arch: Architecture[B],
        model: ModelDir,
        *,
        config: Any = None,
        weights: Weights | None = None,
        **options: Any,
    ) -> B:
        """The model's layers, by an `arch` module (`ippon.arch.qwen35`, ...): from its config.json (or `config`, a
        mapping in its form, or the settings an adapter built where the module takes them, as `dense.Config`) and its
        weight files (or `weights`), in this load's dtype and quantized to its bits; `options` are the module's own
        (causal=False, root="model.")."""
        return arch.load(
            str(model.path),
            model.config if config is None else config,
            dtype=self.dtype,
            bits=self.bits,
            weights=self.weights(model) if weights is None else weights,
            **options,
        )

    def tokenizer(self, model: ModelDir, file: str = "tokenizer.json") -> Tokenizer:
        """The model's tokenizer.json as transformers loads it with its tokenizer_config.json; for a BERT-family
        checkpoint with only a WordPiece vocab.txt, the fast tokenizer transformers converts it to."""
        config = model.json("tokenizer_config.json") if model.has("tokenizer_config.json") else {}
        if not model.has(file) and file == "tokenizer.json" and model.has("vocab.txt"):
            return Tokenizer.from_wordpiece(model.file("vocab.txt"), config)
        return Tokenizer.from_file(model.file(file), config)


@dataclass(frozen=True)
class Cut:
    """Tokens a format left out of a prompt so that it fits the model's window, as the creator's code leaves them
    out: which question's prompt ("" for one prompt every question shares), which part of it ("state",
    "instructions", "option <id>", or "prompt" for the end of the whole prompt), and how many tokens of that part
    were dropped and kept."""

    question: str
    part: str
    dropped: int
    kept: int

    def to_json(self) -> dict[str, Any]:
        return {"question": self.question, "part": self.part, "tokens_dropped": self.dropped, "tokens_kept": self.kept}


@dataclass(frozen=True)
class Rendering:
    """A request as a format renders it: its prompts, each with the readouts that answer its questions, the
    format's own per-request data (labels, spans), passed back to `probabilities`, and what it left out to fit the
    model's window (`cuts`): a format that cuts anything says so there, which `Decider` reports in the response
    (`truncated`) or, with truncation="error", refuses."""

    request: Request
    prompts: tuple[Prompt, ...]
    meta: Any = None
    cuts: tuple[Cut, ...] = ()
    round: Round | None = None  # the caller's request, when this is the first round of its wide choices (rounds.py)
    # media decoded to count its rows, kept for its encoding: images' pixels by digest, videos' frames by sha256
    pixels: Mapping[int, np.ndarray] = field(default_factory=dict)
    clips: Mapping[str, Clip] = field(default_factory=dict)
    videos: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Hidden:
    """The final hidden states of a request's readouts, grouped by question in the order they were rendered."""

    by_question: Mapping[str, mx.array]

    @classmethod
    def gather(cls, readouts: Sequence[Sequence[Readout]], hidden: Sequence[mx.array]) -> Hidden:
        """Each prompt's readout rows (`hidden`, [readouts, hidden] per prompt, in its readouts' order) grouped by
        question, each run of one question's rows taken as one slice."""
        rows: dict[str, list[mx.array]] = {}
        for marks, h in zip(readouts, hidden, strict=True):
            start = 0
            for i in range(1, len(marks) + 1):
                if i == len(marks) or marks[i].question != marks[start].question:
                    rows.setdefault(marks[start].question, []).append(h[start:i])
                    start = i
        return cls({q: r[0] if len(r) == 1 else mx.concatenate(r) for q, r in rows.items()})

    def rows(self, question: str) -> mx.array:
        """[readouts, hidden] of `question`."""
        return self.by_question[question]


class Format(Protocol):
    """How a model's creator asks it questions and reads its answers. `FormatBase` gives the defaults."""

    wide: Wide | None  # how its creator answers a choice wider than one pass (rounds.py), or None

    @property
    def tokenizer(self) -> Tokenizer | None:
        """Its tokenizer, for `literal_state` and `inspect`'s text; None for a format without one."""
        ...

    @property
    def window(self) -> int | None:
        """The creator's window, a prompt's tokens at most, where its code sets one below what the backbone reads
        (`FormatBase.window`): the decider refuses a longer prompt after rendering it."""
        ...

    @window.setter
    def window(self, tokens: int) -> None:
        """Another window in place of the creator's (`Decider.load(window=)`), which a format reads from then on."""
        ...

    @property
    def media(self) -> frozenset[str]:
        """What media a request may carry for this model ("images", "audio", "spoken", "videos"); the decider
        refuses the rest."""
        ...

    @property
    def anchors(self) -> frozenset[str]:
        """Which media fields this format places at inline <kind:N> anchors in the state (frozenset() for formats that
        prepend media ahead of text, so literal '<image:2>' in a ticket is not refused)."""
        ...

    def render(self, request: Request) -> Rendering:
        """The request as the creator's code renders it: prompts with a readout for every question. InvalidRequest for
        what the model can't be asked."""
        ...

    def scores(self, rendering: Rendering, hidden: Hidden) -> Mapping[str, mx.array]:
        """Per question, whatever the format reads its answer from (option probabilities, or per-row values it
        combines): evaluated together, then passed to `answer` as Python lists."""
        ...

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        """From a question's scores: its probabilities in question.options order, and its answer in the shape the
        creator's server returns."""
        ...

    def encode_media(self, renderings: Sequence[Rendering]) -> dict[int, mx.array]:
        """Every medium the renderings' prompts hold, encoded, by the digest its placeholder tokens name it by."""
        ...

    def input_tokens(self, rendering: Rendering) -> int:
        """The input tokens the creator's server counts for this request."""
        ...

    def usage(self, rendering: Rendering) -> dict[str, int]:
        """The usage the creator's server reports for this request: {"input_tokens": ..., "output_tokens": ...}."""
        ...

    def most_options(self, kind: QuestionType) -> int | None:
        """The most options (a score's levels) a question of `kind` may have; None for no fixed count."""
        ...

    def classify_request(self, text: str) -> dict[str, Any]:
        """A classifier's System One request asking for `text`'s labels (`Decider.classify`); UnsupportedModel for a
        format that is not a classifier's."""
        ...

    def classify_result(self, response: Mapping[str, Any]) -> dict[str, float]:
        """Every label's probability, from the response to `classify_request`."""
        ...


class FormatBase:
    """Defaults for a format whose scores are the option probabilities: the plain System One answer, a usage of
    every prompt's tokens in, none out, and no media (images, audio) taken."""

    # how its creator answers a choice wider than one pass, in rounds (rounds.py); None when it never splits one
    wide: Wide | None = None
    # the creator's window, a prompt's tokens at most, where its code sets one below what the backbone reads (a server's
    # context, a training budget); None when the backbone's positions are the only limit. The decider refuses a longer
    # prompt (`Decider.render`), so a format that cuts nothing to fit needn't check it. Decider.load's `window` replaces
    # it, past it unverified.
    window: int | None = None
    # its tokenizer, which every format sets (None for one without): no default here, so that a dataclass format's
    # field stays one without a default
    tokenizer: Tokenizer | None

    @property
    def media(self) -> frozenset[str]:
        """What media a request may carry for this model ("images", "audio", "spoken", "videos"), which may depend on
        the build (a GGUF file without its mmproj reads text only); the decider refuses the rest. None by default."""
        return frozenset()

    @property
    def anchors(self) -> frozenset[str]:
        """Media fields placed at inline <kind:N> anchors in the state. Empty by default: formats that prepend media
        ahead of the state treat literal '<image:2>' in text as ordinary text."""
        return frozenset()

    def most_options(self, kind: QuestionType) -> int | None:
        """The most options (a score's levels) a question of `kind` may have: as many as the model has labels or
        codes to answer with, which the decider holds every question to. None, the default, for a model whose
        answers are not a fixed set of labels."""
        return None

    def encode_media(self, renderings: Sequence[Rendering]) -> dict[int, mx.array]:
        """Every medium the renderings' prompts hold, encoded: its rows [rows, hidden] by the first 48 bits of its
        digest, as its placeholder tokens name it (arch/common.media_id). None by default: a format that takes
        media encodes them here (its tower loaded with the model), before the pass reads the placeholders."""
        return {}

    def answer(self, question: Question, scores: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
        probabilities = tuple(float(p) for p in scores)
        return probabilities, Answer(question, probabilities).to_systemone()

    def input_tokens(self, rendering: Rendering) -> int:
        """Every prompt's tokens; override for a creator that counts a shared prefix once."""
        return sum(len(p.tokens) for p in rendering.prompts)

    def usage(self, rendering: Rendering) -> dict[str, int]:
        return {"input_tokens": self.input_tokens(rendering), "output_tokens": 0}

    def classify_request(self, text: str) -> dict[str, Any]:
        raise UnsupportedModel("classify() is for classifiers; ask this model questions with decide()")

    def classify_result(self, response: Mapping[str, Any]) -> dict[str, float]:
        raise UnsupportedModel("classify() is for classifiers; ask this model questions with decide()")


@dataclass(frozen=True)
class Loaded:
    backbone: Backbone
    format: Format
    details: Mapping[str, Any] = field(default_factory=dict)  # shown by `ippon doctor` (revision, temperatures)


class Adapter(Protocol):
    api: int  # model_dir.API_VERSION it was written for
    name: str  # "creator/family", as its entry point is named

    @property
    def files(self) -> tuple[str, ...]:
        """Glob patterns of the files it needs (safetensors, JSON and text, and its `read_safely`)."""
        ...

    @property
    def generic(self) -> bool: ...

    @property
    def read_safely(self) -> tuple[str, ...]: ...

    @property
    def gguf_rules(self) -> Reads | None: ...

    @property
    def compare_arms(self) -> tuple[str, ...]: ...

    def matches(self, model: ModelDir) -> bool: ...

    def load(self, model: ModelDir, ctx: LoadContext) -> Loaded: ...

    def validate(self, model: ModelDir) -> None: ...

    def base(self, model: ModelDir) -> tuple[str, str] | None: ...

    def gguf_check(self, header: Header, model: ModelDir) -> list[str]: ...

    def bench_questions(self, model: ModelDir, count: int, variant: int = 0) -> dict[str, dict[str, Any]] | None: ...


class AdapterBase:
    """Defaults for an adapter's optional members (the module docstring says what each is for): a model's own
    adapter, no pickles, nothing to validate past `matches`, weights of its own, no GGUF builds, no compare arms, and
    any System One question."""

    generic: bool = False
    read_safely: tuple[str, ...] = ()
    gguf_rules: Reads | None = None
    compare_arms: tuple[str, ...] = ()

    def validate(self, model: ModelDir) -> None:
        return None

    def base(self, model: ModelDir) -> tuple[str, str] | None:
        return None

    def gguf_check(self, header: Header, model: ModelDir) -> list[str]:
        return []

    def bench_questions(self, model: ModelDir, count: int, variant: int = 0) -> dict[str, dict[str, Any]] | None:
        return None
