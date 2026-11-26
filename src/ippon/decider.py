"""The public entry point: a decision model loaded once, answering requests.

    from ippon import Decider, YesNo, Choice

    decider = Decider.load("LiquidAI/d1-2B")                   # a Hub id or a downloaded directory
    result = decider.decide(state="My card was charged twice for one order.", questions={
        "refund": YesNo("Is a refund asked for?"),
        "team": Choice("Which team should handle it?", ["billing", "sales"]),
    })
    result["refund"].p_yes, result["team"].top            # typed answers
    result.response                                       # the System One body the model's own server returns

    decider.systemone({"state": ..., "questions": {...}})  # System One in, System One out

Call it from one thread at a time: tracing MLX's compiled functions is not safe across threads.
"""

from __future__ import annotations

import gc
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import (
    admission,
    asks,
    choices,
    decide_api,
    drift,
    locate,
    names,
    provenance,
    quant,
    quantization,
)
from .adapter import Hidden, LoadContext, Loaded, Rendering
from .answers import DEFAULT_SIZE as ANSWERS_KEPT
from .answers import Answers
from .arch.base import Layered
from .arch.cache import Attention
from .arch.common import MediaRows, is_placeholder, media_digest
from .arch.ops import Ops, for_profile
from .asks import Asker, standard
from .errors import (
    ContextTooLong,
    DecisionError,
    InsufficientMemory,
    InvalidRequest,
    NonFiniteAnswer,
    UnsupportedModel,
    memory_errors,
)
from .media.decode import check_images
from .options import PERIOD_S, Reads, Strategy, Truncation, validate_load_options
from .plan import DEFAULT_CHUNK
from .precision import DTYPES, FP16_CEILING, PeakOps
from .prefixes import Prefixes
from .profiles import Profile, resolve
from .quantization import NAMES, PRECISIONS, listing
from .quantization import get as get_precision
from .questions import Builder, request_body, with_media
from .residency import KeepAwake, pin
from .rounds import SEPARATOR, answered, first_round
from .runtime import phase, read_requests
from .schema import MEDIA_FIELDS, Answer, Request, Result, as_text, check_anchors, parse_request
from .tokenizer import Tokenizer, literal, placeholder

# A state past this many characters for each of the model's positions (and past MOST_STATE_CHARS, so a model with a
# short window still cuts a long text as its creators' code does) is refused before it is tokenized: text averages
# about 4 a token, and tokenizing 120 MiB of it (18 million tokens) took 40 s and 14 GB the server didn't get back
CHARS_PER_POSITION = 8
# what a pass needs beside its cache (activations, the GPU's own buffers), kept free when `tokens_here` counts the room
READING_MARGIN = 2 << 30
MOST_STATE_CHARS = 1 << 22
PREFIX_BUDGET = 4 << 30  # bytes of kept prefixes at most; a sixteenth of memory on smaller Macs


def quantized(backbone: Layered) -> int:
    """How many of the backbone's layer weight matrices (its `linears`) are quantized."""
    return sum(
        isinstance(layer.get(key), quant.QuantizedMatrix) for layer in backbone.layers for key in backbone.linears
    )


def packed_formats(backbone: Layered) -> Iterator[tuple[str, int]]:
    """The (mode, bits) of every packed matrix and expert bank in the backbone's layers."""
    for layer in backbone.layers:
        for value in layer.values():
            parts = value.parts if isinstance(value, quant.JoinedMatrix | quant.JoinedStacked) else (value,)
            for part in parts:
                if isinstance(part, quant.QuantizedMatrix) or (
                    isinstance(part, quant.Stacked) and part.scales is not None
                ):
                    yield part.mode, part.bits


class Decider:
    def __init__(
        self,
        loaded: Loaded,
        *,
        name: str = "",
        profile: Profile | None = None,
        chunk: int = DEFAULT_CHUNK,
        strategy: Strategy = "auto",
        prefix_budget: int | None = None,
        literal_state: bool = Reads.literal_state,
        truncation: Truncation = Reads.truncation,
        window: int | None = Reads.window,
        answer_cache: int = 0,
        resident: int | None = None,
    ):
        """The model `loaded` answering, with `load`'s options as `load` checked them; `resident`, the bytes its weights
        hold (`admission.admit`), lets it work out the room for readings beside them (`tokens_here`)."""
        self.truncation = truncation  # what to do with input that doesn't fit the window (`render`)
        self.backbone, self.format, self.details = loaded.backbone, loaded.format, dict(loaded.details)
        self.literal_state = literal_state  # special tokens' strings in a state read as text (`render`)
        if literal_state and self.tokenizer is None:
            raise UnsupportedModel(f"{type(self.format).__name__}: literal_state needs the format's tokenizer")
        self.name, self.chunk, self.strategy = name, chunk, strategy
        if window is not None:
            self._widen(window)
        self.closed = False  # `close` unloaded the model
        self.pinned = False  # `load` wired its memory (`residency.pin`)
        self.profile = profile or resolve()
        grows = prefix_budget is None  # a default budget: the newest entry may outgrow it (`Prefixes`)
        if prefix_budget is None:
            prefix_budget = default_prefix_budget(self.profile.device.memory_bytes if self.profile.device else None)
        shares = self.backbone.causal or self.backbone.block  # a prefix it reads once, which can be kept
        self.prefixes = Prefixes(prefix_budget, grows) if shares and prefix_budget > 0 else None
        self.answers = Answers(answer_cache) if answer_cache > 0 else None  # identical requests answered again
        self._uncached = False  # within `uncached`: every request is read, none answered from `answers`
        # how its norms, GLUs and matmuls run on this chip (arch/ops.py), each matmul's kernel chosen for its shape
        # from the chip (choices.py)
        tiles = choices.attach(self.backbone, self.profile)
        self.backbone.ops = for_profile(self.profile, tiles)
        self.details["kernels"] = choices.summary(self.profile)
        if resident is not None:
            self.details["resident_estimate"] = resident  # bytes, from the tensors' headers (`admission.admit`)
        # the longest prompt this Mac's GPU holds the reading of beside the weights and the kept prefixes (`room`)
        self.tokens_here = self._room(resident)
        self.lock = threading.Lock()  # held while the model runs (KeepAwake waits for it)
        self.keep_awake: KeepAwake | None = None
        self.fallback: Callable[[], Decider] | None = None  # the model in bf16, should fp16 overflow on a request
        # the questions it asks itself (warm-up, bench) when not any System One question: its adapter's, set by `load`
        self.own_questions: Asker | None = None

    @classmethod
    @memory_errors
    def load(
        cls,
        model: str | Path,
        *,
        adapter: str | None = None,
        accept_license: bool = False,
        precision: str = "auto",
        profile: str | None = None,
        switches: str | None = None,
        chunk: int = DEFAULT_CHUNK,
        strategy: Strategy = "auto",
        warm_up: bool = True,
        pin_memory: bool = True,
        keep_awake: float | None = PERIOD_S,
        prefix_budget: int | None = None,
        settings: Mapping[str, Any] | None = None,
        literal_state: bool = Reads.literal_state,
        truncation: Truncation = Reads.truncation,
        window: int | None = Reads.window,
        answer_cache: int = ANSWERS_KEPT,
        freeze_objects: bool = False,
    ) -> Decider:
        """A Hub id (`org/name`, `org/name@revision`), a downloaded directory, or a model's short name (`kettei-core`,
        `kettei-core:q8` at 8 bits: names.py), a GGUF file, or an `ollama://name:tag` build. `precision` is the compute
        dtype, unless a short name carries it:
        "auto" (the default) computes in fp16 when the model's activations, measured on the warm-up requests, stay
        below two thirds of fp16's range, and in bf16 otherwise (`precision.py`). Check the chosen build against
        its creator's reference; published per-model bounds are in docs/accuracy.md. Or bf16, fp16, fp32 (for checking
        accuracy), or q8 for 8-bit weight matrices, or q6 and q4, when asked for, for fewer. Quantized weights
        compute in the dtype auto would choose. A model below 8 bits, quantized here or published so, warns of its
        drift from the creator's answers (`drift.LowBitWarning`, `details["drift"]`, each response's `warnings`).
        `decider.precision` says what it computes in, `decider.details["weights"]` how its weights are stored.
        `profile` forces a chip profile ("m1".."m5"), and `switches` sets single operations' paths
        ("add_norm=mlx,gemm=mlx", `profiles.Switches`). After loading it answers a warm-up request and wires the model's
        memory (`residency.pin`); `keep_awake`, by default, also sends the GPU a trivial command every that many seconds
        for 5 minutes after each answer, which keeps the model's memory wired across the pauses an agent leaves between
        calls (Kettei Dot, M5 Pro: 30 ms a short call 5 to 15 s after the last, 109 to 115 ms without);
        None sends none. `settings` is a bring-your-own model's `ippon.json` given in code (docs/bring-your-own.md),
        in place of a file beside its weights. `window` replaces the window a creator's code sets below what the
        model reads (a server's context, a training budget: H2O-Lightning's 40,944 tokens, its server's 40,960 less
        16, Clef-Flash's and Torchcast's 16,384, pplx-decider's 32,768); past the creator's it is unverified, and
        `details["window"]` says so. New weight downloads need an accepted licence, or `accept_license=True`; cached
        and local models need none. A model too big for this Mac is refused with the precisions that fit
        (`admission.admit`).
        `answer_cache` keeps the answers to that many requests and answers one identical to them without reading it
        again (answers.py; the default, `answers.DEFAULT_SIZE`, 1,024, suits an agent's checks); 0 keeps none, so
        every request is read."""
        validate_load_options(
            chunk=chunk,
            window=window,
            prefix_budget=prefix_budget,
            answer_cache=answer_cache,
            keep_awake=keep_awake,
            strategy=strategy,
            truncation=truncation,
        )
        asked = names.named_precision(model, precision)  # what precision or a short name (kettei-core:q4) asks for
        meant = get_precision(asked)
        chosen = resolve(profile, switched=switches)
        opened = locate.open_model(model, adapter, settings, accept_license)
        directory, found, base = opened.directory, opened.adapter, opened.base
        if directory.source.as_stored and meant.changes_weights:
            kept = [name for name in NAMES if not PRECISIONS[name].changes_weights]
            raise UnsupportedModel(f"{model} is quantized already: load it as it is (precision {listing(kept)})")
        stored = opened.stored()
        formats = directory.source.formats(directory, stored)
        try:
            needed = admission.admit(
                stored,
                asked,
                chosen.device.working_set_bytes if chosen.device else None,
                can_quantize=not directory.source.as_stored,
            )
        except InsufficientMemory as error:
            raise InsufficientMemory(f"{error}{names.smaller_build(model)}") from None

        def build(dtype: mx.Dtype) -> Decider:
            """The model computing in `dtype`, with what depends on it: its kernel choices and its kept prefixes."""
            loaded = found.load(directory, LoadContext(dtype=dtype, bits=meant.bits, base=base))
            if meant.bits is not None and not quantized(loaded.backbone):
                raise UnsupportedModel(
                    f"{model}: the {found.name} adapter loaded no quantized matrices at {asked} (it must pass the "
                    "load's bits to its layers: LoadContext.backbone does)"
                )
            decider = cls(
                loaded,
                name=str(model),
                profile=chosen,
                chunk=chunk,
                strategy=strategy,
                prefix_budget=prefix_budget,
                literal_state=literal_state,
                truncation=truncation,
                window=window,
                answer_cache=answer_cache,
                resident=needed,
            )
            decider.details |= {
                "adapter": found.name,
                "formats": formats,
                "build": provenance.describe(model, directory, base),
            }
            if (low := drift.low_bit_format(packed_formats(loaded.backbone))) is not None:
                decider.details["drift"] = drift.disclaimer(drift.name_of(model, asked), low)
            if directory.source.checkpoint is not None:
                decider.details["checkpoint"] = directory.source.checkpoint
            decider.own_questions = asks.own(found, directory)
            # The load's temporaries, which no request reuses, before the warm-up fills MLX's buffer cache with what
            # requests do: d1-2B's Q8_0 file served at 2.27 GB instead of 3.51 (M5 Pro), its bf16 checkpoint 63 MB less.
            mx.clear_cache()
            return decider

        if meant.dtype is None:  # the compute dtype measured: fp16 where it fits, else bf16
            decider = build(mx.float16)
            peak = decider.warm_up_measuring_peak()
            if peak < FP16_CEILING:
                decider.details["precision"] = "fp16"
                decider.fallback = lambda: build(mx.bfloat16)
            else:
                decider.close()
                del decider  # its fp16 weights, before the bf16 ones load
                mx.clear_cache()
                decider = build(mx.bfloat16)
                decider.details["precision"] = "bf16"
                if warm_up:
                    decider.warm_up()
            decider.details["peak_activation"] = round(peak, 1) if peak < float("inf") else None  # None: overflowed
        else:
            decider = build(DTYPES[meant.dtype])
            decider.details["precision"] = quantization.SHORT[meant.dtype]  # "fp16" for float16 or kettei-core:fp16
            if warm_up:
                decider.warm_up()
        if "drift" in decider.details:  # below 8 bits, as the caller asked: said once, and in every response
            drift.warn(decider.details["drift"])
        if meant.changes_weights:
            decider.details["weights"] = asked
        if pin_memory:
            decider.details["wired_limit"] = pin()  # bytes, from this Mac's memory as it is now (0: none wired)
            decider.pinned = True
        if keep_awake:
            decider.start_keep_awake(keep_awake)
        # What loading made lives as long as the model, mostly Python objects (about 100,000 for a small encoder:
        # imports, tokenizer, adapter), and a full collection, which a request's own garbage sets off now and then,
        # walked every one: 11 ms inside a request (M5 Pro). Frozen, a collection walks only what came after. Freezing
        # is process-wide, so only a process ippon owns asks for it (the command line and the server: cli.load_options);
        # a library load leaves the caller's collector as it was.
        if freeze_objects:
            gc.collect()
            gc.freeze()
        return decider

    @property
    def precision(self) -> str:
        """What it computes in: "fp16", "bf16", ... as `load` chose it (bf16 for a decider built by hand)."""
        return str(self.details.get("precision", "bf16"))

    @property
    def adapter(self) -> str:
        """The adapter's name ("liquid/d1"), as the load selected it; the format's class for a Decider built by hand."""
        return str(self.details.get("adapter") or type(self.format).__name__)

    @property
    def asks(self) -> Asker:
        """The questions it asks itself, (count, variant) -> System One questions: its own, else the standard ones."""
        return self.own_questions or standard

    def start_keep_awake(self, period: float = PERIOD_S) -> None:
        """A trivial GPU command every `period` seconds for 5 minutes after each answer (`residency.KeepAwake`)."""
        if self.keep_awake is None and mx.metal.is_available():
            self.keep_awake = KeepAwake(self.lock, period)

    def close(self) -> None:
        """Unload the model: stop the keep-awake thread, release the weights and the kept prefixes, empty MLX's buffer
        cache, and wire again only what other loaded models hold. Answering afterwards raises DecisionError; closing
        again does nothing. A decider is a context manager: `with Decider.load(...) as d:` closes it on leaving."""
        if self.closed:
            return
        if self.keep_awake is not None:
            self.keep_awake.close()
            self.keep_awake = None
        self.prefixes, self.fallback, self.answers = None, None, None
        vars(self).pop("backbone", None)  # absent already when `_to_bf16` failed to load the bf16 weights
        vars(self).pop("format", None)
        free_memory()
        self.closed = True  # last: closed means the memory is given back
        if self.pinned:
            pin()  # the limit for what is still loaded

    def __enter__(self) -> Decider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _open(self) -> None:
        if self.closed:
            who = self.name or self.details.get("adapter") or "the model"
            raise DecisionError(f"{who}: this decider is closed (Decider.close unloaded it)")

    @contextmanager
    def with_ops(self, ops: Ops) -> Iterator[None]:
        """Within it, the backbone runs `ops` in place of its own: a copy of them that measures or times each operation
        (precision.PeakOps, profile.TimedOps)."""
        plain, self.backbone.ops = self.backbone.ops, ops
        try:
            yield
        finally:
            self.backbone.ops = plain

    def warm_up_measuring_peak(self) -> float:
        """Answer the warm-up requests while recording the largest magnitude in the residual stream and in every
        matmul's output; that magnitude, or infinity when an answer overflowed (non-finite probabilities)."""
        measuring = PeakOps.of(self.backbone.ops)
        try:
            with self.with_ops(measuring):
                self.warm_up()
        except NonFiniteAnswer:  # the activations overflowed
            return float("inf")
        return measuring.value()

    def warm_up(self) -> None:
        """Answer a small request of every question type (or of the ones the model takes, or its own questions:
        asks.warm_up), so the first real request doesn't pay for reading the weights in and compiling kernels. A
        request the model refuses is skipped, and a model that refuses them all is refused (UnsupportedModel): nothing
        was compiled or measured. It leaves every array the model holds evaluated, which is also what lets another
        thread run it (MLX's streams belong to a thread)."""
        read, refused = 0, None
        with self.uncached():  # a warm-up reads every time: it compiles and measures
            for index, body in enumerate(asks.warm_up(self.own_questions)):
                try:
                    self.answer(parse_request(body))
                except NonFiniteAnswer:  # not a refusal: the activations overflowed (`warm_up_measuring_peak`)
                    raise
                except DecisionError as error:
                    refused = error
                    continue
                read += 1
                if index == 0:  # every type in one request: the rest are each of them alone
                    break
        if self.prefixes is not None:
            self.prefixes.clear()
        if not read:
            who = self.name or self.adapter
            raise UnsupportedModel(f"{who} refused every warm-up request ({refused})") from refused

    def decide(
        self,
        state: Any = None,
        questions: Mapping[str, Builder | Mapping[str, Any]] | None = None,
        *,
        model: str | None = None,
        images: Sequence[Any] = (),
        audio: Any = None,
        videos: Sequence[Any] = (),
    ) -> Result:
        """Answer questions about a state: builders (YesNo, Choice, Score) or System One question dicts. For a model
        that reads them, `images` (bytes or path of a file, a PIL image, an H x W x 3 uint8 array, a data: URI),
        `audio` (a clip or a list of them: bytes or path of a file, or a 1-D array of 16 kHz samples) and `videos`
        (bytes or path of a file, a data: URI): media/decode.as_media."""
        if not questions and isinstance(state, Mapping) and "questions" in state:
            raise InvalidRequest(
                "decide(state, questions) takes the state and its questions apart; a whole System One request "
                "({'state', 'questions'}) goes to systemone(body)"
            )
        if not questions:
            raise InvalidRequest("ask at least one question")
        request = parse_request(request_body(state, questions, model))
        return self.answer(with_media(request, images, audio, videos))

    def systemone(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """A System One request body in, the model's System One response body out."""
        return self.answer(parse_request(body)).response

    def sqwish(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """A request to Sqwish's decide API (`{"context", "decisions": [...]}`, as the sqwishai SDK's `decide`
        sends it) in, its response out: each decision's probabilities by outcome, top, p_top and policy
        (decide_api.py), with the milliseconds rendering and scoring took."""
        systemone, decisions, temperature = decide_api.to_systemone(body)
        started = time.perf_counter()
        rendering = self.render(parse_request(systemone))
        scoring = time.perf_counter()
        result = self.answer_rendered([rendering])[0].timed(scoring - started, time.perf_counter() - scoring)
        return decide_api.response(result, decisions, temperature, str(body.get("model") or self.name))

    def decide_many(self, requests: Sequence[Mapping[str, Any]]) -> list[Result]:
        """Several requests at once, each `{"state": ..., "questions": {...}}` with builders or System One question
        dicts (and `images` or `audio` as `decide` takes them). Requests about the same state are read as one,
        sharing its reading and one pass for all their questions; a non-causal model reads all their prompts in
        shared batches. The answers are each request's own, as `decide` gives them."""
        parsed = []
        for index, body in enumerate(requests):
            try:
                if not isinstance(body, Mapping):
                    raise InvalidRequest("a request must be a mapping with 'state' and 'questions'")
                converted = request_body(body.get("state"), body.get("questions"), body.get("model"))
                given = {key: value for key, value in body.items() if key not in MEDIA_FIELDS}
                parsed.append(
                    with_media(
                        parse_request(given | converted),
                        body.get("images") or (),
                        body.get("audio"),
                        body.get("videos") or (),
                    )
                )
            except InvalidRequest as error:
                raise InvalidRequest(f"requests[{index}]: {error}") from None
        return self.answer_rendered([self.render(p) for p in parsed])

    def systemone_many(self, bodies: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Several System One request bodies in, their response bodies out, answered as `decide_many` does."""
        return [r.response for r in self.answer_rendered([self.render(parse_request(b)) for b in bodies])]

    def classify(self, texts: str | Sequence[str]) -> dict[str, float] | list[dict[str, float]]:
        """Every label's probability for a text, or for each of several (read together, as `decide_many` reads
        them): for a classifier (a `*ForSequenceClassification` checkpoint, `adapters/classifier.py`)."""
        many = [texts] if isinstance(texts, str) else list(texts)
        asked = [self.format.classify_request(t) for t in many]  # UnsupportedModel for a model that isn't a classifier
        out = [self.format.classify_result(result.response) for result in self.decide_many(asked)]
        return out[0] if isinstance(texts, str) else out

    @property
    def tokenizer(self) -> Tokenizer | None:
        """The format's tokenizer (for `literal_state` and `inspect`'s text), when it is ippon's."""
        tokenizer = self.format.tokenizer
        return tokenizer if isinstance(tokenizer, Tokenizer) else None

    def _widen(self, window: int) -> None:
        """The creator's window replaced by `window` tokens (`load`)."""
        creator = self.format.window
        who = self.name or type(self.format).__name__
        if creator is None:
            raise UnsupportedModel(
                f"{who}: its creator's code sets no window; only the model's positions limit a prompt"
            )
        self.format.window = window
        self.details["window"] = {"creator": creator, "set": window, "verified": window <= creator}
        if window > creator:
            warnings.warn(
                f"{who}: a window of {window:,} tokens, past its creator's {creator:,}: answers on longer prompts are "
                "unverified (the creator's code never reads them)",
                stacklevel=3,
            )

    @memory_errors
    def render(self, request: Request) -> Rendering:
        """A validated request's prompts: where a request's own errors (InvalidRequest, ContextTooLong) surface.
        With `literal_state`, a special token's string inside the state (`<|im_end|>` in an email) is read as its
        ordinary text, never as the token: the state's control strings can't end the user turn or start the
        assistant's. A state without one renders exactly as without the option. A prompt past the backbone's
        `max_tokens`, or past the window its creator's code reads (`Format.window`), is refused here, before anything
        is read. What the format left out to fit its window (as the creator's code does) is in `rendering.cuts`, and
        in the response's `truncated`; with truncation="error" a request that would lose anything is refused instead
        (ContextTooLong, naming every cut). An empty state is answered from the questions alone, as every creator's
        code answers it (d1's, elephant-1's and Kettei's references each hold one)."""
        self._open()
        size = 0 if request.state is None else len(as_text(request.state))
        if size > (most := max(MOST_STATE_CHARS, (self.backbone.max_tokens or 0) * CHARS_PER_POSITION)):
            raise ContextTooLong(f"the state is {size:,} characters, past the {most:,} this model takes: shorten it")
        wide = self.format.wide
        first = first_round(request, wide) if wide is not None else None
        rendering = self._render(first[0] if first else request)
        if first:
            rendering = replace(rendering, round=first[1])
        window = self.format.window
        if window is not None and (over := next((p for p in rendering.prompts if len(p.tokens) > window), None)):
            asked = over.readouts[0].question.split(SEPARATOR)[0]  # a wide question's, not one of its groups'
            raise ContextTooLong(
                f"{asked}: a prompt of {len(over.tokens):,} tokens, past the {window:,} its creator's code reads at "
                "most: shorten the state"
            )
        limit = self.backbone.max_tokens
        longest = max(len(p.tokens) for p in rendering.prompts)
        if limit is not None and longest > limit:
            raise ContextTooLong(f"a prompt of {longest} tokens, more than the model's {limit}")
        if (held := self.tokens_here) is not None and longest > held:
            raise ContextTooLong(
                f"a prompt of {longest:,} tokens: its reading needs more memory than this Mac's GPU keeps beside the "
                f"model, which holds about {held:,} of this model's tokens at once. Shorten the state"
            )
        if rendering.cuts and self.truncation == "error":
            lost = "; ".join(
                f"{c.question + ': ' if c.question else ''}{c.part}, {c.dropped} of {c.dropped + c.kept} tokens"
                for c in rendering.cuts
            )
            raise ContextTooLong(f"the request doesn't fit the model's window without dropping input ({lost})")
        phase("render")
        return rendering

    def _room(self, resident: int | None) -> int | None:
        """The longest prompt this Mac's GPU holds the reading of beside the model's weights (`resident` bytes) and its
        kept prefixes: the keys and values each token keeps (measured on a cache of one token) against the room left
        in the working set. A prompt inside the window can be past it (130,000 tokens of Kettei Core took a 36 GB
        Mac's memory to its warning level and kept it there), so it is refused before it is read. The kept prefixes
        fit the same room (`Prefixes.fit`). None when the room or a token's size is unknown."""
        device = self.profile.device
        if device is None or not device.working_set_bytes or not resident:
            return None
        kept = self.prefixes.budget if self.prefixes is not None else 0
        room = device.working_set_bytes - resident - kept - READING_MARGIN
        if self.prefixes is not None:
            self.prefixes.fit(room)
        try:
            cache = self.backbone.new_cache(1)
        except NotImplementedError:  # a backbone that keeps no cache: no reading to hold
            return None
        per_token = sum(
            layer.k.nbytes + layer.v.nbytes for layer in getattr(cache, "layers", ()) if isinstance(layer, Attention)
        )
        return max(room // per_token, 1) if per_token else None

    def inspect(self, request: Request | Mapping[str, Any]) -> dict[str, Any]:
        """What the model reads for a request, without reading it: each prompt's questions, its token count and
        ids, its text (when the format has a tokenizer), and what was cut to fit the window (`truncated`)."""
        rendering = self.render(request if isinstance(request, Request) else parse_request(request))
        tokenizer = self.tokenizer
        prompts = []
        for p in rendering.prompts:
            text = [
                t for t in p.tokens if not is_placeholder(t)
            ]  # a medium's rows are placeholders (arch/common.media_id)
            prompts.append(
                {
                    "questions": sorted({r.question for r in p.readouts}),
                    "tokens": len(p.tokens),
                    **({"media_rows": len(p.tokens) - len(text)} if len(text) < len(p.tokens) else {}),
                    "ids": list(p.tokens),
                    **({"text": tokenizer.decode(text)} if tokenizer is not None else {}),
                }
            )
        return {
            "max_tokens": self.backbone.max_tokens,
            "max_options": {kind: self.format.most_options(kind) for kind in ("choice", "score")},
            "prompts": prompts,
            "truncated": [c.to_json() for c in rendering.cuts],
        }

    def _render(self, request: Request) -> Rendering:
        taken = self.format.media
        check_anchors(request, self.format.anchors)
        spoken = any(q.spoken is not None for q in request.questions)
        given_media = (
            ("images", request.images),
            ("audio", request.audio),
            ("spoken", spoken),
            ("videos", request.videos),
        )
        for kind, given in given_media:
            if given and kind not in taken:
                who = self.name or self.adapter
                raise InvalidRequest(f"{kind}: {who} takes {', '.join(sorted(taken)) or 'text only'}")
        check_images(request.images)
        for q in request.questions:
            most = self.format.most_options(q.type)
            if most is not None and len(q.options) > most:
                who = self.name or self.adapter
                raise InvalidRequest(f"{q.id}: {len(q.options)} options; {who} answers at most {most} in a question")
        if not self.literal_state:
            return self.format.render(request)
        tok = self.tokenizer
        assert tok is not None  # checked in __init__
        mapping: dict[str, str] = {}
        state = _placeholders(request.state, tok.special_strings, mapping)
        if not mapping:
            return self.format.render(request)
        with literal(mapping):
            rendering = self.format.render(replace(request, state=state))
        return replace(rendering, request=request)

    @contextmanager
    def uncached(self) -> Iterator[None]:
        """Within it, every request is read, none answered from the kept answers (`answer_cache`): a warm-up's, a
        timed one's. Answers read within it are not kept either."""
        before, self._uncached = self._uncached, True
        try:
            yield
        finally:
            self._uncached = before

    def answer(self, request: Request) -> Result:
        """The answer to a parsed request (`schema.parse_request`)."""
        return self.answer_rendered([self.render(request)])[0]

    @memory_errors
    def answer_rendered(self, renderings: Sequence[Rendering]) -> list[Result]:
        """The answers to rendered requests, in order, reading together the ones that can be."""
        self._open()
        with self.lock:
            answers = None if self._uncached else self.answers
            keys = [Answers.key(r, self.chunk, self.strategy) if answers is not None else None for r in renderings]
            kept = [answers.get(k) if answers is not None else None for k in keys]
            # the build is the decider's, as it is now, not the kept request's; and none of its input was read again
            out: list[Result | None] = [
                None
                if r is None
                else replace(r, build=self.details.get("build"), kept_tokens=sum(len(p.tokens) for p in g.prompts))
                for r, g in zip(kept, renderings, strict=True)
            ]
            todo = [i for i, r in enumerate(out) if r is None]
            fallback: str | None = None
            try:
                results = self._answer_rounds([renderings[i] for i in todo]) if todo else []
            except NonFiniteAnswer:
                if self.fallback is None:
                    raise
                self._to_bf16()
                fallback = "an answer overflowed fp16: answered again in bf16, as from now on"
                todo = list(range(len(renderings)))  # kept fp16 answers too: one precision for the whole batch
                results = self._answer_rounds(renderings)
            for i, result in zip(todo, results, strict=True):
                out[i] = answered = replace(result, precision=self.precision, fallback=fallback)
                if answers is not None:
                    answers.put(keys[i], replace(answered, fallback=None))  # the fallback is this answer's story only
            if self.keep_awake is not None:
                self.keep_awake.busy()
            return [r for r in out if r is not None]

    def _to_bf16(self) -> None:
        """Compute in bf16 from now on: precision="auto" chose fp16 from the warm-up's activations, and a request
        overflowed it after all. The fp16 weights go first, then the model loads again in bf16."""
        assert self.fallback is not None
        warnings.warn(
            f"{self.name}: an answer overflowed fp16 (activations past 65,504); computing in bf16 from now on",
            stacklevel=3,
        )
        rebuild, self.fallback = self.fallback, None
        if self.prefixes is not None:
            self.prefixes.clear()  # fp16 readings: never mixed with bf16 ones
        if self.answers is not None:
            self.answers.clear()  # fp16 answers too
        del self.backbone, self.format
        mx.clear_cache()
        try:
            other = rebuild()
            other.warm_up()
        except BaseException:
            self.close()  # the fp16 weights are gone and the bf16 ones didn't load: closed, never half-built
            raise
        self.backbone, self.format = other.backbone, other.format
        self.prefixes, self.tokens_here = other.prefixes, other.tokens_here  # the bf16 build's own, fitting its room
        self.details |= {"precision": "bf16", "peak_activation": None}

    def _answer_rounds(self, renderings: Sequence[Rendering]) -> list[Result]:
        """The answers, a request whose wide choices were asked in groups finished with their final rounds
        (rounds.answered)."""
        return answered(renderings, self._read, self.render, self.format.wide, self.format.answer)

    def _read(self, renderings: Sequence[Rendering]) -> list[Result]:
        """The answers to `renderings`, read with their media (`Format.encode_media`) as runtime.read_requests reads
        them together, each finished as soon as it is read."""
        # pictures encoded only when a pass reads them (MediaRows): those inside a kept prefix never are
        pictured = any(r.request.images for r in renderings)
        eager = self.format.encode_media([_images(r, None) for r in renderings] if pictured else renderings)
        if eager or pictured:
            later = (lambda d: self.format.encode_media([_images(r, d) for r in renderings])) if pictured else None
            self.backbone.media = MediaRows(eager, later)
        try:
            results: list[Result | None] = [None] * len(renderings)
            prompts = [r.prompts for r in renderings]
            for i, hidden, kept in read_requests(self.backbone, prompts, self.chunk, self.strategy, self.prefixes):
                results[i] = replace(self._finish(renderings[i], hidden), kept_tokens=kept)
        finally:
            self.backbone.media = None
        assert all(r is not None for r in results)
        return [r for r in results if r is not None]

    def _finish(self, rendering: Rendering, hidden: Sequence[mx.array]) -> Result:
        """The result from the hidden states of each of the rendering's prompts' readouts: every question's scores
        (the format's head), evaluated, then its answer and the response body."""
        scores = self.format.scores(rendering, Hidden.gather([p.readouts for p in rendering.prompts], hidden))
        mx.eval(list(scores.values()))
        phase("readout")
        request = rendering.request
        answers: dict[str, Answer] = {}
        shaped: dict[str, dict[str, Any]] = {}
        for question in request.questions:
            if question.id not in scores:
                raise RuntimeError(f"{self.adapter}: no scores for question {question.id!r}")
            probabilities, shaped[question.id] = self.format.answer(
                question, scores[question.id].astype(mx.float32).tolist()
            )
            answers[question.id] = Answer(question, probabilities)  # checked: one per option, finite, summing to 1
        response: dict[str, Any] = {
            "model": request.model or self.name or None,
            "answers": shaped,
            "usage": self.format.usage(rendering),
        }
        if rendering.cuts:  # what the model didn't read: present only when something was left out
            response["truncated"] = [c.to_json() for c in rendering.cuts]
        notes: list[str] = []
        if "drift" in self.details:  # loaded below 8 bits (drift.py)
            notes.append(self.details["drift"])
        if rendering.videos:
            response["videos"] = [dict(v) for v in rendering.videos]
            for index, video in enumerate(rendering.videos, 1):
                if video.get("capped"):
                    gap = video.get("seconds_per_frame")
                    spacing = f" (about 1 frame every {gap:.1f} s)" if isinstance(gap, (int, float)) else ""
                    notes.append(
                        f"video {index}: read {video['frames_read']} frames across {video['duration_s']:.1f} s"
                        f"{spacing}; brief events between sampled frames may be missed"
                    )
        if notes:
            response["warnings"] = notes
        result = Result(answers, response, build=self.details.get("build"))
        if request.unsure_below is not None:  # asked for: the questions to send on
            response["unsure"] = result.unsure(request.unsure_below)
        phase("answer")
        return result


def free_memory() -> None:
    """Give back what a model let go of (closed, unloaded, or a load that failed) now: collect it, frozen objects
    included (`Decider.load` freezes what loading made, and a reference cycle between a backbone and its compiled
    functions would otherwise keep the weights), then freeze what is still alive again, and empty MLX's buffer cache,
    so other apps get the memory."""
    frozen = gc.get_freeze_count() > 0
    gc.unfreeze()
    gc.collect()
    if frozen:
        gc.freeze()
    mx.clear_cache()


def _images(rendering: Rendering, digests: set[int] | None) -> Rendering:
    """The rendering with none of its request's pictures (`digests` None), or only the pictures `digests` name and no
    other media, for encoding its media apart from its pictures, or those pictures alone."""
    request = rendering.request
    if digests is None:
        return replace(rendering, request=replace(request, images=()))
    kept = tuple(i for i in request.images if media_digest(i.sha256) in digests)
    return replace(rendering, request=replace(request, images=kept, audio=(), videos=()))


def default_prefix_budget(memory_bytes: int | None) -> int:
    """Bytes of kept prefixes when none is asked for: PREFIX_BUDGET, or a sixteenth of a smaller Mac's memory."""
    return min(PREFIX_BUDGET, (memory_bytes or 0) // 16)


def _placeholders(value: Any, specials: Sequence[str], mapping: dict[str, str]) -> Any:
    """`value` (a state: text, or JSON's lists and objects) with each special token's string replaced by its
    placeholder, which `mapping` records (placeholder -> string)."""
    if isinstance(value, str):
        for s in specials:
            if s in value:
                p = next((k for k, v in mapping.items() if v == s), None) or placeholder(len(mapping))
                mapping[p] = s
                value = value.replace(s, p)
        return value
    if isinstance(value, Mapping):
        return {_placeholders(k, specials, mapping): _placeholders(v, specials, mapping) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_placeholders(v, specials, mapping) for v in value]
    return value
