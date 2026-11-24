"""The model's System One API over HTTP, with requests that arrive together answered together.

    ippon serve LiquidAI/d1-2B --port 8000
    curl localhost:8000/v1/systemone -d '{"state": "...", "questions": {"refund": {"type": "noul", ...}}}'

- `POST /v1/systemone`: a System One request in, the model's System One response out (`?diagnostics=1`: with
  how it was answered: build, precision, whether input was cut, the fallback's reason); `POST /v1/decisions`, the
  same, at the path Perplexity's Decisions API and its Decider for Mac serve the same schema at, so their clients
  can call ippon unchanged; and there too SGLang's decisions request, told apart by its list of questions
  (sglang.from_sglang), answered in SGLang's shape (sglang.sglang_response);
- `GET /v1/models`: the models, what each loads and on what weights (`full`, `q8`, `q6`, `q4`), each one's adapter
  and chip profile once loaded, as `data` (OpenAI's list) and as
  `models` (TypeSafe's); `GET /health`: 200 while the model thread is running (with the number of requests
  waiting), 503 once it has stopped or the server is closing;
- `GET /v1/status`: the server's memory: its physical footprint, its budget, each model's weights and kept prefixes
  (bytes held, and the most they may hold now), the memory pressure last read, and the latest give-backs.

A request body may come compressed (`Content-Encoding`): gzip, deflate, or zstd where Python reads it (3.14's
compression.zstd, or backports.zstd), which TypeSafe's SDK sends when it can; another encoding gets a 415.

One thread runs the model. Requests that arrive while it is busy wait in a queue, and it takes all of them (up to
`max_batch`) at once, after waiting up to `window_ms` from the first for more to arrive (0, the default, never
waits; `bench --scenario concurrent` measures what a window gives): requests about the same state are read as one,
and a non-causal model reads their prompts in shared batches (`Decider.answer_rendered`).

A server can answer for several models (`ippon serve A B C`, `Models`): a request names one by its `model`
(the name it was served as; none: the first), and the one thread answers each model's waiting requests together,
oldest first. A model loads on its first request (the first at start), and when one doesn't fit the memory left
(`budget`: by default what admission.py allows one model), the least recently used are unloaded first. A model whose
weights alone exceed the budget is refused when the server starts, before anything loads. The room the loaded models'
weights leave is shared equally between their kept prefixes (each at most its own default), shared again whenever a
model loads or unloads, so the weights and the kept prefixes together stay within the budget. A request for a model
the server doesn't answer for, by any of its names, or at weights it doesn't run (kettei-core:q8 from a server of
Kettei Core as published), gets a 404 (`model_not_found`), from a server of one model too.

A request's own errors fail that request only, and no error stops the model thread: one escaping a batch fails that
batch's requests. At most `max_queue` requests wait, counting those held for another model's turn; past that a request
is refused at once (503, `Retry-After`). A request is parsed before its model loads, so a malformed one never loads or
unloads a model. A request waits for its answer as long as the model takes (a long record with many questions can take
minutes on a small Mac); with `timeout_s`, one not answered within it gets a 504, and if the model hasn't started on it,
held or not, it is dropped: it no longer counts against `max_queue` and is never read (one the model is already reading
finishes, unread). A request whose caller hangs up while it waits is dropped the same way (the connection is looked at
every `POLL_S`). A load that fails gives back what it held at once; one that ran out of GPU memory, as when other apps
hold it, is a 503 (`insufficient_memory`, `Retry-After`), as is a pass that does.

Sharing the Mac. The models' thread sends the GPU a trivial command every `heartbeat.period` seconds while no
request waits, which keeps their memory wired and the next request fast (residency.py), for `heartbeat.window`
seconds after the last request, then none until the next. With `keep_alive`, a model that has answered nothing for
that many seconds is unloaded (0: as soon as nothing waits) and its memory given back; its next request loads it
again. `ippon serve --policy` names three settings of the two: `balanced` (the default) sends the command for
300 s after the last request and keeps every model loaded; `latency` sends it as long as the server runs and keeps
every model loaded; `battery` sends none and unloads a model after 5 minutes without a request. Under `balanced`
and `battery` the first request after an idle spell is slower: its model's memory is wired again (a 500-token
request took 227 ms instead of 141 after 5 s idle, M5 Pro, 10 GB model) or, unloaded, the model loads again.
On the default prefix budget, an agent's growing state stays kept past the budget while requests come, and goes
300 s after the last one (`Models.rest`), as the heartbeat stops.

Memory pressure. A thread of the server's own reads this Mac's memory every second while requests come and every ten
seconds otherwise, without MLX or Metal (pressure.py, which also says where its lines are and why). While memory is
short (soft: macOS says warning, or estimated available memory falls under 8% of RAM), the model
thread, between batches, gives back in order: kept prefixes (each model's share cut to a quarter), MLX's buffer
cache, and, when the next reading is still short, one idle model a reading, the least recently used, never the most
recently used one or one a waiting request names. While memory is very short (hard: macOS says critical, or under
3%), it keeps no prefixes and unloads at once every idle model but the most recently used, and a request for a model
that isn't loaded gets a 503 (`overloaded`, `Retry-After: 5`), since loading it would take more; a loaded model goes
on answering, as answering takes little beside its weights (on an 8 GB Mac a 2B model at 16 bits leaves too little
for the line to clear while it is loaded, so refusing it too would refuse it for good). Once memory is plenty
again the prefixes' shares come back; an unloaded model loads on its next request. None of this changes an answer:
a dropped prefix is read again.

Stopping. SIGTERM stops the server as Ctrl-C does (`Server`): it stops taking connections, answers the requests
already waiting for up to 5 s and refuses the rest (503, `unavailable`), sends the answers, stops the heartbeat and
the memory thread, and unloads its models, so a launcher can stop it.

Security. The server listens on 127.0.0.1 unless told otherwise. There, a request whose Host header is not localhost,
127.0.0.1 or [::1] (with any port) is refused (403): a web page can rebind its own domain to 127.0.0.1, and its requests
then carry that domain. A request with an Origin header, which browsers add to a page's cross-site requests, is refused
(403) unless its origin is in `allow_origins` ("*": any); an allowed one gets CORS headers and its preflight (OPTIONS)
is answered. With `api_key`, requests other than OPTIONS need `Authorization: Bearer KEY` (401 otherwise). Without one,
an API client's Authorization header (TypeSafe's SDK sends one) is ignored. Errors come as `{"error": {"type",
"message"}}`, these too (`forbidden`, `unauthorized`).
"""

from __future__ import annotations

import functools
import hmac
import importlib
import json
import math
import queue
import select
import shutil
import signal
import socket
import socketserver
import statistics
import sys
import threading
import time
import zlib
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import decide_api, locate, names, profiles, terminal
from .admission import budget
from .batches import answer_each
from .decider import Decider, free_memory
from .device import max_body as device_max_body
from .errors import (
    Abandoned,
    DecisionError,
    Forbidden,
    InsufficientMemory,
    InvalidRequest,
    MemoryShort,
    MethodNotAllowed,
    ModelNotFound,
    NotFound,
    Overloaded,
    Timeout,
    TooLarge,
    Unauthorized,
    Unavailable,
    UnsupportedEncoding,
    as_decision_error,
)
from .errors import out_of_memory as _out_of_memory
from .media.decode import check_images
from .options import AWAKE_WINDOW_S, Reads
from .policy import MAX_BATCH, MAX_MEDIA_BATCH, MAX_QUEUE
from .pressure import Level, Monitor, footprint
from .residency import Heartbeat, pin, release
from .runtime import STOP
from .schema import Request, Result, parse_request, read_json
from .served import CLIENT_HEADER, DEFAULT_PORT, LOOPBACK, ServedModel, is_loopback
from .sglang import from_sglang, is_sglang, sglang_response

TIMEOUT_S: float | None = None  # no deadline: the model takes what it takes, and a caller that stops waiting hangs up
HISTORY = 1024  # batch sizes kept for logs and tests
POLL_S = 0.1  # seconds between looks at whether a waiting request's caller has hung up
LOCAL_HOSTS = tuple(f"[{h}]" if ":" in h else h for h in LOOPBACK)  # as a Host header names them
SOFT_KEEP = 0.25  # of each model's share, the kept prefixes it keeps while memory is short; none while very short
GIVEN_BACK = 32  # the latest give-backs kept for /v1/status
POSTED = ("/v1/systemone", "/v1/decisions", "/v1/decide")  # System One, SGLang's decisions and Sqwish's decide API
GOT = ("/health", "/v1/models", "/v1/status")
CLOSE_GRACE_S = 5.0  # seconds a closing server goes on answering waiting requests before it refuses the rest
SEND_S = 2.0  # seconds a closing server waits for the answers it has to be sent


@dataclass
class Model:
    """A model a server answers for, by `name`: `load()` loads it (warmed up), `needed` its weights' resident bytes as
    estimated before loading (`locate.estimate`), `kept` its kept prefixes' share of the budget while loaded."""

    name: str
    load: Callable[[], Decider]
    needed: int = 0
    loads: str = ""  # what `load` loads (a Hub id, a folder), so a request may name it by any of its names
    weights: str = "full"  # "q8", "q6" or "q4" when quantized as it loads (names.weights_at): a request asks for those
    # how its loading options change what it answers, so a client that reuses the server can tell whether it answers
    # as the client asks (None: not known, for a decider loaded elsewhere)
    reads: Reads | None = None
    precision: str | None = None  # the precision it loads at, as asked ("auto", "fp32", "q8"; None: not known)
    decider: Decider | None = None  # while loaded
    media: tuple[str, ...] = ()  # the inputs it reads besides text, while loaded: read without its decider's format
    used: float = 0.0  # when it last answered (time.monotonic): the least recently used unload first

    def listed(self) -> ServedModel:
        """The model as `GET /v1/models` lists it."""
        d = self.decider
        listed = ServedModel(
            self.name, self.loads or self.name, self.weights, self.reads, precision=self.precision, media=self.media
        )
        if d is None:
            return listed
        return replace(
            listed, loaded=True, adapter=d.adapter, profile=d.profile.name, details=d.details, media=self.media
        )


@dataclass
class _Job:
    body: Any
    model: Model  # the model it names
    request: Request  # parsed on the caller's thread, so the model thread only reads
    future: Future[Result] = field(default_factory=Future)
    waiting: bool = True  # counted against max_queue: until the model starts on it, it is cancelled or it fails


class Models:
    """The models a server answers for, loaded within `budget` bytes (None: no limit). Only the model thread loads,
    unloads and answers with them. When `enforce_memory` is true, the server also sets MLX's process-wide memory and
    cache limits to `budget` and aborts forward passes whose active GPU memory exceeds `budget`."""

    def __init__(self, models: Sequence[Model], budget: int | None = None, *, enforce_memory: bool = False):
        if not models or len({m.name for m in models}) != len(models):
            raise ValueError("a server answers for one model or more, each by its own name")
        self.models, self.budget, self.enforce_memory = list(models), budget, enforce_memory
        over = [m for m in self.models if budget is not None and m.needed > budget]
        if budget is not None and over:
            raise InsufficientMemory(
                ", ".join(f"{m.name} needs about {m.needed / 2**30:.1f} GiB" for m in over)
                + f": more than the server's {budget / 2**30:.1f} GiB alone (--memory-gb, or a smaller precision)"
            )
        self.loaded_here: set[str] = set()  # the models this registry loaded (and so closes)
        self.prefix_scale = 1.0  # the part of its share each model's kept prefixes may hold: less while memory is short
        self.resting = False  # no request for a while (`rest`): no kept prefix outgrows its share

    @classmethod
    def of(cls, decider: Decider) -> Models:
        """One model, loaded already."""
        needed = int(decider.details.get("resident_estimate") or 0)
        media = tuple(sorted(decider.format.media))
        return cls([Model(decider.name or decider.adapter, lambda: decider, needed, decider=decider, media=media)])

    def named(self, body: Any) -> Model:
        """The model a request body names (its `model`): the one served by that name, or the one that loads what the
        name loads (a short name, Hub id or folder of it) at the weights it asks for (kettei-core:q8 8-bit, any other
        name the build as published); the first when it names none. ModelNotFound for any other name, a server of one
        model too, so no answer is labelled with a model or precision that didn't give it."""
        name = body.get("model") if isinstance(body, Mapping) else None
        if not isinstance(name, str) or not name:
            return self.models[0]
        weights = names.weights_at(name)
        for m in self.models:
            if m.name == name or m.listed().answers(name, weights):
                return m
        served = ", ".join(f"{m.name!r} ({m.weights})" for m in self.models)
        raise ModelNotFound(f"{name!r} ({weights}): this server answers for {served}")

    def weights(self) -> int:
        """The bytes the loaded models' weights take."""
        return sum(m.needed for m in self.models if m.decider is not None)

    def prefix_bytes(self) -> int:
        """The bytes the loaded models' kept prefixes hold."""
        return sum(p.nbytes for m in self.models if (d := m.decider) is not None and (p := d.prefixes) is not None)

    def share(self) -> int:
        """Give each loaded model's kept prefixes an equal share of the room the weights leave in the budget (at most
        what it loaded with, times `prefix_scale`), dropping their least recently used entries now where a share
        shrank; the bytes dropped."""
        before = self.prefix_bytes()
        keeping = [p for m in self.models if m.decider is not None and (p := m.decider.prefixes) is not None]
        room = None if self.budget is None else max(0, self.budget - self.weights())
        for prefixes in keeping:
            prefixes.share(None if room is None else room // len(keeping), self.prefix_scale, self.resting)
        return before - self.prefix_bytes()

    def rest(self, now: float, after: float = AWAKE_WINDOW_S) -> None:
        """Once no model has answered for `after` seconds, the kept prefixes past their shares go (`share`), whether
        a default budget's growing state or one kept alone while memory is short:
        an agent's session has stopped (Claude Code's steps: 2 s apart at the median, under 0.7% over 5 minutes, in
        44 public sessions). The next request lets them outgrow it again (`ready`)."""
        if not self.resting and all(now - m.used >= after for m in self.models if m.decider is not None):
            self.resting = True
            self.share()

    def ready(self, model: Model) -> Decider:
        """`model` loaded, unloading the least recently used others first while its weights wouldn't fit the budget;
        then the kept prefixes are shared again. A load that fails gives back what it held before its error is
        raised, as InsufficientMemory when the GPU ran out of memory (other apps may hold it: worth a retry)."""
        if self.resting:
            self.resting = False
            self.share()
        if model.decider is not None and model.decider.closed:  # its bf16 fallback failed to load (`Decider`)
            self.unload(model)
        if model.decider is None:
            if self.budget is not None:
                if model.needed > self.budget:  # refused at start; kept here so no caller can evict for nothing
                    raise InsufficientMemory(f"{model.name} alone needs more than the server's budget")
                others = sorted(
                    (m for m in self.models if m.decider is not None and m is not model), key=lambda m: m.used
                )
                while others and self.weights() + model.needed > self.budget:
                    self.unload(others.pop(0))
            failed: Exception | None = None
            try:
                decider = model.load()
            except Exception as error:
                failed = error
                _unwind(error)  # its frames hold what the load made so far
            if failed is not None:
                free_memory()
                if _out_of_memory(failed):
                    raise InsufficientMemory(
                        f"{model.name} doesn't fit in the GPU memory free now ({failed}): other apps or models hold "
                        "it; send the request again later"
                    )
                raise failed
            model.media = tuple(sorted(decider.format.media))
            model.decider = decider
            self.loaded_here.add(model.name)
            self.share()
        model.used = time.monotonic()
        return model.decider

    def unload(self, model: Model) -> None:
        """Close `model`'s decider, which gives back its memory now (`Decider.close`); its next request loads it
        again."""
        decider, model.decider = model.decider, None  # unlisted first: a request reading it never meets it half-closed
        if decider is not None:
            decider.close()
            self.share()

    def close(self) -> None:
        """Unload the models this registry loaded (one it was given loaded is its owner's to close)."""
        for m in self.models:
            if m.name in self.loaded_here:
                self.unload(m)

    def expire(self, idle: float, now: float) -> None:
        """Unload each model this registry loaded (and so can load again) that has answered nothing for `idle`
        seconds."""
        for m in self.models:
            if m.decider is not None and m.name in self.loaded_here and now - m.used >= idle:
                self.unload(m)


def _unwind(error: BaseException) -> None:
    """Drop the frames `error` and the errors it chains hold, so that what those frames made can be freed."""
    seen: set[int] = set()
    link: BaseException | None = error
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        link.__traceback__ = None
        link = link.__cause__ or link.__context__


def _fail(job: _Job, error: BaseException) -> None:
    """`error` delivered to `job`, unless its client cancelled it first. A request not yet started is claimed with
    set_running_or_notify_cancel, which a cancel can't land in the middle of, so setting its error can't raise
    InvalidStateError on the model's thread; one already started can't be cancelled any more."""
    future = job.future
    if future.done() or (not future.running() and not future.set_running_or_notify_cancel()):
        return
    future.set_exception(error)


def _has_media(job: _Job) -> bool:
    """Whether a job carries images, videos, or audio (batched smaller to keep multimodal bursts responsive)."""
    return bool(job.request.images or job.request.videos or job.request.audio)


class Batcher:
    """Answers submitted requests on one thread, taking every request waiting when it is free. While no request
    waits, that thread also sends `heartbeat`'s GPU command when one is due, and with `keep_alive` (seconds; None or
    infinity: never) unloads the models that have answered nothing for that long. With `monitor`, it gives memory
    back while the Mac is short of it (`_relieve`)."""

    def __init__(
        self,
        decider: Decider | Models,
        max_batch: int = MAX_BATCH,
        window_ms: float = 0.0,
        max_queue: int = MAX_QUEUE,
        history: int = HISTORY,
        heartbeat: Heartbeat | None = None,
        keep_alive: float | None = None,
        monitor: Monitor | None = None,
    ):
        if max_batch < 1 or window_ms < 0 or max_queue < 1 or history < 1:
            raise ValueError("max_batch, max_queue and history are at least 1, window_ms at least 0")
        if keep_alive is not None and keep_alive < 0:
            raise ValueError("keep_alive is a number of seconds of at least 0, or None for never")
        self.models = decider if isinstance(decider, Models) else Models.of(decider)
        self.max_batch, self.window = max_batch, window_ms / 1000
        self.heartbeat, self.keep_alive = heartbeat, keep_alive
        for m in self.models.models:  # on this thread: the model thread can then use every array a decider holds
            if m.decider is not None:
                m.decider.warm_up()
        self._held: deque[_Job] = deque()  # taken for another model than the batch's: answered next
        self._abandoned: set[Future[Result]] = set()  # being read, with nobody waiting for them any more (`abandon`)
        self.batches: deque[int] = deque(maxlen=history)  # the sizes of the last `history` batches answered
        self.answered = 0  # batches answered in all
        self.max_queue = max_queue
        self._queue: queue.Queue[_Job] = queue.Queue()  # cancelled jobs stay in it, uncounted, until taken
        self._waiting, self._lock = 0, threading.Lock()
        self._closing = threading.Event()
        self._give_up = float("inf")  # when closing: past this (time.monotonic), the requests still waiting fail
        self.given_back: deque[dict[str, Any]] = deque(maxlen=GIVEN_BACK)  # what memory pressure took, latest last
        self.monitor = monitor if monitor is not None and monitor.start() else None
        self._seen = 0  # the monitor's readings acted on
        self._spell = 0  # readings acted on since memory was last plenty
        self._thread = threading.Thread(target=self._run, name="ippon-model", daemon=True)
        self._thread.start()

    @property
    def alive(self) -> bool:
        """Whether requests submitted now will be answered: the model thread runs and the batcher is not closing."""
        return self._thread.is_alive() and not self._closing.is_set()

    @property
    def waiting(self) -> int:
        """Requests waiting for the model, not counting cancelled ones."""
        return self._waiting

    def _leave(self, job: _Job) -> None:
        """`job` stops counting against max_queue (once, whichever comes first: taken or cancelled)."""
        with self._lock:
            if job.waiting:
                job.waiting = False
                self._waiting -= 1

    def submit(self, body: Any) -> Future[Result]:
        """A future of the request's Result; its exception is the request's error. Raises Unavailable when the batcher
        can't answer, ModelNotFound for a model it doesn't answer for, and Overloaded when `max_queue` requests are
        waiting already. Cancelling the future before the model takes the request leaves it unread. The body is
        parsed (its media decoded and hashed) here, on the caller's thread, while the model thread reads others; a
        body that isn't a request fails its future, never loading or unloading a model."""
        try:
            request = parse_request(body)
            check_images(request.images)
        except Exception as error:
            failed: Future[Result] = Future()
            failed.set_exception(error)
            return failed
        with self._lock:
            if not self.alive:
                raise Unavailable("the server is closing" if self._closing.is_set() else "the model thread has stopped")
            job = _Job(body, self.models.named(body), request)
            if self.monitor is not None and self.monitor.level is Level.HARD and job.model.decider is None:
                raise MemoryShort("this Mac is very short of memory to load this model: send it again later")
            if self._waiting >= self.max_queue:
                raise Overloaded(f"{self.max_queue} requests are waiting already: send it again later")
            self._waiting += 1
            if self.monitor is not None:
                self.monitor.begin()
            job.future.add_done_callback(lambda f: self._done(job))  # cancelled, failed or answered
            self._queue.put(job)
        return job.future

    def abandon(self, future: Future[Result]) -> None:
        """Nobody waits for `future` any more (past its deadline, or its caller gone): out of the queue if the model
        hasn't started on it, else the reading stops at its next pass once every request in it is abandoned."""
        if future.cancel() or future.done():  # cancel() runs `_done` at once: never under the lock
            return
        self._abandoned.add(future)
        if future.done():  # answered meanwhile: `_done` has been
            self._abandoned.discard(future)

    def _done(self, job: _Job) -> None:
        self._abandoned.discard(job.future)
        self._leave(job)
        if self.monitor is not None:
            self.monitor.end()

    def close(self, grace: float | None = None) -> None:
        """Answers the requests already waiting (with `grace`, those the model hasn't started on within that many
        seconds fail as Unavailable instead), then stops the model thread and the memory monitor."""
        with self._lock:
            if grace is not None:
                self._give_up = min(self._give_up, time.monotonic() + grace)
            self._closing.set()
        self._thread.join()
        if self.monitor is not None:
            self.monitor.close()

    def _take(self, timeout: float | None) -> _Job | None:
        """The next waiting request that is still wanted, still counted and cancellable until the model starts on it
        (`_start`); None if none arrives within `timeout` (0: only one already waiting)."""
        until = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = None if until is None else until - time.monotonic()
            try:
                job = self._queue.get(timeout=wait) if wait is None or wait > 0 else self._queue.get_nowait()
            except queue.Empty:
                return None
            if not job.future.cancelled():
                return job

    def _start(self, job: _Job) -> bool:
        """Whether the model may start on `job`: marks it running (no longer cancellable, no longer counted against
        max_queue), unless it was cancelled."""
        if not job.future.set_running_or_notify_cancel():
            return False
        self._leave(job)
        return True

    def _next(self, timeout: float) -> _Job | None:
        """The oldest request held for its model, else the next waiting one (None: none came)."""
        while self._held:
            held = self._held.popleft()
            if not held.future.cancelled():
                return held
        return self._take(timeout)

    def _idle(self) -> None:
        """Between batches, with no request waiting: act on a new memory reading, unload the models idle past
        `keep_alive`, then send the heartbeat's command if one is due and a model is still loaded to keep fast."""
        self._relieve()
        now = time.monotonic()
        self.models.rest(now)
        if self.keep_alive is not None:
            with self._lock:
                if not self._waiting:
                    self.models.expire(self.keep_alive, now)
        if self.heartbeat is not None and any(m.decider is not None for m in self.models.models):
            self.heartbeat.tick(now)

    def _relieve(self) -> None:
        """Between batches, on a memory reading not yet acted on: while memory is short, give back, in order, kept
        prefixes (each model's share cut to SOFT_KEEP, to none while very short), MLX's buffer cache, and, once the
        shortage outlasts a reading (at once while very short), idle models (`_unload_idle`); once memory is plenty
        again, the prefixes' shares come back. Nothing here changes an answer: a request whose prefix was dropped
        reads it again, and an unloaded model loads again on its next request."""
        monitor = self.monitor
        if monitor is None or monitor.readings == self._seen:
            return
        seen, level = monitor.latest()
        try:
            self._give_back(level)
        finally:
            self._seen = seen  # once acted on: /v1/status and tests see the steps taken

    def _give_back(self, level: Level) -> None:
        if level is Level.NORMAL:
            if self.models.prefix_scale < 1:
                self.models.prefix_scale = 1.0
                self.models.share()
                if any(m.decider is not None and m.decider.pinned for m in self.models.models):
                    pin()  # wired again, as far as memory now allows
                self._gave(level, "restore", 0)
            self._spell = 0
            return
        release()  # what is wired can be paged out while memory is short
        scale = min(self.models.prefix_scale, 0.0 if level is Level.HARD else SOFT_KEEP)
        lowered = scale < self.models.prefix_scale
        if lowered:
            self.models.prefix_scale = scale
            self._gave(level, "prefixes", self.models.share())
        cached = mx.get_cache_memory()
        mx.clear_cache()
        if lowered or cached:
            self._gave(level, "cache", cached)
        if self._spell or level is Level.HARD:
            self._unload_idle(level)
        self._spell += 1

    def _unload_idle(self, level: Level) -> None:
        """Unload loaded models no waiting request names, least recently used first: while memory is short one per
        reading; while it is very short, all of them at once; either way keeping the most recently used."""
        with self._lock:  # admission cannot add a waiting request between this snapshot and unloading
            loaded = sorted(
                (m for m in self.models.models if m.decider is not None and m.name in self.models.loaded_here),
                key=lambda m: m.used,
            )
            wanted = self._wanted()
            idle = [m for m in loaded if id(m) not in wanted]
            idle = [m for m in idle if m is not loaded[-1]]
            if level is Level.SOFT:
                idle = idle[:1]
            for m in idle:
                before = footprint()
                self.models.unload(m)
                self._gave(level, "unload", max(0, before - footprint()), m.name)

    def _wanted(self) -> set[int]:
        """The models (by id) that requests held or waiting in the queue name."""
        with self._queue.mutex:
            waiting = [*self._held, *self._queue.queue]
        return {id(job.model) for job in waiting if not job.future.cancelled()}

    def _gave(self, level: Level, step: str, nbytes: int, model: str | None = None) -> None:
        """Record a give-back (or a restore) for /v1/status."""
        event = {"at": round(time.time(), 3), "pressure": str(level), "step": step, "bytes": nbytes}
        self.given_back.append(event if model is None else {**event, "model": model})

    def _abandon(self, first: _Job | None) -> None:
        """Fail every request still waiting (closing, past its grace): Unavailable, to send again."""
        jobs = ([first] if first is not None else []) + list(self._held)
        self._held.clear()
        while True:
            try:
                jobs.append(self._queue.get_nowait())
            except queue.Empty:
                break
        error = Unavailable("the server stopped before the model started on it: send it again")
        for job in jobs:
            _fail(job, error)

    def _loop(self) -> None:
        while True:
            first = self._next(0.05)  # wakes up every 50 ms to see whether the batcher is closing
            if self._closing.is_set() and time.monotonic() >= self._give_up:
                self._abandon(first)
                return
            if first is None:
                if self._closing.is_set() and self._queue.empty() and not self._held:
                    return
                self._idle()
                continue
            model = first.model
            media_limit = min(self.max_batch, MAX_MEDIA_BATCH)
            has_media = _has_media(first)
            limit = media_limit if has_media else self.max_batch
            jobs = [first]
            for held in list(self._held):
                if held.future.cancelled():
                    continue
                if held.model is not model or len(jobs) >= limit:
                    continue
                held_media = _has_media(held)
                if held_media and len(jobs) >= media_limit:
                    continue
                if held_media:
                    has_media = True
                    limit = media_limit
                jobs.append(held)
            self._held = deque(j for j in self._held if j not in jobs and not j.future.cancelled())
            until, scanned = time.monotonic() + self.window, 0
            while (
                len(jobs) < limit
                and scanned < self.max_batch  # another model's requests held per batch: bounded
                and (more := self._take(max(0.0, until - time.monotonic())))
            ):
                if more.model is model:
                    more_media = _has_media(more)
                    if more_media and len(jobs) >= media_limit:
                        self._held.append(more)
                        break
                    if more_media:
                        has_media = True
                        limit = media_limit
                    jobs.append(more)
                else:
                    scanned += 1
                    self._held.append(more)  # another model's: answered after this batch, still counted
            model.used = time.monotonic()  # the model being read is never the idle one relief unloads mid-reading
            try:
                self._answer(self.models.ready(model), jobs)
            except Exception as error:  # a failure outside any one request: those of this batch get it
                for j in jobs:
                    _fail(j, error)
            model.used = time.monotonic()  # answered: keep_alive counts from here
            if self.heartbeat is not None:
                self.heartbeat.busy(model.used)
            if self.monitor is not None:
                self.monitor.active()
                self._relieve()

    def _run(self) -> None:
        budget = self.models.budget if self.models.enforce_memory else None
        prev_limits: tuple[int, int] | None = None
        if budget is not None:
            # Cap MLX's allocator and Metal's free buffer cache to the server budget (default MLX allows 1.5x working
            # set) and restore the previous limits when this server exits.
            prev_limits = (
                mx.set_memory_limit(budget),
                mx.set_cache_limit(min(budget, max(budget // 4, 256 << 20))),
            )
        try:
            self._loop()
        finally:
            with self._lock:
                self._closing.set()
            self._abandon(None)
            self.models.close()
            if prev_limits is not None:
                mx.set_memory_limit(prev_limits[0])
                mx.set_cache_limit(prev_limits[1])

    def _answer(self, decider: Decider, jobs: list[_Job]) -> None:
        started = [job for job in jobs if self._start(job)]  # not cancelled while waiting
        if not started:
            return
        self.batches.append(len(started))
        self.answered += 1
        futures = [job.future for job in started]
        budget = self.models.budget if self.models.enforce_memory else None

        def stop() -> DecisionError | None:
            if all(f in self._abandoned for f in futures):
                return Abandoned("nobody waits for this answer any more: stopped before its next pass")
            # memory short while a long reading runs: relief between its passes, not only once it ends (a reading
            # of minutes held a Mac at its memory warning, given back nothing)
            self._relieve()
            if budget is not None and mx.get_active_memory() > budget:
                mx.clear_cache()
                active = mx.get_active_memory()
                if active > budget:
                    return InsufficientMemory(
                        f"request exceeded the server's {budget / 2**30:.1f} GiB memory budget "
                        f"({active / 2**30:.1f} GiB active): send fewer images or a shorter prompt"
                    )
            return None

        token = STOP.set(stop)
        try:
            answered = answer_each(decider, [job.request for job in started])
        finally:
            STOP.reset(token)
            if budget is not None and mx.get_active_memory() + mx.get_cache_memory() > budget:
                mx.clear_cache()
        for job, got in zip(started, answered, strict=True):
            if isinstance(got, Exception):
                job.future.set_exception(got)
            else:
                job.future.set_result(got)


def _too_large(limit: int) -> TooLarge:
    return TooLarge(f"body size: more than the server's {limit >> 20} MiB (ippon serve --max-body-mb)")


# A compressed body may expand this many times at most: JSON with pictures or audio in base64 barely compresses, and
# text JSON compresses tens of times, while a crafted body expands about 32,700 times (zlib's most) to the body limit
MAX_EXPANSION = 1000
# Compressed bodies decoding at once: each may hold the body limit while it decodes, so many at once could take more
# memory than the Mac has; the rest wait their turn
DECODING = threading.BoundedSemaphore(2)
READ_TIMEOUT_S = 30  # the longest a client may leave the server waiting for the next bytes of its request


def _decoded(data: bytes, encoding: str, limit: int) -> bytes:
    """A request body as sent with Content-Encoding `encoding`, decoded, at most `limit` bytes: identity, gzip and
    deflate (zlib), and zstd where this Python reads it (3.14's compression.zstd, or backports.zstd). Raises
    UnsupportedEncoding for another encoding, TooLarge past `limit`, and InvalidRequest for a body that doesn't
    decode."""
    if encoding in ("", "identity"):
        return data
    if encoding in ("gzip", "x-gzip", "deflate"):
        decoder: Any = zlib.decompressobj(wbits=47)  # 32 + 15: a gzip or a zlib stream, told by its header
        failed: tuple[type[Exception], ...] = (zlib.error,)
    elif encoding == "zstd":
        for module in ("compression.zstd", "backports.zstd"):
            try:
                zstd = importlib.import_module(module)
                break
            except ImportError:
                continue
        else:
            raise UnsupportedEncoding(
                "Content-Encoding zstd needs Python 3.14 or backports.zstd here: send the body uncompressed"
            )
        decoder, failed = zstd.ZstdDecompressor(), (zstd.ZstdError,)
    else:
        raise UnsupportedEncoding(f"Content-Encoding {encoding!r}: send the body as it is, or in gzip or zstd")
    most = min(limit, len(data) * MAX_EXPANSION)
    with DECODING:
        try:
            out = decoder.decompress(data, most + 1)
        except failed as error:
            raise InvalidRequest(f"the body is not valid {encoding}: {error}") from None
    if len(out) > limit:
        raise _too_large(limit)
    if len(out) > most:
        raise TooLarge(f"the body expands more than {MAX_EXPANSION} times its {encoding} size: send it less compressed")
    if not decoder.eof:
        raise InvalidRequest(f"the body is not valid {encoding}: it ends early")
    return bytes(out)


def _local_host(host: str | None) -> bool:
    """Whether a request's Host header names this Mac's loopback (localhost, 127.0.0.1 or [::1], with or without a
    port), as a request sent to a server on a loopback address does unless a web page rebound a domain to it."""
    name = (host or "").strip().lower()
    return any(name == h or (name.startswith(h + ":") and name[len(h) + 1 :].isdigit()) for h in LOCAL_HOSTS)


def _bearer(authorization: str | None, key: str) -> bool:
    """Whether an Authorization header is `Bearer KEY`, compared in constant time."""
    scheme, _, token = (authorization or "").strip().partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), key.encode())


class _Answering:
    """How many requests the server's handler threads are answering: a closing server waits for them (`wait`)."""

    def __init__(self) -> None:
        self._count, self._changed = 0, threading.Condition()

    def __enter__(self) -> None:
        with self._changed:
            self._count += 1

    def __exit__(self, *exc: object) -> None:
        with self._changed:
            self._count -= 1
            self._changed.notify_all()

    def wait(self, timeout: float) -> bool:
        """Whether every answer was sent within `timeout` seconds."""
        with self._changed:
            return self._changed.wait_for(lambda: self._count == 0, timeout)


def _terminate(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt  # as Ctrl-C does: whoever called serve_forever closes the server


class Server(ThreadingHTTPServer):
    """The HTTP server `make_server` returns. On the main thread, `serve_forever` stops on SIGTERM as on Ctrl-C (its
    caller's KeyboardInterrupt handling runs), so a launcher can stop it; `server_close` stops taking connections,
    closes the batcher (the requests waiting are answered for CLOSE_GRACE_S, the rest refused as unavailable) and
    waits up to SEND_S for the answers to be sent."""

    daemon_threads = True
    # Connections the kernel holds until they are accepted (macOS caps it at kern.ipc.somaxconn, 128): socketserver's
    # 5 reset the 6th of 8 clients connecting while a long prompt kept this thread from accepting (M5 Pro, 8 clients
    # of 1,000-word requests: 2 of 8 reset).
    request_queue_size = MAX_QUEUE
    batcher: Batcher
    answering: _Answering

    def server_bind(self) -> None:
        """Binds as socketserver does, without HTTPServer's lookup of the host's fully qualified name: that reverse
        DNS lookup can stall for seconds (it hung on GitHub's macOS runners), and nothing here reads `server_name`."""
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), int(self.server_address[1])

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        if threading.current_thread() is not threading.main_thread():
            super().serve_forever(poll_interval)
            return
        previous = signal.signal(signal.SIGTERM, _terminate)
        try:
            super().serve_forever(poll_interval)
        finally:  # a second SIGTERM while closing ends the process as it would have before
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)

    def server_close(self) -> None:
        super().server_close()
        if hasattr(self, "batcher"):  # TCPServer also calls this if binding fails, before make_server starts a batcher
            self.batcher.close(CLOSE_GRACE_S)
            self.answering.wait(SEND_S)


@dataclass(frozen=True)
class Spec:
    """A model `ippon serve` is given: the name requests use, what it loads (a short name resolved, once) and the
    precision it loads at (one its name carries, kettei-core:q8, over the server's --precision)."""

    name: str
    loads: str
    precision: str


def parse_served(spec: str) -> tuple[str, str]:
    """A served model as `serve` names it: NAME=MODEL, or a model named as given (a folder whose name has "=" too)."""
    if "=" in spec and not Path(spec).exists():
        name, _, model = spec.partition("=")
        return name, model
    return spec, spec


def specs(given: Sequence[str], precision: str) -> list[Spec]:
    """The models `ippon serve MODEL...` is given (`parse_served`), each at `precision` unless its name carries one."""
    out = []
    for spec in given:
        name, model = parse_served(spec)
        out.append(Spec(name, str(names.resolve(model)), names.named_precision(model, precision)))
    return out


def _hold(host: str, port: int) -> socket.socket | None:
    """A socket bound to `host`:`port`, so no other process takes the port while a server loads its first model; None
    for port 0 (any free port, chosen as the server binds)."""
    if not port:
        return None
    held = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    try:
        held.bind((host, port))
    except OSError:
        held.close()
        raise DecisionError(f"port {port} on {host} is in use: give another --port, or leave it out") from None
    return held


def run(
    served: Sequence[Spec],
    options: Mapping[str, Any],
    memory_gb: float | None,
    host: str,
    port: int,
    listening: Callable[[str, int, Decider], None],
    **server: Any,
) -> None:
    """`ippon serve`: `served` answered on `host`:`port` until Ctrl-C or SIGTERM, each loaded with `Decider.load`'s
    `options` at its own precision, within `memory_gb` (default: what admission.py allows one model on this Mac).
    The first loads now, so a model that can't load fails the start; `listening` hears where the server listens and
    the first model, once it does. `server` holds make_server's options."""
    reads = Reads.of(options)
    models = []
    for s in served:
        loading = {**options, "precision": s.precision, "keep_awake": None}  # the server's heartbeat keeps it awake
        needed = locate.estimate(
            s.loads, precision=s.precision, settings=options.get("settings"), accept_license=options["accept_license"]
        )
        load = functools.partial(Decider.load, s.loads, **loading)
        weights = names.weights_at(s.loads, s.precision)
        found = names.catalogued(s.loads)
        known_media = tuple(sorted(kind for kind, _ in found[1].media)) if found is not None else ()
        models.append(Model(s.name, load, needed, s.loads, weights, reads, s.precision, media=known_media))
    device = profiles.resolve(options.get("profile")).device
    limit = (
        int(memory_gb * 2**30)
        if memory_gb is not None
        else (budget(device.working_set_bytes) if device and device.working_set_bytes else None)
    )
    registry = Models(models, limit, enforce_memory=True)
    held = _hold(host, port)  # the port stays ours through the seconds the first model takes to load
    try:
        first = registry.ready(models[0])  # fail at startup if the first model cannot load
        if held is not None:
            held.close()
        http, batcher = make_server(registry, host, port, **server)
    except BaseException:
        if held is not None:
            held.close()
        registry.close()
        raise
    try:
        bound, at = http.server_address[:2]
        listening(str(bound), int(at), first)
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()
        batcher.close()
        registry.close()


class _Traffic:
    """What a server's clients are doing now, for its status: the connections open (ippon's own clients keep theirs
    between requests, so each attached agent counts once) and the answers sent in the last WINDOW_S seconds. A
    server reading long requests answers in bursts as each pass ends, so the window spans several passes.

    A request asks one question or many, and one question may weigh hundreds of options, so its status counts the
    questions and the tokens read beside the requests: ten questions a commit at 1.9 requests a second were 19 answers
    a second, and the same server read about 3,000 tokens a second through both (Kettei Core, 8 clients, M5 Pro)."""

    WINDOW_S = 30.0  # long enough to hold a few passes of a busy server, which answers a batch at a time
    BURST_S = 5.0  # the shortest span a rate is taken over: one request's answers alone are not a rate
    CLIENTS = 16  # client names it counts apart; answers for any more count as "others"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.open = 0
        self.asked = 0  # requests received and not yet answered: waiting for the model, or being read
        self.asking_by: Counter[str] = Counter()  # those requests by the client asking ("" for one that gave no name)
        self.last: tuple[float, float] | None = None  # (when, ms per decision) at the last answer with a time
        self.total = 0
        self.questions = 0  # the questions those requests asked
        # (when, its questions, its tokens, the model's ms per decision) for each answer
        self.recent: deque[tuple[float, int, int, float | None]] = deque()
        self.tokens = 0  # the input tokens of every answer, as each response's usage counts them
        self.clients: Counter[str] = Counter()  # questions answered by the name a client gave (CLIENT_HEADER)

    def opened(self) -> None:
        with self.lock:
            self.open += 1

    def closed(self) -> None:
        with self.lock:
            self.open -= 1

    def asking(self, change: int, client: str | None = None) -> None:
        with self.lock:
            self.asked += change
            self.asking_by[client or ""] += change
            if self.asking_by[client or ""] <= 0:
                del self.asking_by[client or ""]

    def answered(self, client: str | None = None, tokens: int = 0, questions: int = 1, ms: float | None = None) -> None:
        """An answer sent, with the model's time for it in `ms` (its rendering and pass), when known."""
        with self.lock:
            self.total += 1
            self.questions += questions
            self.tokens += tokens
            if client is not None:
                named = client in self.clients or len(self.clients) < self.CLIENTS
                self.clients[client if named else "others"] += questions
            self.recent.append((time.monotonic(), questions, tokens, None if ms is None else ms / max(questions, 1)))
            each = [e for *_, e in self.recent if e is not None]
            if each:
                self.last = (time.monotonic(), round(statistics.median(each), 1))

    def now(self) -> dict[str, Any]:
        """The model's median ms per decision over the window (once the window is empty, the last one, with how long ago
        the answer it was last taken at came: a server gone quiet keeps its speed on screen), its connections besides
        the one asking, its requests in
        progress (waiting or being read: a server reading a long batch answers none for a while, yet is busy), its
        requests, questions and tokens a second over the window
        (from the oldest answer still in it to the newest, at least BURST_S: a job that began seconds ago isn't read
        low, as a job answering about 13 questions a second showed 8.5 over the whole window, and a job that just
        ended keeps its own pace until its answers leave the window, where measuring to now showed it slide to 1.2
        beside "0 in progress"), and every answer so far."""
        with self.lock:
            now = time.monotonic()
            while self.recent and self.recent[0][0] < now - self.WINDOW_S:
                self.recent.popleft()
            covered = max(self.recent[-1][0] - self.recent[0][0], self.BURST_S) if self.recent else self.WINDOW_S
            each = [e for *_, e in self.recent if e is not None]
            quiet = None if each or self.last is None else round(now - self.last[0], 1)
            return {
                "ms_per_decision": round(statistics.median(each), 1) if each else self.last and self.last[1],
                **({"last_answer_s": quiet} if quiet is not None else {}),
                "connections": max(self.open - 1, 0),
                "in_progress": self.asked,
                "clients_asking": len(self.asking_by),
                "answers_per_second": round(len(self.recent) / covered, 1),
                "questions_per_second": round(sum(q for _, q, *_ in self.recent) / covered, 1),
                "tokens_per_second": round(sum(t for _, _, t, _ in self.recent) / covered),
                "answered": self.total,
                "questions": self.questions,
                "tokens": self.tokens,
                "by_client": dict(self.clients.most_common()),
            }


CLIENT_CHARS = 40


def client_name(value: str | None) -> str | None:
    """The name a client gave in CLIENT_HEADER, as the log and status show it: letters, digits and - _ . : / only, at
    most CLIENT_CHARS of them; None when it gave none."""
    name = "".join(c for c in (value or "") if c.isascii() and (c.isalnum() or c in "-_.:/"))[:CLIENT_CHARS]
    return name or None


WAIT_SHOWN_MS = 100.0  # a wait behind other requests this long is said on the answer's line


def duration(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def answered_line(
    response: Mapping[str, Any],
    ms: float,
    client: str | None = None,
    kept: int = 0,
    model: str | None = None,
    colour: bool = False,
    width: int | None = None,
    waited: float = 0.0,
) -> str:
    """The lines for a person watching the server, in columns that line up down the screen: when, which `model` (given
    when the server has several), for which client (by the name it gave), how long the model took per decision (the
    request's `ms` less the time it `waited`, which is neither its rendering nor its pass: the queue and the HTTP
    exchange, shared among its questions), how many tokens it read and how many of them it `kept` from an
    earlier request, the wait when it is WAIT_SHOWN_MS or more, then each question's answer as `ippon decide` says
    it, the answer itself in the accent for what is settled. At a terminal (`width`) the answers go on as many lines
    as they need, each answer whole and under the first, so ten questions about a commit can all be read. Never the
    state's text."""
    tokens = response.get("usage", {}).get("input_tokens")
    queued = waited >= WAIT_SHOWN_MS  # a busy server's line also says how long the request waited
    took = duration((ms - waited) / max(1, len(response.get("answers", {}))))
    read = f"{tokens:,} tok" if isinstance(tokens, int) else ""
    head: list[tuple[str, tuple[str, ...]]] = [(time.strftime("%H:%M:%S"), ("2",))]
    if model:
        head.append((f"  {model}", ("2",)))
    if client:
        head.append((f"  {client:<9}", ()))
    head.append((f"  {took:>7}", ()))
    head.append((f"  {read:>10}", ("2",)))
    if kept:
        head.append((f" · {kept:,} kept", ("pink",)))
    if queued:
        head.append((f" · {duration(waited)} queued", ("2",)))
    answers: list[list[tuple[str, tuple[str, ...]]]] = []
    for qid, answer in response.get("answers", {}).items():
        said, _, sure = terminal.said(answer).rpartition(" ")
        answers.append([(f"   {qid} ", ("2",)), (said, ("1", "mint")), (f" {sure}", ("2",))])
    # more answers go under the first, or, where that leaves them under half the terminal, under the client's name
    indent = sum(len(text) for text, _ in head)
    under = indent if width is None or indent <= width // 2 else len(head[0][0])
    lines, line, used, placed = [], head, indent, 0  # placed: answers on this line so far
    for parts in answers:
        size = sum(len(text) for text, _ in parts)
        if width is not None and used + size > width and placed:
            lines.append(line)
            line, used, placed = [(" " * under, ())], under, 0
        line, used, placed = [*line, *parts], used + size, placed + 1
    lines.append(line)
    return "\n".join(terminal.fitted(parts, colour, width) for parts in lines)


def make_server(
    decider: Decider | Models,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    max_batch: int = MAX_BATCH,
    log: bool = False,
    window_ms: float = 0.0,
    max_queue: int = MAX_QUEUE,
    timeout_s: float | None = TIMEOUT_S,
    *,
    api_key: str | None = None,
    allow_origins: Sequence[str] = (),
    heartbeat: Heartbeat | None = None,
    keep_alive: float | None = None,
    monitor: Monitor | bool = True,
    max_body: int | None = None,
) -> tuple[Server, Batcher]:
    """An HTTP server for `decider` or several (not yet serving: call `serve_forever()`), and the batcher behind it.
    With `api_key`, requests other than OPTIONS need `Authorization: Bearer KEY`. `allow_origins` lists web origins that
    may call it ("*": any); `heartbeat` and `keep_alive` are the batcher's (`Batcher`), and so is `monitor`, which
    reads this Mac's memory (True, the default: `pressure.Monitor()`; False: none). `max_body` is the most bytes a
    request body may hold, compressed or decoded (default: `device.max_body`, by this Mac's memory)."""
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s is above 0, or None for no deadline")
    origins, key = frozenset(allow_origins), api_key or None
    body_limit = max_body if max_body is not None else device_max_body()
    if body_limit <= 0:
        raise ValueError("max_body is above 0")
    answering = _Answering()
    traffic = _Traffic()

    def models() -> dict[str, Any]:
        """Every model the server answers for: as OpenAI's list (`data`, with a loaded one's adapter, chip profile
        and details) and as TypeSafe's SDK reads it (`models`: a name, a description, and a release date, which a
        server of downloaded weights doesn't know: "")."""
        data, listed = [], []
        for m in batcher.models.models:
            d = m.decider
            data.append(m.listed().to_json())
            about = f"{d.adapter}, on ippon" if d is not None else "on ippon, loaded on its first request"
            listed.append({"name": m.name, "description": about, "release_date": ""})
        return {"object": "list", "data": data, "models": listed}

    def status() -> dict[str, Any]:
        """The server's memory: its footprint, its budget, each model's weights and kept prefixes (bytes held, and
        the share they may hold), the pressure the monitor last read and what pressure took lately."""
        monitor, served = batcher.monitor, batcher.models
        memory: dict[str, Any] = {"footprint": footprint() or None, "budget": served.budget}
        memory |= monitor.status() if monitor is not None else {"pressure": "unknown", "monitor_error": "disabled"}
        listed = []
        for m in served.models:
            d = m.decider
            prefixes = d.prefixes if d is not None else None
            listed.append(
                {
                    "name": m.name,
                    "loaded": d is not None,
                    "weights": m.needed,
                    "prefixes": prefixes.nbytes if prefixes is not None else 0,
                    "prefix_budget": prefixes.budget if prefixes is not None else 0,
                    "tokens_reused": prefixes.stats.tokens_reused if prefixes is not None else 0,
                }
            )
        state = "ok" if batcher.alive else "unavailable"
        given = list(batcher.given_back.copy())
        return {
            "status": state,
            "waiting": batcher.waiting,
            "memory": memory,
            "models": listed,
            "given_back": given,
            "traffic": traffic.now(),
        }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ippon"
        origin: str | None = None  # the request's Origin, when allowed: its response carries CORS headers
        # a client that stops sending mid-request frees its thread after this, never holds it open (300 half-sent
        # requests held 330 threads)
        timeout = READ_TIMEOUT_S

        def setup(self) -> None:
            super().setup()
            traffic.opened()

        def finish(self) -> None:
            traffic.closed()
            super().finish()

        def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
            """The standard library's own refusals (a method with no handler, a malformed request line or header) as
            the JSON error every other refusal is, never its HTML page."""
            if code == HTTPStatus.NOT_IMPLEMENTED:  # a method this server has no handler for
                self._fail(MethodNotAllowed(f"{self.command} is not a method this server takes"), close=True)
                return
            error = InvalidRequest(message or HTTPStatus(code).phrase)
            self._send(code, error.to_json(), close=True)

        def _send(self, status: int, body: Mapping[str, Any], close: bool = False, retry: str | None = None) -> None:
            """Send a JSON response; with `close`, end the connection after it (a body left unread). A 503 asks its
            caller to wait `retry` seconds (default 1)."""
            data = json.dumps(body, ensure_ascii=False, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if status == 503:
                self.send_header("Retry-After", retry or "1")
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            if status == 405:
                path = getattr(self, "path", "").partition("?")[0]
                self.send_header("Allow", "GET, OPTIONS" if path in GOT else "POST, OPTIONS")
            self._cors()
            if close:
                self.close_connection = True
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def _cors(self) -> None:
            if self.origin is not None:
                self.send_header("Access-Control-Allow-Origin", self.origin)
                self.send_header("Access-Control-Expose-Headers", "Retry-After")
                self.send_header("Vary", "Origin")

        def _fail(self, error: BaseException, close: bool = False) -> None:
            """Send a request's error as its status and body (`errors.as_decision_error`); a caller that hung up while
            its answer was written gets nothing more."""
            if isinstance(error, (BrokenPipeError, ConnectionResetError)):
                self.close_connection = True
                return
            failed = as_decision_error(error)
            self._send(failed.http_status, failed.to_json(), close=close, retry=failed.retry_after)
            if log:
                sys.stderr.write(f"{time.strftime('%H:%M:%S')}  {failed.http_status} {self.path}: {failed}\n")

        def _refuse(self, error: DecisionError) -> None:
            """Send `error` and end the connection: the request's body is left unread."""
            self._fail(error, close=True)

        def _admitted(self) -> bool:
            """Whether the request may go on; if not, it is refused: a Host other than the loopback's on a server
            listening on a loopback address (a web page that rebound its domain to this Mac), an Origin not allowed
            (403), or without the API key when there is one (401). A preflight (OPTIONS) needs no key."""
            self.origin = None
            if loopback and not _local_host(self.headers.get("Host")):
                host = self.headers.get("Host")
                self._refuse(Forbidden(f"Host {host!r}: this server answers localhost, 127.0.0.1 or [::1]"))
                return False
            origin = self.headers.get("Origin")
            if origin is not None:
                if "*" not in origins and origin not in origins:
                    self._refuse(Forbidden(f"Origin {origin!r} may not call this server (--allow-origin)"))
                    return False
                self.origin = origin
            if self.command != "OPTIONS" and key is not None and not _bearer(self.headers.get("Authorization"), key):
                self._refuse(
                    Unauthorized(
                        "this server needs its API key: Authorization: Bearer KEY (ippon's clients send IPPON_API_KEY)"
                    )
                )
                return False
            return True

        def _hung_up(self) -> bool:
            """Whether the caller has closed its connection (EOF, or reset), looking without reading or waiting: a
            socket with a timeout (READ_TIMEOUT_S) waits that long in recv for bytes, whatever its flags, so `select`
            looks first."""
            if not select.select([self.connection], [], [], 0)[0]:
                return False  # nothing to read: still there
            try:
                return not self.connection.recv(1, socket.MSG_PEEK)
            except (BlockingIOError, TimeoutError):
                return False
            except OSError:
                return True

        def _wait(self, future: Future[Result]) -> Result | None:
            """The answer, or None when the caller hung up before it came (the request is then cancelled: unread,
            unless the model has started on it); raises FutureTimeout past `timeout_s`, if any."""
            until = math.inf if timeout_s is None else time.monotonic() + timeout_s
            while (left := until - time.monotonic()) > 0:
                try:
                    return future.result(timeout=min(POLL_S, left))
                except FutureTimeout:
                    if self._hung_up():
                        batcher.abandon(future)
                        return None
            raise FutureTimeout

        def do_OPTIONS(self) -> None:
            if not self._admitted():
                return
            self.send_response(204)
            self.send_header("Allow", "GET, POST, OPTIONS")
            if self.origin is not None:  # a page's preflight
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                asked = self.headers.get("Access-Control-Request-Headers")
                self.send_header("Access-Control-Allow-Headers", asked or "Authorization, Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
                self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            if not self._admitted():
                return
            if self.path == "/health":
                if batcher.alive:
                    self._send(200, {"status": "ok", "waiting": batcher.waiting})
                else:
                    self._send(503, {"status": "unavailable", "waiting": batcher.waiting})
            elif self.path == "/v1/models":
                self._send(200, models())
            elif self.path == "/v1/status":
                self._send(200, status())
            elif self.path.partition("?")[0] in POSTED:  # a browser or a bare curl: say how it is asked
                self._fail(MethodNotAllowed(f"{self.path} takes a POST with a JSON request body"))
            else:
                self._fail(NotFound(f"no route {self.path}"))

        def do_POST(self) -> None:
            if not self._admitted():
                return
            path, _, query = self.path.partition("?")
            if path in GOT:
                self._refuse(MethodNotAllowed(f"{path} takes a GET"))
                return
            if path not in POSTED:
                self._refuse(NotFound(f"no route {self.path}"))
                return
            encoding = (self.headers.get("Content-Encoding") or "").strip().lower()
            try:
                body = read_json(_decoded(self._body(body_limit), encoding, body_limit))
            except InvalidRequest as error:
                self._refuse(error)  # a body refused for its size is left unread: the connection ends with it
                return
            with answering:
                self._answer(path, query, body)

        def _body(self, limit: int) -> bytes:
            """The request's body as sent, by its Content-Length or in chunks (Transfer-Encoding: chunked), at most
            `limit` bytes; TooLarge past it, InvalidRequest for a body that doesn't say its size."""
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                parts, size = [], 0
                while True:
                    line = self.rfile.readline(64).split(b";")[0].strip()
                    if not line or line.strip(b"0123456789abcdefABCDEF"):  # hex digits only: no sign, no 0x
                        raise InvalidRequest("a chunk of the body doesn't start with its size")
                    chunk = int(line, 16)
                    if chunk == 0:
                        while self.rfile.readline(1024).strip():  # trailers
                            pass
                        break
                    size += chunk
                    if size > limit:
                        raise _too_large(limit)
                    parts.append(self.rfile.read(chunk))
                    self.rfile.readline(8)  # the chunk's CRLF
                if not size:
                    raise InvalidRequest("body size")
                return b"".join(parts)
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise InvalidRequest("Content-Length is not a number") from None
            if length <= 0:
                raise InvalidRequest("body size")
            if length > limit:
                raise _too_large(limit)
            return self.rfile.read(length)

        def _answer(self, path: str, query: str, body: Any) -> None:
            client = client_name(self.headers.get(CLIENT_HEADER))
            traffic.asking(+1, client)
            try:
                self._answering(path, query, body)
            finally:
                traffic.asking(-1, client)

        def _answering(self, path: str, query: str, body: Any) -> None:
            begun = time.monotonic()
            try:
                sglang: Any = None
                native: tuple[list[decide_api.Decision], float | None] | None = None
                if path == "/v1/decisions" and is_sglang(body):
                    body, sglang = from_sglang(body)
                if path == "/v1/decide":  # Sqwish's decide API, as the sqwishai SDK calls it
                    body, decisions, temperature = decide_api.to_systemone(body)
                    native = (decisions, temperature)
                future = batcher.submit(body)
                result = self._wait(future)
                if result is None:  # the caller has gone: nobody to answer
                    self.close_connection = True
                    return
                if native is not None:
                    decisions, temperature = native
                    model = str(body.get("model") or result.response.get("model") or "")
                    answer = decide_api.response(result, decisions, temperature, model)
                elif sglang is not None:
                    answer = sglang_response(result.answers, result.response, sglang)
                else:
                    answer = result.response
                if "diagnostics=1" in query.split("&"):  # how it was answered, when asked
                    answer = {**answer, "diagnostics": result.diagnostics}
            except FutureTimeout:
                batcher.abandon(future)  # still waiting: out of the queue; being read: stopped at its next pass
                self._fail(Timeout(f"not answered within {timeout_s:g} s: the server is busy, send it again later"))
                return
            except Exception as error:
                self._fail(error)
                return
            try:
                self._send(200, answer)  # once sent, nothing below may answer the request a second time
            except (BrokenPipeError, ConnectionResetError):  # the caller hung up while its answer was written
                self.close_connection = True
                return
            try:
                client = client_name(self.headers.get(CLIENT_HEADER))
                tokens = result.response.get("usage", {}).get("input_tokens")
                model_ms = sum(
                    result.timing_ms.values()
                )  # its rendering and pass: neither the queue nor the HTTP exchange
                traffic.answered(
                    client, tokens if isinstance(tokens, int) else 0, len(result.answers), model_ms or None
                )
                if log:
                    # the name it answered by, as the server lists it, when it serves more than one
                    served = batcher.models.named(body).name if len(batcher.models.models) > 1 else None
                    ms = (time.monotonic() - begun) * 1000
                    colour = terminal.colour_on(sys.stderr)
                    width = shutil.get_terminal_size().columns - 1 if sys.stderr.isatty() else None
                    waited = ms - model_ms  # what was neither its rendering nor its pass
                    line = answered_line(result.response, ms, client, result.kept_tokens, served, colour, width, waited)
                    sys.stderr.write(line + "\n")
            except Exception as error:
                sys.stderr.write(f"{time.strftime('%H:%M:%S')}  answered {self.path}, but not counted: {error!r}\n")

        def log_message(self, format: str, *args: Any) -> None:
            pass  # each answer and error has its own line (`answered_line`); the HTTP status line adds nothing

    server = Server((host, port), Handler)  # bound before the batcher starts: a port in use starts nothing
    loopback = is_loopback(str(server.server_address[0]))  # as bound: "localhost" is 127.0.0.1 by now
    watch = Monitor() if monitor is True else (monitor if isinstance(monitor, Monitor) else None)
    try:
        batcher = Batcher(
            decider, max_batch, window_ms, max_queue, heartbeat=heartbeat, keep_alive=keep_alive, monitor=watch
        )
    except BaseException:
        ThreadingHTTPServer.server_close(server)  # nothing of the batcher's to close
        raise
    server.batcher, server.answering = batcher, answering
    return server, batcher
