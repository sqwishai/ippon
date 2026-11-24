"""Keeping a loaded model fast between requests, without taking more of a shared Mac than that needs.

Two things make a request after a pause slower than one right after another, both measured on an M5 Pro with a
10 GB model:

- **Memory let go.** When the GPU has had nothing to do for a few seconds, macOS stops keeping the model's memory
  wired; the next command buffer waits while the driver wires it back (Instruments' Metal System Trace: hundreds of
  "Wire Memory" events, 66 ms before the GPU starts), and a 500-token request took 227 ms instead of 141 after 5 s
  idle. `pin()` raises MLX's wired limit to cover the model, so its buffers stay wired, and caps MLX's buffer cache
  so a long prompt's activations don't stay behind. A trivial GPU command every second keeps the memory wired for
  requests that come with pauses between them: `Heartbeat` sends it from a server's model thread between passes,
  for a while after the last request; `KeepAwake` from a thread of its own, for one decider (`Decider.load`'s
  `keep_awake`).
- **The clock.** The GPU's clock falls within half a second of idle and the next request pays about 12 ms; nothing
  short of real load holds it up, so nothing here tries.

MLX doesn't promise that one thread may run operations while another traces a compiled function, so neither command
runs while a model does: `KeepAwake` waits for the decider's `lock`, and `Heartbeat` runs on the one thread that runs
the models, so a server of several models (each with a lock of its own) never sends it during any model's pass.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from . import pressure
from .options import AWAKE_WINDOW_S, PERIOD_S

WIRED_HEADROOM = 8 << 30  # bytes wired beyond what the model holds after loading: its caches and activations
WIRED_SHARE = 0.9  # of the GPU working set macOS recommends, at most wired
RAM_SHARE = 0.5  # of the Mac's memory at most wired, whatever the working set
CACHE_SHARE = 0.10  # of the GPU working set, at most kept as freed buffers for reuse
CACHE_CAP = 1 << 30  # bytes of freed buffers kept for reuse however large the Mac


def wired_limit(
    working_set: int,
    active: int,
    headroom: int = WIRED_HEADROOM,
    memory: int | None = None,
    reading: pressure.Reading | None = None,
) -> int:
    """The wired limit for models holding `active` bytes, from this Mac as it is now (`reading`): none while macOS
    reports memory pressure or the memory available is under the line at which memory counts as plenty again
    (`pressure.Thresholds.clear`), for wiring there would push the user's apps out; else room for the models and
    what their passes allocate (`headroom`, as far as the memory available above that line allows), within 90% of
    what macOS recommends the GPU keep resident (`working_set`) and half of the Mac's `memory`.

    Wired memory can't be paged out, so it is what other apps lose for certain. MLX wires only what it allocates, up
    to the limit: the weights, and beyond them what passes allocate (caches, activations, kept prefixes), for which
    the headroom leaves room. macOS recommends a working set of two thirds to three quarters of the Mac's memory
    (37.4 of 48 GiB on the M5 Pro), so the 90% bound alone would wire most of a small Mac for a model near its
    working set: Torchcast-4B's 9.7 GB on a 16 GB M5 (a 12 GB working set) wired 10.8 GB, and its kernel A/B, which
    wires and unwires that in a new process every round, ended in a kernel panic in Apple's GPU driver on macOS 26.2
    ("completeMemory() prepare count underflow", IOGPUMemory.cpp:550) while the Mac swapped. Half the Mac's memory
    is the most wired: 8 GB on 16 GB, 24 GB on 48 GB (where a 10 GB model still wires its 18 GB); the rest of a
    larger model is ordinary memory macOS manages, and also holds where `sysctl iogpu.wired_limit_mb` raised the
    working set to all of the Mac's memory."""
    if reading is not None:
        spare = reading.available - pressure.Thresholds.of(reading.memory).clear
        if reading.kernel >= pressure.WARNING or spare <= 0:
            return 0
        headroom = min(headroom, spare)
    limit = min(working_set * WIRED_SHARE, active + headroom)
    if memory:
        limit = min(limit, memory * RAM_SHARE)
    return int(limit)


def cache_limit(working_set: int, lower: int | None = None) -> int:
    """The bytes of freed buffers MLX keeps for reuse: 10% of the GPU working set, at most 1 GiB, and never more
    than a `lower` limit already set.

    MLX's own default keeps up to its memory limit (45.6 GiB on a 48 GiB M5 Pro), so after a long prompt the
    buffers its activations used stay with the process. Reused, they spare the next pass its allocations; kept
    past that, they are memory a shared Mac's other apps can't have. A request reuses the same few buffers, so
    little is enough: on an M5 Pro, 1 GiB against 10% of its working set (3.7 GiB) timed Clef-Flash 9B's MXFP8
    build within 0.6% on 4,000-token prompts and 8 requests answered together, and d1-2B and d3-mini within 1% at
    0.5 GiB, while no cache at all cost d1-2B 5% on long prompts. An 8 GB Mac (5.3 GiB) keeps about 0.5 GiB. A
    caller's lower limit (MLX's, set before, or `pin`'s `cache`) stays."""
    limit = min(CACHE_CAP, int(working_set * CACHE_SHARE))
    return limit if lower is None else min(limit, lower)


def pin(headroom: int = WIRED_HEADROOM, cache: int | None = None) -> int | None:
    """Wire what the loaded models hold and cap MLX's buffer cache (`wired_limit`, `cache_limit`, where `cache`
    caps it lower still); the wired limit set, or None without a GPU."""
    if not mx.metal.is_available():
        return None
    info = mx.device_info(mx.gpu)
    working_set, memory = int(info["max_recommended_working_set_size"]), int(info.get("memory_size") or 0)
    limit = wired_limit(working_set, mx.get_active_memory(), headroom, memory, pressure.now())
    mx.set_wired_limit(limit)
    ours = cache_limit(working_set, cache)
    current = mx.set_cache_limit(ours)  # MLX reads it back only by setting it
    if current < ours:  # set lower before (by its caller, or a `cache` given earlier): it stays
        mx.set_cache_limit(current)
    return limit


def release() -> None:
    """Wire nothing more and let macOS page what is wired: while memory is short (serve.py gives it back, with kept
    prefixes and idle models, and pins again once memory is plenty)."""
    if mx.metal.is_available():
        mx.set_wired_limit(0)


def _beat(stream: Any = mx.gpu) -> None:
    """The trivial GPU command: sixteen numbers added on `stream`, the GPU's default or a thread's own."""
    mx.eval(mx.add(mx.zeros((16,), stream=stream), 1, stream=stream))


class Heartbeat:
    """A trivial GPU command every `period` seconds, for `window` seconds after the last request (`busy`), then none
    until the next request: sent by a server's model thread between passes (`tick`), so never during one.

    Every second forever would wake the GPU 86,400 times a day for a server nobody calls; within five minutes of a
    request, it keeps the next one of a conversation or a pipeline fast and then lets the Mac rest."""

    def __init__(self, period: float = PERIOD_S, window: float = AWAKE_WINDOW_S, send: Callable[[], None] = _beat):
        if period <= 0 or window < 0:
            raise ValueError("period is a number of seconds above 0, window one of at least 0")
        self.period, self.window, self.send, self.beats = period, window, send, 0
        self._last = self._sent = time.monotonic()  # the start counts as a request

    def busy(self, now: float | None = None) -> None:
        """A request was just answered: the window starts again from `now`, and the next beat is a period away."""
        self._last = self._sent = time.monotonic() if now is None else now

    def tick(self, now: float | None = None) -> bool:
        """Send the command when one is due, on the thread that runs the models and between their passes; whether
        it was sent."""
        now = time.monotonic() if now is None else now
        if now - self._sent < self.period or now - self._last >= self.window:
            return False
        self.send()
        self._sent, self.beats = now, self.beats + 1
        return True


class KeepAwake:
    """`Heartbeat` for one decider: a trivial GPU command every `period` seconds for `window` seconds after its last
    answer (`busy`), from a thread and a stream of its own, never while `lock` is held by whoever runs the model. A
    server of several runs `Heartbeat` on its model thread instead, since one model's lock doesn't keep the command
    out of another's pass."""

    def __init__(self, lock: threading.Lock, period: float = PERIOD_S, window: float = AWAKE_WINDOW_S):
        self.lock, self.heartbeat = lock, Heartbeat(period, window, self._send)
        self._stream: Any = None  # the thread's own GPU stream, made on that thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ippon-keep-awake")
        self._thread.start()

    @property
    def beats(self) -> int:
        return self.heartbeat.beats

    def busy(self) -> None:
        """The decider just answered: the window starts again."""
        self.heartbeat.busy()

    def _send(self) -> None:
        assert self._stream is not None
        _beat(self._stream)

    def _run(self) -> None:
        self._stream = mx.new_thread_local_stream(mx.gpu)
        while not self._stop.wait(self.heartbeat.period):
            with self.lock:
                self.heartbeat.tick()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
