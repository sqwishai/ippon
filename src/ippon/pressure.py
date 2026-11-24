"""Noticing that this Mac is short of memory, without asking MLX or Metal.

A server shares its Mac with the apps its user runs. `Monitor` reads, on a thread of its own, two things: this
process's physical footprint (Activity Monitor's Memory column, the GPU's buffers included) and an estimate of
available memory: free pages, the file cache and purgeable pages, along with the kernel's own pressure level
(normal, warning or critical: what macOS tells apps).
A reading is three system calls (proc_pid_rusage, host_statistics64, sysctl), never MLX or Metal, so it runs while
the model thread is inside a pass. It reads every second while requests come (and while memory is short), every ten
seconds otherwise.

Levels, from the memory available (`Thresholds.of`):

- soft: under 8% of the Mac's memory (at least 1 GiB, at most 4 GiB), or macOS says warning;
- hard: under 3% (at least 512 MiB, at most 1.5 GiB), or macOS says critical;
- normal again once macOS says normal and the memory available is back above 1.5 times soft's line, so that a
  server giving back memory near the line doesn't take it again a second later.

These are conservative policy thresholds, not measured guarantees against swapping or process termination. At 8%,
the soft line is 1.28 GiB on a 16 GiB Mac and 3.84 GiB on a 48 GiB one; the floor reserves some room on smaller
machines and the ceiling avoids excessive eviction on large ones. The hard line is 0.5 GiB on 16 GiB and 1.44 GiB
on 48 GiB. File-backed pages are not all immediately reclaimable, and another process can allocate between readings.
What a server does at each level is serve.py's (`Batcher`).
"""

from __future__ import annotations

import ctypes
import enum
import os
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from typing import Any

SOFT_SHARE, SOFT_FLOOR, SOFT_CEILING = 0.08, 1 << 30, 4 << 30  # soft below this share of memory, within these bytes
HARD_SHARE, HARD_FLOOR, HARD_CEILING = 0.03, 512 << 20, 1536 << 20  # hard below this share, within these bytes
CLEAR = 1.5  # normal again once available is this many times the soft line (and macOS says normal)
BUSY_S = 1.0  # seconds between readings while requests come or memory is short
IDLE_S = 10.0  # seconds between readings otherwise; also how long after a request the busy cadence lasts
WARNING, CRITICAL = 2, 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical
KERNEL = {1: "normal", 2: "warning", 4: "critical"}


class Level(enum.IntEnum):
    NORMAL = 0
    SOFT = 1
    HARD = 2

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Reading:
    """One look at memory, in bytes: this process's physical `footprint`; the Mac's `memory`; what macOS can hand
    estimate is available (`available`: free pages, file cache and purgeable pages); what the
    compressor occupies (`compressed`); and the kernel's pressure level (`kernel`: 1 normal, 2 warning, 4 critical)."""

    footprint: int
    memory: int
    available: int
    compressed: int
    kernel: int


@dataclass(frozen=True)
class Thresholds:
    """The bytes available below which memory is short (`soft`) or very short (`hard`), and above which it is plenty
    again (`clear`), for a Mac with `memory` bytes."""

    soft: int
    hard: int
    clear: int

    @classmethod
    def of(cls, memory: int) -> Thresholds:
        soft = min(max(int(memory * SOFT_SHARE), SOFT_FLOOR), SOFT_CEILING)
        hard = min(max(int(memory * HARD_SHARE), HARD_FLOOR), HARD_CEILING)
        return cls(soft, hard, int(soft * CLEAR))


class _Rusage(ctypes.Structure):  # rusage_info_v0, sys/resource.h
    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        *((name, ctypes.c_uint64) for name in ("user", "system", "idle_wakeups", "wakeups", "pageins", "wired")),
        ("resident", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
        ("start", ctypes.c_uint64),
        ("exit", ctypes.c_uint64),
    ]


class _VMStatistics(ctypes.Structure):  # vm_statistics64, mach/vm_statistics.h (its first revision)
    _fields_ = [
        *((name, ctypes.c_uint32) for name in ("free", "active", "inactive", "wired")),
        *((name, ctypes.c_uint64) for name in ("zero_fill", "reactivations", "pageins", "pageouts", "faults")),
        *((name, ctypes.c_uint64) for name in ("cow_faults", "lookups", "hits", "purges")),
        ("purgeable", ctypes.c_uint32),
        ("speculative", ctypes.c_uint32),
        *((name, ctypes.c_uint64) for name in ("decompressions", "compressions", "swapins", "swapouts")),
        *((name, ctypes.c_uint32) for name in ("compressor", "throttled", "external", "internal")),
        ("uncompressed_in_compressor", ctypes.c_uint64),
    ]


HOST_VM_INFO64 = 4
RUSAGE_INFO_V0 = 0


class _Mac:
    """The system calls a reading makes, bound once (macOS's libSystem). OSError where they can't be: no libSystem,
    or one without these calls (ctypes says AttributeError, which here means only that)."""

    def __init__(self) -> None:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        try:
            self.rusage, self.statistics, self.sysctl = lib.proc_pid_rusage, lib.host_statistics64, lib.sysctlbyname
            host = lib.mach_host_self
        except AttributeError as error:
            raise OSError(f"libSystem: {error}") from None
        self.rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Rusage)]
        self.statistics.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(_VMStatistics),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.sysctl.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        host.restype = ctypes.c_uint32
        self.host = host()  # once: each call adds a reference to the port
        self.page = self.number("hw.pagesize", ctypes.c_uint64)
        self.memory = self.number("hw.memsize", ctypes.c_uint64)

    def number(self, name: str, kind: type[ctypes.c_uint64] | type[ctypes.c_int]) -> int:
        value, size = kind(), ctypes.c_size_t(ctypes.sizeof(kind))
        if self.sysctl(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), f"sysctl {name}")
        return int(value.value)

    def footprint(self) -> int:
        usage = _Rusage()
        if self.rusage(os.getpid(), RUSAGE_INFO_V0, ctypes.byref(usage)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage")
        return int(usage.phys_footprint)

    def read(self) -> Reading:
        stats, count = _VMStatistics(), ctypes.c_uint32(ctypes.sizeof(_VMStatistics) // 4)
        if self.statistics(self.host, HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)) != 0:
            raise OSError("host_statistics64 failed")
        # free counts the speculative pages, which the file cache (external) may count again: once, here
        available = max(0, stats.free - stats.speculative + stats.external + stats.purgeable) * self.page
        kernel = self.number("kern.memorystatus_vm_pressure_level", ctypes.c_int)
        return Reading(self.footprint(), self.memory, available, stats.compressor * self.page, kernel)


@cache
def _mac() -> _Mac:
    return _Mac()


def read() -> Reading:
    """This Mac's memory now (OSError where it can't be read)."""
    return _mac().read()


def now() -> Reading | None:
    """This Mac's memory now, or None where it can't be read: what a load or the wired limit goes by, which then
    holds to the fixed bounds alone."""
    try:
        return read()
    except OSError:
        return None


def footprint() -> int:
    """This process's physical footprint now, in bytes; 0 where it can't be read."""
    try:
        return _mac().footprint()
    except OSError:
        return 0


class Monitor:
    """This Mac's memory, read on a thread of its own every `busy` seconds while requests come or memory is short,
    every `idle` seconds otherwise: `level` is what the server acts on, `readings` counts the readings taken so the
    model thread acts on each new one once. `read` is how a reading is taken (`read`; tests give their own)."""

    def __init__(self, read: Callable[[], Reading] = read, busy: float = BUSY_S, idle: float = IDLE_S):
        if busy <= 0 or idle <= 0:
            raise ValueError("busy and idle are numbers of seconds above 0")
        self._read, self.busy, self.idle = read, busy, idle
        self.reading: Reading | None = None
        self.thresholds: Thresholds | None = None
        self.level = Level.NORMAL
        self.readings = 0
        self.error: str | None = None
        self._sampled: float | None = None
        self._requests = 0  # accepted requests, including one whose pass lasts longer than the idle window
        self._active = -float("inf")  # when a request last came or was answered (time.monotonic)
        self._resting = False  # reading at the idle cadence: a request wakes the thread
        self._lock, self._wake, self._stop = threading.Lock(), threading.Event(), False
        self._thread: threading.Thread | None = None

    def sample(self) -> Level:
        """Take a reading now and classify it: the level, set before the reading counts."""
        try:
            reading = self._read()
        except OSError as error:
            with self._lock:
                self.error = str(error)
            raise
        with self._lock:
            if self.thresholds is None:
                self.thresholds = Thresholds.of(reading.memory)
            self.level = self.classify(reading, self.thresholds)
            self.reading = reading
            self.readings += 1
            self.error, self._sampled = None, time.monotonic()
            return self.level

    def latest(self) -> tuple[int, Level]:
        """How many readings were taken, and the latest one's level, together."""
        with self._lock:
            return self.readings, self.level

    def status(self) -> dict[str, Any]:
        """A consistent reading for HTTP status, including a failure or the age of the last successful reading."""
        with self._lock:
            result: dict[str, Any] = {
                "pressure": str(self.level) if self.reading is not None else "unknown",
                "monitor_error": self.error,
                "sample_age_s": None if self._sampled is None else round(time.monotonic() - self._sampled, 3),
            }
            if self.reading is not None and self.thresholds is not None:
                result |= {
                    "macos": KERNEL.get(self.reading.kernel, str(self.reading.kernel)),
                    "available": self.reading.available,
                    "total": self.reading.memory,
                    "compressed": self.reading.compressed,
                    "soft_below": self.thresholds.soft,
                    "hard_below": self.thresholds.hard,
                    "clear_above": self.thresholds.clear,
                }
            return result

    def classify(self, reading: Reading, t: Thresholds) -> Level:
        """The level for `reading`: hard, soft, or, once short, soft until there is plenty again."""
        if reading.kernel >= CRITICAL or reading.available < t.hard:
            return Level.HARD
        if reading.kernel >= WARNING or reading.available < t.soft:
            return Level.SOFT
        if self.level is not Level.NORMAL and reading.available < t.clear:
            return Level.SOFT
        return Level.NORMAL

    def start(self) -> bool:
        """Take the first reading and go on reading on a thread of its own, retrying unavailable readings."""
        if self._thread is not None:
            return self._thread.is_alive()
        with suppress(OSError):
            self.sample()
        self._thread = threading.Thread(target=self._run, name="ippon-memory", daemon=True)
        self._thread.start()
        return True

    def active(self) -> None:
        """A request came or was answered: read at the busy cadence for `idle` seconds, at once if resting."""
        self._active = time.monotonic()
        if self._resting:
            self._resting = False
            self._wake.set()

    def begin(self) -> None:
        """An accepted request keeps readings at the busy cadence until it finishes or is cancelled."""
        with self._lock:
            self._requests += 1
            self.active()

    def end(self) -> None:
        """An accepted request finished or was cancelled."""
        with self._lock:
            self._requests -= 1
            self.active()

    def close(self) -> None:
        """Stop reading (again: nothing)."""
        self._stop = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop:
            with self._lock:
                busy = (
                    self._requests > 0 or self.level is not Level.NORMAL or time.monotonic() - self._active < self.idle
                )
                self._resting = not busy
            self._wake.wait(self.busy if busy else self.idle)
            self._wake.clear()
            if self._stop:
                return
            try:
                self.sample()
            except OSError:
                continue
