"""Sample competing CPU use for a whole run, excluding only the benchmark's own process tree.

Each reading keeps its time, so a timed request is judged by the readings that overlapped it. No GPU APIs run here.
Keep names in live output; the published serializer retains only CPU percentages. A failed ps command invalidates
the measurement rather than silently claiming the machine was idle.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path


def competing(snapshot: str, roots: set[int]) -> list[str]:
    rows = [line.split(maxsplit=3) for line in snapshot.splitlines()]
    parsed = [(int(pid), int(ppid), float(cpu), command) for pid, ppid, cpu, command in rows]
    owned = set(roots)
    while True:
        children = {pid for pid, ppid, _, _ in parsed if ppid in owned}
        if children <= owned:
            break
        owned |= children
    return sorted(
        f"{Path(command.split()[0]).name} ({cpu:.0f}%)"
        for pid, _, cpu, command in parsed
        if pid not in owned and cpu > 50
    )


class Monitor:
    def __init__(self, server_pid: int | None):
        self.roots = {os.getpid()}
        if server_pid:
            self.roots.add(server_pid)
        self.busy: set[str] = set()
        self.samples: list[tuple[float, list[str]]] = []  # (time.monotonic() as ps started, competing processes)
        self.sampled = threading.Condition()
        self.errors: list[str] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, name="benchmark-activity", daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            taken = time.monotonic()
            try:
                snapshot = subprocess.run(
                    ["ps", "-A", "-o", "pid=,ppid=,pcpu=,comm="],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout
                names = competing(snapshot, self.roots)
                self.busy.update(names)
                with self.sampled:
                    self.samples.append((taken, names))
                    self.sampled.notify_all()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.errors.append(type(exc).__name__)
            self.stop.wait(0.5)

    def during(self, start: float, end: float) -> list[str]:
        """Competing processes read from `start` to the first sample begun after `end` (time.monotonic()). ps
        reports a decaying CPU average, so that later sample still shows a burst that overlapped the request's end."""
        with self.sampled:
            if not self.sampled.wait_for(lambda: bool(self.samples) and self.samples[-1][0] > end, timeout=10):
                self.errors.append("no sample within 10 s")
                return []
            after = next(at for at, _ in self.samples if at > end)
            return sorted({name for at, names in self.samples if start <= at <= after for name in names})

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=6)
