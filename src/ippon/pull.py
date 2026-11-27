"""One pull plan for CLI and other local clients: inspect, admit, accept, then download with progress events."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO

from . import check, downloads, hub, licences, locate, names, quantization, recommend
from .device import detect
from .errors import InsufficientMemory, UnsupportedModel
from .hub import Download


@dataclass(frozen=True)
class Plan:
    asked: str
    model: str
    report: dict[str, Any]
    licenses: tuple[licences.Licence, ...]
    files: tuple[downloads.File, ...]
    previous_revision: str | None
    groups: tuple[Download, ...]  # what is downloaded, by repository: a GGUF build's file first
    local: str | None  # the model's own folder or GGUF file, when it is on this Mac already

    def run(
        self,
        *,
        accept_license: bool = False,
        approve: Callable[[licences.Licence], bool] | None = None,
        progress: Callable[[downloads.Progress], None] | None = None,
    ) -> tuple[str | None, str | None]:
        downloads.admit_disk(self.files)
        for licence in self.licenses:
            licence.require(self.asked, accept_license, approve)
        if not self.files:
            return self.local, None
        complete: downloads.Progress | None = None

        def forward(event: downloads.Progress) -> None:
            nonlocal complete
            if event.phase == "complete":
                complete = event
            elif progress:
                progress(event)

        before = {names.split(group.model)[0]: hub.blobs(names.split(group.model)[0]) for group in self.groups}
        downloads.transfer(self.files, forward)
        # Recorded only once every file is complete, so loads reuse the exact reviewed revision offline.
        for group in self.groups:
            repo, revision = names.split(group.model)
            hub.complete(group, hub.snapshot_dir(repo, str(revision)), before[repo])
        if progress and complete:
            progress(complete)
        first = self.groups[0]
        repo, revision = names.split(first.model)
        folder = hub.snapshot_dir(repo, str(revision))
        return str(folder / first.patterns[0] if first.gguf else folder), revision


def _asked(need: check.Need, carried: str | None) -> bool:
    """Whether `need` is a precision the model is pulled to run at: the one its name carries, or one of the same
    width at full precision, else any that keeps the weights as published."""
    if carried is None:
        return not need.changes_weights
    width = quantization.PRECISIONS[carried].width
    same = (
        not need.changes_weights and need.precision != "auto" and quantization.PRECISIONS[need.precision].width == width
    )
    return need.precision == carried or same


def prepare(model: str) -> Plan:
    """Resolve a name and inspect compatibility, metadata and exact missing bytes before fetching model assets."""
    resolved = str(names.resolve(model))
    try:
        inspected = locate.inspect(resolved, metadata_only=True)
        device = detect()
        found = check.facts(resolved, inspected, device)
    except UnsupportedModel as error:
        raise UnsupportedModel(f"{error}. No weights downloaded.") from error
    named = names.of_load(resolved)
    # a name that carries a precision (kettei-core:q8, jeff:fp32) has its fit checked at that precision, where fp16
    # takes bf16's row (one width), and any other name at the precisions it computes in as published
    weights, carried = names.weights_at(model), names.precision_of(model)
    at = f"at {carried}" if carried else "as published"
    room = check.room(device)
    every = check.needs(inspected, room)
    rows = [n for n in every if _asked(n, carried)]
    if not any(n.fits for n in rows):
        needed = min((n.bytes for n in rows), default=0)
        others = names.families()[named.family] if named else ()
        smallest = names.smallest(others, lambda n: recommend.may_fit(n.resident, device))
        # a family that stores only full precision (Kettei Core) still fits at 8 bits, which ippon computes as it loads
        eight = named is not None and weights == "full" and any(n.precision == "q8" and n.fits for n in every)
        instead = (
            f" The smallest build of {smallest.family} that fits is {smallest.name} "
            f"({smallest.build.weights}, {smallest.status}): ippon pull {smallest.name}."
            if smallest
            else f" At 8 bits it fits: ippon pull {named.family}:q8."
            if eight and named
            else ""
        )
        if not instead and (first := recommend.start_with(recommend.recommend("routing", device))) is not None:
            # nothing of this family fits (Kettei Core on an 8 GB Mac): the model to start with here, and the list
            instead = f" ippon pull {first.name} fits this Mac, and `ippon recommend` lists what else does."
        raise InsufficientMemory(
            f"{model} doesn't fit this Mac {at} (needs {needed:,} bytes; "
            f"model budget {room} bytes). No weights downloaded.{instead}"
        )
    listed = names.catalogued(resolved)
    checked = listed[1] if listed else None
    at = (names.parse(resolved).revision or "").lower()  # any case, as the Hub reads commits
    pinned = (names.parse(checked.load).revision or "").lower() if checked else ""
    if checked and pinned != at:  # the build ippon checked, at another revision than this one
        support = (
            f"checked against its creator's code at revision {pinned[:12]} ({checked.status}), "
            f"not at {at[:12] or 'the newest'}: pull {checked.load} for the checked one"
        )
    elif checked and checked.status == "verified":
        support = "supported: checked against its creator's code on Macs"
    elif checked:
        support = f"supported with model-specific speed work; verification: {checked.status}"
        if checked.note:
            support += f" ({checked.note})"
    elif inspected.adapter.generic:
        support = "runs as published through ippon's generic path, without model-specific speed work or a creator check"
    else:
        support = (
            f"runs through the installed {inspected.adapter.name} adapter; no creator check recorded for this build"
        )
    terms: dict[str, licences.Licence] = {}
    files: dict[tuple[str, str, str], downloads.File] = {}
    groups = inspected.downloads
    for group in groups:
        license_, items = downloads.files(group.model, group.patterns, gguf=group.gguf)
        terms[license_.key] = license_
        files.update({(f.repo, f.revision, f.name): f for f in items})
    pending = tuple(files.values())
    needed, free = downloads.admit_disk(pending)
    found |= {
        "support": support,
        "creator": inspected.creator,
        "licenses": [{"name": lic.name, "url": lic.url, "accepted": lic.accepted} for lic in terms.values()],
        "disk": {"total_bytes": sum(f.size for f in pending), "needed_bytes": needed, "free_bytes": free},
        "name": named.name if named else model,
    }
    previous = hub.known(resolved) if groups and not groups[0].gguf else None
    return Plan(model, resolved, found, tuple(terms.values()), pending, previous, groups, inspected.local)


class TerminalProgress:
    """Render library events on one TTY line, or periodically on separate lines in a redirected command."""

    def __init__(self, stream: TextIO, *, tty: bool):
        self.stream, self.tty = stream, tty
        self.last = 0.0
        self.width = 0

    def __call__(self, event: downloads.Progress) -> None:
        now = time.monotonic()
        final = event.phase in {"complete", "cancelled", "failed"}
        if not final and now - self.last < (0.2 if self.tty else 3.0):
            return
        eta = f"{event.seconds_left:.0f}s left" if event.seconds_left is not None else "estimating time left"
        line = f"{event.done:,} / {event.total:,} bytes · {event.bytes_per_second / 1e6:.1f} MB/s · {eta}"
        if self.tty:
            self.stream.write("\r" + line.ljust(self.width) + ("\n" if final else ""))
            self.width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last = now
