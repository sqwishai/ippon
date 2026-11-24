"""`ippon recommend [TASK]`: which builds this Mac can run for a kind of input, and the one to start with, before
anything is downloaded.

It reads verified.py's builds only (names.py's families), so it says what the models table says: for each family, the
build for the task (the one `ippon pull FAMILY` takes; for images or audio, the family's recommended build among
those that read them), its weights, what has been checked of it, and whether it fits this Mac as published. The order
is fixed: verified builds first, then those whose prompts are checked, then the rest; within each, those that fit
before those that don't, then the smallest by published weight size. Nothing here says which model answers a given task
best: `ippon eval` measures that on the caller's own labelled examples.

A build that doesn't fit is listed with the family's smaller builds that do, never swapped for one: a smaller build
or precision changes the model's answers, and choosing one is the caller's call.

`ippon setup` and `ippon pull` ask the same questions here: whether a build fits (`fit`), what it holds in memory at a
precision (`gigabytes`) and which build to start with (`starter`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from . import names, terminal
from .admission import budget, quantized_bytes
from .device import Device
from .quantization import FULL, GROUP_SIZE, PRECISIONS, Precision
from .schema import MediaField

TASKS: dict[str, Literal["text"] | MediaField] = {  # what a task's requests carry: text, or a media field with it
    "text": "text",
    "routing": "text",
    "classification": "text",
    "moderation": "text",
    "extraction": "text",
    "images": "images",
    "audio": "audio",
    "videos": "videos",
}
SAYS = {
    "text": "text (routing, classification, moderation, extraction checks)",
    "images": "images",
    "audio": "audio",
    "videos": "videos",
}
RANK = {"verified": 0, "prompts": 1, "experimental": 2, "past bound": 3}
CHECKED = ("verified", "prompts")  # what a build to start with must be
Fit = Literal["fits", "tight", "too big", "unknown"]


def fit(gigabytes: float, device: Device) -> Fit:
    """Whether weights holding `gigabytes` in memory fit what this Mac gives a model of its GPU's working set
    (admission's budget): with room to spare (in half of it), only just (other apps' memory may slow it), or not at
    all; unknown where the working set is."""
    if not device.working_set_bytes:
        return "unknown"
    room = budget(device.working_set_bytes) / 1e9
    return "fits" if gigabytes <= room / 2 else "tight" if gigabytes <= room else "too big"


def may_fit(gigabytes: float, device: Device) -> bool:
    """Whether weights holding `gigabytes` may fit this Mac (`fit`): with room, only just, or where it is unknown."""
    return fit(gigabytes, device) != "too big"


def gigabytes(named: names.Named, at: Precision) -> float:
    """About what `named`'s weights hold in memory loaded at `at`: as published (`Named.resident`), or quantized here
    as admission counts it (`admission.quantized_bytes`), from its full-precision checkpoint's values."""
    if at.bits is None:
        return named.resident
    values = round(named.resident * 1e9 / PRECISIONS[named.tag].width)
    return quantized_bytes(values, at.bits, at.width, GROUP_SIZE) / 1e9


T = TypeVar("T")


def starter(items: Sequence[T], named: Callable[[T], names.Named], usable: Callable[[T], bool]) -> T | None:
    """The one of `items` to start with: the first starter among the `usable` (verified.py's `starter`: Kettei Core,
    else Kettei Dot), else the first usable; None when none is."""
    able = [item for item in items if usable(item)]
    starters = [item for item in able if named(item).entry.starter is not None]
    starters.sort(key=lambda item: named(item).entry.starter or 0)
    return (starters or able)[0] if able else None


@dataclass(frozen=True)
class Pick:
    """A family's build for the task: what has been checked of it for the task (the build's status, or its media's),
    whether it fits this Mac (None: unknown), the family's smaller builds for the task that fit when it doesn't, and
    a verified full-precision build that fits and would read faster here (`faster_here`)."""

    named: names.Named
    status: str
    fits: bool | None
    smaller: tuple[names.Named, ...] = ()
    faster: names.Named | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.named.name,
            "model": self.named.model,
            "load": self.named.build.load,
            "precision": self.named.tag,
            "weights": str(self.named.build.weights),
            "gigabytes": self.named.build.gigabytes,
            "status": self.status,
            "fits": self.fits,
            "licence": self.named.licence,
            "media": sorted(self.named.media),
            "smaller": [n.name for n in self.smaller],
            "faster_here": self.faster.name if self.faster else None,
        }


def recommend(task: str, device: Device) -> list[Pick]:
    """Each family's build for `task` (a TASKS key) on `device`, in the fixed order (the module's docstring)."""
    kind = TASKS[task]

    def fits(n: names.Named) -> bool | None:
        return None if fit(n.resident, device) == "unknown" else may_fit(n.resident, device)

    picks: list[Pick] = []
    for builds in names.families().values():
        able = [n for n in builds if not n.low_bit and (kind == "text" or kind in n.media)]  # never below 8 bits
        chosen = names.recommended(able) if able else None
        if chosen is None:
            continue
        n = chosen[0]
        status = n.build.status if kind == "text" else n.media[kind]
        fitting = fits(n)
        smaller = tuple(sorted((m for m in able if m.size < n.size and fits(m)), key=lambda m: m.size))
        faster = faster_here(n, able, device)
        picks.append(Pick(n, status, fitting, smaller if fitting is False else (), faster))
    order = {id(p): i for i, p in enumerate(picks)}
    return sorted(
        picks,
        key=lambda p: (RANK.get(p.status, len(RANK)), {True: 0, None: 1, False: 2}[p.fits], p.named.size, order[id(p)]),
    )


def start_with(picks: list[Pick]) -> names.Named | None:
    """The build to pull first (`starter`): among the picks whose prompts are checked and that fit this Mac. Its
    faster full-precision build on this Mac if it has one, else itself. When this Mac didn't say what it gives a
    model, nothing is known to fit, and the smallest of the starters is the one every supported Mac holds."""
    if all(p.fits is None for p in picks):
        starters = [p for p in picks if p.status in CHECKED and p.named.entry.starter is not None]
        return min(starters, key=lambda p: p.named.size).named if starters else None
    first = starter(picks, lambda p: p.named, lambda p: p.fits is True and p.status in CHECKED)
    return None if first is None else first.faster or first.named


def as_json(task: str, picks: list[Pick], device: Device) -> dict[str, Any]:
    """`recommend` as data, for `ippon recommend --json` and the MCP tool alike: the task, this Mac, every pick, and
    the build to pull first (`start_with`), the one an agent should start from rather than the first pick."""
    first = start_with(picks)
    mac = {"chip": device.name, "memory_bytes": device.memory_bytes, "working_set_bytes": device.working_set_bytes}
    return {
        "task": task,
        "mac": mac,
        "picks": [p.to_json() for p in picks],
        "start_with": first.name if first else None,
    }


# Without neural accelerators (M1 to M4) ippon reads an 8-bit build's prompts through MLX's quantized matmul, slower
# than the same model at full precision: d1-Omni-2B's Q8_0 file against its bf16 build took 11% to 34% longer on an
# M1, M2 Pro and M4 Pro (619 against 460 ms at 1,000 words on the M4 Pro). An M5's 8-bit kernels time both the same.
# Only with room to spare: on an 8 GB M1 the bf16 build held 5.2 GB of a 5 GB working set and the Mac swapped, and
# the 8-bit file was the faster one; a full-precision build is offered where it fits with room to spare (`fit`).
def faster_here(n: names.Named, builds: list[names.Named], device: Device) -> names.Named | None:
    """On a Mac without neural accelerators, the smallest verified full-precision build of the family that fits with
    room to spare, when `n` is an 8-bit one (names.OWN_EIGHT_BIT): faster here, and nearer its creator's answers.
    None otherwise."""
    if device.neural_accelerators or n.tag not in names.OWN_EIGHT_BIT:
        return None
    full = [m for m in builds if m.tag in FULL and m.build.status == "verified" and fit(m.resident, device) == "fits"]
    return min(full, key=lambda m: m.size) if full else None


def report(task: str, picks: list[Pick], device: Device, colour: bool = False, width: int | None = None) -> str:
    """`recommend`'s table, the first build to pull, and what the order does and doesn't say. At a terminal
    (`colour`, and its `width` in columns) the build to start with is in the accent of what is settled, the commands
    to run in the accent of what to do next, and a row longer than the terminal is cut instead of wrapping."""
    room = budget(device.working_set_bytes) if device.working_set_bytes else None
    mac = f"{device.name or 'this Mac'} ({(device.memory_bytes or 0) / 2**30:.0f} GB)"
    keeps = f"; it keeps {room / 1e9:.1f} GB for a model" if room else ""
    lines = [f"For {SAYS[TASKS[task]]} on {mac}{keeps}:", ""]
    rows = [("", "build", "weights", "checked", "fits", "licence")]  # a build's name ends with its precision
    for i, p in enumerate(picks, 1):
        fit = {True: "yes", None: "unknown", False: "no"}[p.fits]
        status = names.STATUS.get(p.status, p.status)
        rows.append((str(i), p.named.name, str(p.named.build.weights), status, fit, p.named.licence))
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    first = start_with(picks)
    for r in rows:
        number, build, *rest = (cell.ljust(w) for cell, w in zip(r, widths, strict=True))
        tone = ("2",) if r is rows[0] else ("mint",) if first is not None and r[1] == first.name else ()
        dim = ("2",) if r is rows[0] else ()
        row = [(f"  {number}  ", dim), (f"{build}  ", tone), ("  ".join(rest).rstrip(), dim)]
        lines.append(terminal.fitted(row, colour, width))
    for p in picks:
        if p.fits is False:
            others = ", ".join(f"{m.name} ({m.build.weights}, {m.status})" for m in p.smaller)
            lines.append(
                f"  {p.named.name} doesn't fit as published"
                + (f"; smaller builds that do, if their answers are acceptable to you: {others}" if others else "")
            )
    for p in picks:
        if p.faster is not None:
            lines.append(
                f"  on this Mac, {p.faster.name} ({p.faster.size:g} GB) fits and reads prompts faster than "
                f"{p.named.name}: 8-bit builds take 11% to 34% longer on M1 to M4"
            )
    lines.append("")
    if first is not None:
        lines.append(
            f"Start with: {terminal.styled(f'ippon pull {first.name}', colour, '1', 'amber')}   "
            f"({first.build.weights}, it answers a sample request when the download ends)"
        )
    lines.append(
        terminal.commands(
            "The order is by what has been checked of each build, then fit, then published weight size; "
            "it doesn't say which answers your task best: `ippon eval DATA.jsonl --model A --model B` measures that "
            "on your own labelled examples.",
            colour,
        )
    )
    return "\n".join(lines)
