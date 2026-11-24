"""`ippon check MODEL`: what a model is before its weights are downloaded. The adapter that runs it, the files
a load downloads and their size, the memory each precision needs and whether it fits this Mac, and what has been
verified of it (`verified.py`). Only the model's small files (configs, tokenizer) are downloaded: the weights'
tensor shapes come from their safetensors headers, which the Hub serves without the weights (a folder's are read
from its files), and a GGUF build's size from the Hub's listing.

A model downloaded already is checked from its files, without the network: its small files and weights at the
commit recorded when it was downloaded (`hub.downloaded_files`, downloads.json), and a GGUF build's file from the Hub
cache. `ippon pull`'s check asks the Hub for the newest commit's small files, and reads that commit's weights
from the cache when they are the ones downloaded. locate.py finds what is reported here (`locate.inspect`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from . import hub, locate, names, quantization
from .admission import budget, resident_bytes
from .device import Device, detect
from .locate import Inspected
from .quantization import FULL, PRECISIONS
from .verified import status_line

# the precisions a check lists, in this order: auto, a full precision per width (fp16 holds what bf16 does), then those
# quantizing here
SHOWN = (
    "auto",
    *(
        name
        for i, name in enumerate(FULL)
        if PRECISIONS[name].width not in {PRECISIONS[other].width for other in FULL[:i]}
    ),
    *quantization.QUANTIZED,
)


def _published(model: str) -> str:
    """The Hub id a model is published as: as given, or a Hub cache folder's (models--org--name/snapshots/...); a
    folder of any other kind as given (it matches no recorded build)."""
    found = hub.parse_snapshot(model)
    return found[0] if found else str(model)


def check(
    model: str, adapter: str | None = None, device: Device | None = None, *, metadata_only: bool = False
) -> dict[str, Any]:
    """What `check` reports of `model` (a Hub id, a GGUF build, or a folder), on `device` (this Mac's)."""
    return facts(str(model), locate.inspect(model, adapter, metadata_only=metadata_only), device)


@dataclass(frozen=True)
class Need:
    """The memory a model's weights hold loaded at one precision, and whether that fits this Mac's room for a model
    (`room`; it fits where the room is unknown): what `check` reports and `pull` refuses a download by."""

    precision: str
    bytes: int
    fits: bool
    changes_weights: bool
    low_bit: bool


def room(device: Device) -> int | None:
    """The bytes this Mac keeps for a model (admission's budget of its GPU's working set); None where it is unknown."""
    return budget(device.working_set_bytes) if device.working_set_bytes else None


def needs(found: Inspected, room: int | None) -> list[Need]:
    """What `found` (`locate.inspect`'s) needs at each precision a check lists (SHOWN), against `room`: a quantized
    build only as stored, never quantized again; none for a model whose weights' shapes are unknown."""
    out = []
    for name in SHOWN:
        meant = quantization.PRECISIONS[name]
        if found.as_stored and meant.changes_weights:
            continue
        if found.resident is not None:  # a GGUF file is resident as it is stored
            needed = found.resident
        elif found.stored:
            needed = resident_bytes(found.stored, meant.width, meant.bits)
        else:
            continue
        fits = room is None or needed <= room
        out.append(Need(name, needed, fits, meant.changes_weights, meant.low_bit))
    return out


def facts(model: str, found: Inspected, device: Device | None = None) -> dict[str, Any]:
    """What `check` reports of `model`, as `locate.inspect` found it, on `device` (this Mac's)."""
    device = device or detect()
    directory, stored = found.directory, found.stored
    formats = directory.source.formats(directory, stored)
    kept = room(device)
    base = found.base
    builds = names.for_model(_published(model)) or (names.for_model(base[0]) if base else ())
    files, size = found.download
    return {
        "model": model,
        "downloads": [d.to_json() for d in found.downloads],
        "generic": found.adapter.generic,
        "revision": found.revision,
        "adapter": found.adapter.name,
        "architecture": directory.config.get("model_type"),
        "download": {"files": files, "bytes": size},
        "working_set_bytes": device.working_set_bytes,
        "room_bytes": kept,
        "memory": [asdict(need) for need in needs(found, kept)],
        "formats": formats,
        "verified": [{"load": b.load, "status": b.status, "note": b.note} for b in builds],
    }


def report(found: dict[str, Any]) -> str:
    gb = 1e9
    lines = [f"{found['model']}{' @ ' + found['revision'][:12] if found['revision'] else ''}"]
    lines.append(f"  adapter {found['adapter']} ({found['architecture'] or 'no model_type'})")
    if found.get("formats"):
        lines.append(
            "  stored formats: " + ", ".join(f"{mode}: {count} matrices" for mode, count in found["formats"].items())
        )
    d = found["download"]
    lines.append(
        f"  download: {d['files']} weight file(s), {d['bytes'] / gb:.1f} GB" if d["files"] else "  download: none"
    )
    if found["room_bytes"]:
        lines.append(f"  this Mac keeps {found['room_bytes'] / gb:.1f} GB for a model")
    for m in found["memory"]:
        fits = "fits" if m["fits"] else "does not fit"
        note = (
            ", below 8 bits, so its answers drift from the creator's"
            if m.get("low_bit")
            else ", changes the weights (check it with ippon verify)"
            if m["changes_weights"]
            else ""
        )
        lines.append(f"  {m['precision']:5s} {m['bytes'] / gb:6.1f} GB  {fits}{note}")
    if found["verified"]:
        lines.append("  verified:")
        lines += [f"    {v['load']}: {status_line(v['status'], v['note'])}" for v in found["verified"]]
    else:
        lines.append(
            "  verified: nothing recorded for it (a folder is matched only in the Hub's cache: check the build by its"
            " Hub id); ippon verify MODEL --reference R checks it"
        )
    return "\n".join(lines)
