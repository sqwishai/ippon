"""Which artifact a model was loaded from, so that every answer can say what produced it (`Result.diagnostics`'s
`build`, and each loaded model's `details` in `serve`'s `/v1/models`): the build's short name when one names it
(names.py), the Hub repository and the commit actually read, a GGUF file and its sha256, or an Ollama build's
checkpoint identity (ollama.checkpoint_identity: its tensor blobs and the assets its adapter reads), a LoRA's
base model at its commit, and the licence of each (licences.py), which ippon shows wherever a model runs.

    {"name": "d1-2b:bf16", "repo": "LiquidAI/d1-2B", "revision": "0f3c...", "base": None,
     "licence": {"name": "LFM Open License v1.0", "url": "https://huggingface.co/LiquidAI/d1-2B#license"}}
    {"name": "d1-2b:q8_0", "repo": "LiquidAI/d1-2B-GGUF", "revision": "e968...",
     "file": "d1-2B-Q8_0.gguf", "sha256": "cd94...", "base": None}
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import hub, names
from .licences import of_folder

if TYPE_CHECKING:
    from .model_dir import ModelDir


def describe(model: str | Path, directory: ModelDir, base: ModelDir | None = None) -> dict[str, Any]:
    """Where the load of `model` (as it was named: a short name, a Hub id, a GGUF build or file, an ollama:// build or
    a folder) read its files from, `directory` being what it opened and `base` a LoRA's base model."""
    asked = str(model)
    out: dict[str, Any] = {"name": None}
    origin = directory.source.origin(directory)
    if origin.path is not None:
        found = hub.parse_snapshot(origin.path)
        out |= {"repo": found[0], "revision": found[1]} if found else {"path": str(origin.path)}
    out |= origin.identity
    named = names.find(asked)[0] if names.is_name(asked) else names.of_load(asked)
    if named is None and out.get("repo") and "file" not in out:  # a Hub cache folder given by its path (not one file)
        named = names.of_load(out["repo"])
    pinned = names.parse(named.build.load).revision if named is not None and named.kind == "safetensors" else None
    if pinned and out.get("revision") and str(out["revision"]).lower() != pinned.lower():
        # another commit of a named build: its own weights, which no check covers, so never the build's name and status
        out["unchecked"] = f"{named.name if named else ''} is checked at revision {pinned[:12]}, not this one"
        out["name"] = None
    else:
        out["name"] = named.name if named else None
    known = named.licence if named else ""
    out["licence"] = of_folder(directory.path, out.get("repo"), out.get("revision"), known).to_json()
    of_base = hub.parse_snapshot(base.path) if base is not None else None
    out["base"] = {"repo": of_base[0], "revision": of_base[1]} if of_base else None
    if base is not None:
        repo, revision = of_base if of_base else (None, None)
        out["base"] = (out["base"] or {}) | {"licence": of_folder(base.path, repo, revision).to_json()}
    return out


def licence(details: Mapping[str, Any]) -> dict[str, Any] | None:
    """The licence of the build a loaded model's `details` describe (its `build`, `describe`'s), {"name", "url"};
    None before a load."""
    found = (details.get("build") or {}).get("licence")
    return dict(found) if found else None
