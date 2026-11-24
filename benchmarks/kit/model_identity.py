"""Content fingerprints for the model builds a campaign actually opens, without loading a runtime."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import common

SUFFIXES = {".safetensors", ".gguf", ".json", ".txt", ".model", ".bin", ".mlmodel", ".py", ".pt", ".npz"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(table: dict, engines: list[str], models: Path) -> dict:
    records, files = {}, {}

    def fingerprint(path: Path) -> str:
        path = path.resolve()
        key = str(path)
        if key not in files:
            files[key] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        return key

    for engine in engines:
        cmd = table[engine]["cmd"]
        options = dict(pairwise(cmd))
        name = options.get("--gguf", options.get("--model"))
        if not name:
            raise ValueError(f"{engine}: no model identity")
        tag = name.removeprefix("ollama://")
        store = models / "ollama"
        if name.startswith("ollama://") or "--models-dir" in options:
            family, version = tag.split(":", 1)
            path = store / "manifests/registry.ollama.ai/library" / family / version
            manifest = json.loads(path.read_text())
            entries = [fingerprint(path)]
            for layer in [manifest["config"], *manifest["layers"]]:
                blob = store / "blobs" / layer["digest"].replace(":", "-")
                key = fingerprint(blob)
                if files[key]["sha256"] != layer["digest"].removeprefix("sha256:"):
                    raise ValueError(f"{engine}: content does not match its registry digest")
                entries.append(key)
            records[engine] = {"model": name, "files": entries}
            continue
        path = (Path(options["--model-dir"]) / name) if "--model-dir" in options else Path(name)
        path = path.expanduser().resolve()
        if path.is_file():
            entries = [fingerprint(path)]
        else:
            entries = [
                fingerprint(file)
                for file in sorted(path.rglob("*"))
                if file.is_file()
                and file.suffix in SUFFIXES
                and not any(part in {".git", ".cache", "__pycache__"} for part in file.relative_to(path).parts)
            ]
        if not entries:
            raise ValueError(f"{engine}: no model files found")
        records[engine] = {"model": str(path), "hub_revision": common.repo_revision(str(path)), "files": entries}
    return {"engines": records, "files": files}
