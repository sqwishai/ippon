"""Read an already-pulled Ollama MLX manifest and its blobs without a server or cache writes.

The manifest pins content hashes. No model code runs, no weights are converted, and no download is attempted.
Tensor headers carry format metadata independently of config.json (mixed builds use both MXFP8 and NVFP4). Builds
below 8 bits (NVFP4, MXFP4, int4) load only when their tag is named, with a warning of their drift (drift.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import UnsupportedModel
from .hashes import content_sha256
from .model_dir import Kind, ModelDir, Origin
from .quantization import MODES, Quantization, biases_of, inspect, scales_of, unsupported_format
from .safetensors import Tensor, tensor_header
from .verified import BUILDS

if TYPE_CHECKING:
    from .weights import Weights

TENSOR = "application/vnd.ollama.image.tensor"
JSON = "application/vnd.ollama.image.json"
TEMPLATE = "application/vnd.ollama.image.template"  # an Ollama chat template, where the manifest stores one apart
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
# the Ollama models the catalogue has builds of (verified.py), whose other tags a refusal names
CATALOGUED = frozenset(
    b.load.removeprefix("ollama://").partition(":")[0] for b in BUILDS if b.load.startswith("ollama://")
)
# The JSON assets an adapter reads from an Ollama build (config, the tokenizer and its config, which carries the
# chat template, and the processor and decision configs). generation_config.json and vocab.json are stored by some
# builds and read by none of them. A licence, a params layer and the registry manifest Ollama keeps as provenance
# are not in this set either: none of them changes an answer.
READ_ASSETS = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "decision_config.json",
        "joint_head_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.txt",
    }
)


def counts_for_checkpoint(layer: Mapping[str, Any]) -> bool:
    """Whether a manifest layer can change an answer: a tensor blob, a chat template, or a JSON asset the adapters
    read. Licence, params and provenance layers do not."""
    media = layer.get("mediaType")
    return media in (TENSOR, TEMPLATE) or (media == JSON and layer.get("name") in READ_ASSETS)


def checkpoint_identity(layers: list[dict[str, Any]]) -> str:
    """The build's checkpoint identity: the SHA-256 of the sorted digests of the layers `counts_for_checkpoint`
    keeps, one per layer. `verify`, `check` and a decider's `details["checkpoint"]` all read this, and nothing else
    defines it. An Ollama re-pull that adds a provenance or licence layer leaves it unchanged; a changed tensor blob
    does not."""
    digests = sorted(str(layer["digest"]) for layer in layers if counts_for_checkpoint(layer))
    return hashlib.sha256("\n".join(digests).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            return value
    except (OSError, ValueError, RecursionError) as error:  # RecursionError: JSON nested past Python's limit
        raise UnsupportedModel(f"{path}: cannot read Ollama metadata: {error}") from error
    raise UnsupportedModel(f"{path}: expected an Ollama JSON object")


def tensor_metadata(
    h: dict[str, Any], alternative: str = "an MXFP8 or bf16 Ollama tag"
) -> tuple[dict[str, str], dict[str, Any]]:
    """Canonical MLX companion names and per-matrix quantization from Ollama's header."""
    metadata = h.get("__metadata__", {})
    renamed, quantization = {}, {}
    for name in h:
        if name == "__metadata__" or name + ".scale" not in h:
            continue
        base = name.removesuffix(".weight")
        kind = metadata.get(name + ".quant_type", metadata.get("quant_type", "")).lower()
        if kind in ("int2", "int3", "int5", "int6"):
            raise unsupported_format(f"{name}: Ollama {kind}", alternative)
        if kind in ("int4", "int8"):
            mode, bits = "affine", int(kind[-1])
        elif kind in MODES and kind != "affine":
            mode = kind
            bits = 8 if kind == "mxfp8" else 4
        else:
            raise UnsupportedModel(f"{name}: missing or unsupported Ollama quant_type {kind!r}")
        group = metadata.get(name + ".group_size", metadata.get("group_size", MODES[mode][1]))
        try:
            spec = {"mode": mode, "bits": bits, "group_size": int(group), "global_scale_convention": "multiplier"}
        except (TypeError, ValueError) as error:
            raise UnsupportedModel(f"{name}: invalid group size {group!r}") from error
        Quantization.read(spec, name, tuple(h[name]["shape"]), tuple(h[name + ".scale"]["shape"]))
        quantization[base] = spec
        renamed[name + ".scale"] = scales_of(name)
        if name + ".bias" in h:
            renamed[name + ".bias"] = biases_of(name)
    return renamed, quantization


@dataclass(frozen=True)
class Build:
    """A pulled Ollama MLX build, as a model's source (`model_dir.Source`): its manifest, checkpoint identity, the JSON
    assets the manifest stores by file name, its tensor blobs, their tensors, each packed matrix's quantization
    (`packing`) and the companion names normalized for the shared weight loader (`renames`).

    Ollama merges the creator's weight files into one set of tensor blobs and keeps no file names for them, so
    every request for weights reads that whole set: the tensors of `files` are among it (a format finds a file's
    tensors by their names, as Clef's head does), and those of `exclude` too."""

    spec: str  # the ollama:// name it was opened by
    manifest: Path
    checkpoint: str
    assets: dict[str, Path]
    blobs: tuple[Path, ...]
    tensors: tuple[Tensor, ...]
    packing: dict[str, Any]
    renames: dict[str, str]
    kind: Kind = field(default="ollama", init=False)
    as_stored: bool = field(default=True, init=False)

    def file(self, name: str) -> Path:
        if name not in self.assets:
            raise UnsupportedModel(f"{self.manifest}: no {name!r} asset in the MLX manifest")
        return self.assets[name]

    def has(self, name: str) -> bool:
        return name in self.assets

    def quantization(self, config: Mapping[str, Any]) -> Any:
        """Each packed matrix's, from its tensor headers."""
        return self.packing

    def weights(self, model: ModelDir, files: Sequence[str] | None, exclude: Sequence[str]) -> Weights:
        """The original packed tensors, with companion names normalized for the shared weight loader."""
        import mlx.core as mx  # MLX, loaded when weights are read

        from .weights import Weights

        values: dict[str, mx.array] = {}
        paths: dict[str, tuple[Path, str]] = {}  # where each is stored, for tables read from their blobs
        for path in self.blobs:
            loaded = mx.load(str(path), format="safetensors")
            assert isinstance(loaded, dict)
            values.update({self.renames.get(k, k): v for k, v in loaded.items()})
            paths.update({self.renames.get(k, k): (path, k) for k in loaded})
        return Weights(values, str(self.manifest), self.packing, paths=paths)

    def media_weights(self, model: ModelDir) -> Weights:
        return self.weights(model, None, ())

    def media_names(self, model: ModelDir, weights: Weights) -> Container[str]:
        return weights.tensors.keys()

    def stored(self, model: ModelDir) -> list[Tensor]:
        return list(self.tensors)

    def origin(self, model: ModelDir) -> Origin:
        return Origin(None, {"ollama": self.spec, "manifest": self.checkpoint})

    def formats(self, model: ModelDir, stored: Sequence[Tensor]) -> dict[str, int]:
        return inspect(stored, model.quantization)

    def model_dir(self, extra: Mapping[str, Any] | None = None) -> ModelDir:
        """The model folder it stands for: its manifest's folder in the store, with its config.json, the build
        serving its files and weights; `extra` as `ModelDir.open` takes it."""
        return ModelDir(self.manifest.parent, _json(self.assets["config.json"]), self, dict(extra or {}))


def _manifest(spec: str) -> tuple[Path, Path]:
    """The Ollama store (OLLAMA_MODELS, or ~/.ollama/models) and the manifest there that
    ollama://[registry/][namespace/]name[:tag] names: registry.ollama.ai, library and latest where it leaves them out.
    Each part is a plain file name (no "." or ".."), so a name never reaches outside the store's manifests."""
    text = spec.removeprefix("ollama://")
    model, colon, tag = text.rpartition(":")
    if not colon or "/" in tag:
        model, tag = text, "latest"
    parts = model.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library", *parts]
    elif len(parts) == 2:
        parts = ["registry.ollama.ai", *parts]
    if len(parts) != 3 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in (".", "..") for p in [*parts, tag]):
        raise UnsupportedModel(f"{spec!r}: invalid Ollama model name")
    root = Path(os.environ.get("OLLAMA_MODELS", "~/.ollama/models")).expanduser().resolve()
    return root, root.joinpath("manifests", *parts, tag)


def pulled(spec: str) -> bool:
    """Whether Ollama has pulled the build an ollama:// name names (its manifest is in the store)."""
    return _manifest(spec)[1].is_file()


def _pulled(spec: str) -> tuple[Path, Path]:
    """`_manifest`'s store and manifest, which must be there."""
    root, manifest = _manifest(spec)
    if not manifest.is_file():
        raise UnsupportedModel(f"{spec}: not pulled locally at {manifest}; pull its MLX variant with Ollama first")
    return root, manifest


def open_build(spec: str, *, verify: bool = True) -> Build:
    """The pulled MLX build an ollama:// name (`_pulled`) resolves to, verifying blob hashes when requested."""
    model, _, tag = spec.removeprefix("ollama://").rpartition(":")
    size = tag.rpartition("-")[0]
    alternative = "its MXFP8 or bf16 Ollama tag"
    if model in CATALOGUED and size:
        alternative = f"ollama://{model}:{size}-mxfp8 or ollama://{model}:{size}-bf16"
    root, manifest = _pulled(spec)

    def blob(entry: dict[str, Any], check_hash: bool) -> Path:
        digest = entry.get("digest", "")
        if not isinstance(digest, str) or DIGEST.fullmatch(digest) is None:
            raise UnsupportedModel(f"{spec}: invalid blob digest {digest!r}")
        path = root / "blobs" / digest.replace(":", "-")
        if not path.is_file() or ("size" in entry and path.stat().st_size != entry["size"]):
            raise UnsupportedModel(f"{spec}: missing or incomplete blob {digest}; pull the model again")
        # hashed the first time this version of the blob is seen, then remembered (hashes.content_sha256): rehashing
        # d3:4b-mxfp8's 5.6 GB took 1.7 s of its 2.8 s load on an M5 Pro, every load
        if check_hash and content_sha256(path) != digest[7:]:
            raise UnsupportedModel(f"{spec}: checksum mismatch for {digest}, so pull the model again")
        return path

    selected = _json(manifest)
    if "manifests" in selected:
        entries = selected["manifests"]
        if not isinstance(entries, list) or any(not isinstance(m, dict) for m in entries):
            raise UnsupportedModel(f"{spec}: invalid manifest list")
        candidates = [m for m in entries if m.get("runner") == "mlx"]
        if len(candidates) != 1:
            raise UnsupportedModel(f"{spec}: expected exactly one MLX manifest; select an explicit MLX tag")
        selected = candidates[0]
        if "digest" in selected and not selected.get("layers"):
            selected = _json(blob(selected, True))
    layers = selected.get("layers", [])
    if not isinstance(layers, list) or any(not isinstance(layer, dict) for layer in layers):
        raise UnsupportedModel(f"{spec}: invalid manifest layers")
    if not any(layer.get("mediaType") == TENSOR for layer in layers):
        raise UnsupportedModel(f"{spec}: this is not an Ollama MLX tensor build; choose its MLX variant")
    assets: dict[str, Path] = {}
    blobs, stored, renames, quantization = [], [], {}, {}
    names: set[str] = set()
    for layer in layers:
        media = layer.get("mediaType")
        if media not in (TENSOR, JSON):
            continue
        path = blob(layer, verify or media == JSON)
        if media == JSON:
            name = layer.get("name", "")
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json") or name in assets:
                raise UnsupportedModel(f"{spec}: invalid or duplicate config layer {name!r}")
            assets[name] = path
            continue
        h = tensor_header(path)
        mapping, q = tensor_metadata(h, alternative)
        for name, value in h.items():
            if name == "__metadata__":
                continue
            canonical = mapping.get(name, name)
            if canonical in names:
                raise UnsupportedModel(f"{spec}: duplicate tensor {canonical!r}")
            names.add(canonical)
            stored.append(Tensor(canonical, value["dtype"], tuple(value["shape"])))
        renames.update(mapping)
        quantization.update(q)
        blobs.append(path)
    if not {"config.json", "tokenizer.json"} <= assets.keys():
        raise UnsupportedModel(f"{spec}: MLX manifest needs config.json and tokenizer.json")
    _json(assets["config.json"])  # malformed model configuration is a model error, before architecture selection
    identity = checkpoint_identity(layers)
    return Build(spec, manifest, identity, assets, tuple(blobs), tuple(stored), quantization, renames)
