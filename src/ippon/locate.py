"""A model by the reference a caller gives: its files, the adapter that runs them, and what downloading it takes.

A reference is one of five forms (names.parse): a Hub id (org/name[@revision]), a GGUF build ippon knows
(repo:QUANT), a GGUF file on this Mac, an Ollama build Ollama has pulled (ollama://name:tag) or a folder; a short
name (names.py) resolves to one of them. Three things read a reference, each through this module:

- a load (`open_model`, `Decider.load`): the model's files downloaded, its adapter, and a LoRA adapter's base;
- a check (`inspect`, `ippon check` and `ippon pull`): its small files (or, before a pull, only its configs and the
  Hub's listing), its adapter, its tensors from their headers, and the files a pull downloads (`Download`);
- `downloaded`: whether its weights are on this Mac, found without the network.

A Hub id downloads in two steps: the small JSON and text files first, which choose the adapter, then the files that
adapter reads, from the commit the first step read. Hub metadata is read through `downloads.info`. Nothing here loads
MLX: adapters load it when they are chosen, and a source when it reads weights.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import downloads, hub, licences, names, ollama, registry
from .admission import resident_bytes
from .checkpoint import invalid_file, read_json, weight_files
from .errors import DownloadFailed, ModelNotFound, UnsupportedModel
from .gguf import builds
from .gguf import source as gguf_source
from .gguf.header import Header, read_header
from .hub import Download
from .model_dir import BASE_FILES, SETTINGS, Listing, ModelDir, config_of
from .quantization import get as get_precision
from .safetensors import Tensor, tensors

if TYPE_CHECKING:
    from .adapter import Adapter


@dataclass(frozen=True)
class Opened:
    """A model ready to load: its files, the adapter that runs them, and a LoRA adapter's base model."""

    directory: ModelDir
    adapter: Adapter
    base: ModelDir | None

    def stored(self) -> list[Tensor]:
        """The tensors its files hold, from their headers (no weights read), a LoRA adapter's base's too: the
        adapter's own tensors merge into the base's, so the base's size is the model's."""
        found = self.directory.source.stored(self.directory)
        if self.base is not None:
            found += tensors(self.base.path, weight_files(self.base.path))
        return found


@dataclass(frozen=True)
class Inspected:
    """A model as a check finds it before its weights are downloaded: its files (small ones, or configs and a
    listing), its adapter, the revision it reads, what a pull downloads (`downloads`, and their count and bytes in
    `download`), its tensors from their headers (none for a GGUF build, which is `resident` bytes as it is stored),
    whether it is a quantized build that loads only as stored (its source's `as_stored`), a LoRA adapter's base
    (repo, revision), the organisation that published it (`creator`, "local model" for a folder or an Ollama build),
    and where it is when it is on this Mac already (`local`: a folder or a GGUF file)."""

    directory: ModelDir
    adapter: Adapter
    revision: str | None
    downloads: tuple[Download, ...]
    download: tuple[int, int]
    stored: list[Tensor]
    resident: int | None
    as_stored: bool
    base: tuple[str, str] | None
    creator: str
    local: str | None


def open_model(
    model: str | Path,
    adapter: str | None = None,
    settings: Mapping[str, Any] | None = None,
    accept_license: bool = False,
) -> Opened:
    """The model's files and the adapter that runs them (the one named `adapter`, else the one its files match), and
    a LoRA adapter's base, downloaded with the licences `accept_license` accepts; a licence's hint names the model as
    the caller did. `settings`: a bring-your-own model's ippon.json given in code."""
    extra = {SETTINGS: dict(settings)} if settings is not None else None
    with licences.accepting(accept_license, str(model)):
        text = str(names.resolve(model))
        form = names.parse(text).form
        if form == "ollama":
            directory = ollama.open_build(text).model_dir(extra)
            found = registry.select(directory, adapter)
        elif form in ("gguf", "gguf-file"):
            directory, header = _gguf_build(text, extra)
            found = registry.select(directory, adapter)
            directory = gguf_source.bound(directory, header, found)
        elif form == "path":
            directory = ModelDir.open(_folder(text), extra)
            found = registry.select(directory, adapter)
        else:
            small = hub.fetch(text, hub.probe())
            found = registry.select(ModelDir.open(small, extra), adapter)
            directory = ModelDir.open(hub.fetch(hub.pinned(text, small), found.files), extra)
        _refuse_activation_quantized(directory)
        base = open_base(found, directory)
        if base is not None:
            _refuse_activation_quantized(base)
        return Opened(directory, found, base)


def open_base(found: Adapter, directory: ModelDir) -> ModelDir | None:
    """A LoRA adapter's base model, as its adapter's `base` names it, downloaded at that revision (its BASE_FILES);
    None for a model with weights of its own."""
    pinned = found.base(directory)
    if pinned is None:
        return None
    repo, revision = pinned
    return ModelDir.open(hub.fetch(f"{repo}@{revision}", BASE_FILES))


def estimate(
    model: str | Path,
    *,
    adapter: str | None = None,
    accept_license: bool = False,
    precision: str = "auto",
    settings: Mapping[str, Any] | None = None,
) -> int:
    """The bytes `Decider.load` would find the model holds resident at `precision`, before loading it: admission's
    estimate from its tensors' headers (admission.py); what `serve` weighs to decide which models stay loaded."""
    meant = get_precision(names.named_precision(model, precision))
    return resident_bytes(open_model(model, adapter, settings, accept_license).stored(), meant.width, meant.bits)


def inspect(model: str, adapter: str | None = None, *, metadata_only: bool = False) -> Inspected:
    """`model` as a check finds it, downloading only its small files, or with `metadata_only` (before a pull) only
    its configs, read with the Hub's listing of its files."""
    text = str(model)
    form = names.parse(text).form
    revision: str | None = None
    planned: list[Download] = []
    stored: list[Tensor] = []
    files, size = 0, 0
    resident: int | None = None
    creator = ""
    as_stored: bool | None = None  # a GGUF build's, whose source is bound only once its file is read
    local: str | None = None
    if form == "ollama":
        build = ollama.open_build(text, verify=False)
        directory = build.model_dir()
        found = registry.select(directory, adapter)
        stored, revision = list(build.tensors), build.checkpoint
    elif form == "gguf":
        named = builds.find(text)
        beside = f"{named.base}@{named.base_revision}"
        directory = _small(beside, metadata_only)
        wanted = (named.file, *((named.mmproj,) if named.mmproj else ()))  # the model, then its picture/audio parts
        planned += [
            Download(f"{named.repo}@{named.revision}", wanted, pins=(named.sha256, named.mmproj_sha256)[: len(wanted)]),
            Download(beside, builds.SMALL_FILES),
        ]
        found = registry.select(directory, adapter)
        as_stored = gguf_source.Gguf.as_stored
        sizes = _gguf_sizes(named, wanted)
        resident = sizes[0]  # resident as stored; the mmproj only once a picture or clip arrives
        files, size, revision, creator = len(wanted), sum(sizes), named.revision, named.base
    elif form == "gguf-file":  # a GGUF file on this Mac: one of the builds ippon knows, by its header
        path, _, named = _local_gguf(text)
        beside = f"{named.base}@{named.base_revision}"  # the creator's config and tokenizer the file is read with
        directory = _small(beside, metadata_only)
        planned.append(Download(beside, builds.SMALL_FILES))
        found = registry.select(directory, adapter)
        as_stored = gguf_source.Gguf.as_stored
        resident, revision, creator, local = path.stat().st_size, named.revision, named.base, text
    elif form == "path":
        directory = ModelDir.open(_folder(text))
        found = registry.select(directory, adapter)
        stored, local = _local(directory.path, tuple(found.files)), text
    else:
        if metadata_only:
            try:
                directory = _listed(text)
                small = directory.path
            except DownloadFailed:  # no Hub: a complete download already here answers, as an offline load reads it
                small = hub.fetch(text, hub.probe())
                directory = ModelDir.open(small)
        else:
            small = hub.fetch(text, hub.probe())
            directory = ModelDir.open(small)
        pinned = hub.pinned(text, small)
        revision = names.split(pinned)[1]
        found = registry.select(directory, adapter)
        planned.append(Download(pinned, tuple(found.files), requested=text))
        stored, files, size = _stored(pinned, tuple(found.files))
        if not files and found.base(directory) is None:
            raise UnsupportedModel(f"{model}: no supported safetensors weights")
        creator = pinned
    found.validate(directory)
    _refuse_activation_quantized(directory)
    base = found.base(directory)
    if base is not None:  # a LoRA: its base's weights are read too
        planned.append(Download(f"{base[0]}@{base[1]}", BASE_FILES))
        more, count, more_bytes = _stored(f"{base[0]}@{base[1]}", BASE_FILES)
        stored += more
        files, size = files + count, size + more_bytes
    # a folder or an Ollama build names no publisher of its own: a LoRA folder's is its base's
    creator = creator or (planned[0].model if planned else "")
    return Inspected(
        directory,
        found,
        revision,
        tuple(planned),
        (files, size),
        stored,
        resident,
        directory.source.as_stored if as_stored is None else as_stored,
        base,
        creator.split("/")[0] if creator else "local model",
        local,
    )


def downloaded(model: str) -> bool:
    """Whether `model`'s weights are on this Mac, found without the network: a folder or a GGUF file that exists, an
    Ollama build Ollama has pulled, a GGUF build (repo:QUANT) whose file is in the Hub cache, or a Hub id a load
    downloaded whole (`hub.downloaded_weights`). Only a model that isn't there is False: a name that names nothing
    ippon knows, or a damaged manifest or adapter config, is an error."""
    return _downloaded(str(model), set())


def _downloaded(text: str, seen: set[str]) -> bool:
    if text in seen:  # a LoRA whose base is itself, by some path: never complete
        return False
    seen.add(text)
    reference = names.parse(text)
    if reference.form == "ollama":
        if not ollama.pulled(text):
            return False
        ollama.open_build(text, verify=False)
        return True
    if reference.form == "gguf-file":
        return Path(text).expanduser().is_file()
    if reference.form == "path":
        path = Path(text).expanduser()
        return path.is_dir() and any(path.glob("*.safetensors")) and _base_downloaded(path, seen)
    if reference.form == "gguf":
        named = builds.find(text)
        return hub.cached_file(named.repo, named.file, named.revision) is not None
    folder = hub.downloaded_weights(text)
    return folder is not None and _base_downloaded(folder, seen)


def _base_downloaded(path: Path, seen: set[str]) -> bool:
    """An adapter is usable offline only when its base is present too."""
    config = path / "adapter_config.json"
    if not config.exists():
        return True
    adapter = read_json(config)
    base = adapter.get("base_model_name_or_path") if isinstance(adapter, dict) else None
    revision = adapter.get("revision") if isinstance(adapter, dict) else None
    if not isinstance(base, str) or not base or (revision is not None and not isinstance(revision, str)):
        raise invalid_file(config, "expected base_model_name_or_path and an optional revision, as text")
    return _downloaded(base + (f"@{revision}" if revision else ""), seen)


def _gguf_build(text: str, extra: Mapping[str, Any] | None) -> tuple[ModelDir, Header]:
    """A GGUF build's files and its file's header: the build's file (downloaded for one named by its spec, else the
    local file, recognised by its header), the creator's small files at the pinned revision, and its mmproj file."""
    if text.endswith(".gguf"):
        path, _, named = _local_gguf(text)
    else:
        named = builds.find(text)
        path = hub.pinned_file(named.repo, named.revision, named.file, named.sha256)
    header = gguf_source.pinned(path, named.repo, named.file, named.revision, named.header)
    folder = hub.fetch(f"{named.base}@{named.base_revision}", builds.SMALL_FILES)
    mmproj = _mmproj(named, path, fetch=not text.endswith(".gguf"))
    source = gguf_source.Gguf(folder, path, named, mmproj)
    return ModelDir(folder, config_of(folder), source, dict(extra or {})), header


def _local_gguf(text: str) -> tuple[Path, Header, builds.Build]:
    """A GGUF file on this Mac, its header, and the build ippon knows it as, recognised by that header."""
    path = Path(text).expanduser()
    if not path.is_file():
        raise ModelNotFound(f"{path}: no such file")
    header = read_header(path)
    header.refuse_unread()
    return path, header, builds.by_header(header.digest)


def _mmproj(named: builds.Build, path: Path, *, fetch: bool) -> Path | None:
    """The build's mmproj file: beside its GGUF file (a llama.cpp user's folder, or the Hub cache's snapshot), or for
    a build named by its spec downloaded with it, as llama.cpp's --mmproj reads it; None for a build without one or a
    local file without it beside, which then reads text only. Its header must be the pinned one."""
    if not named.mmproj:
        return None
    found = path.parent / named.mmproj
    if not found.is_file():
        if not fetch:
            return None
        found = hub.pinned_file(named.repo, named.revision, named.mmproj, named.mmproj_sha256)
    gguf_source.pinned(found, named.repo, named.mmproj, named.revision, named.mmproj_header)
    return found


def _folder(text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_dir():
        raise ModelNotFound(f"{text}: no such folder (a model is a folder, a .gguf file or a Hub id)")
    return path


def _small(model: str, metadata_only: bool) -> ModelDir:
    """A GGUF build's creator's small files: downloaded, or with `metadata_only` only its configs (`_listed`)."""
    return _listed(model) if metadata_only else ModelDir.open(hub.fetch(model, builds.SMALL_FILES))


def _listed(model: str) -> ModelDir:
    """`model` before anything but its configs is downloaded (`downloads.configs`): presence from the Hub's
    listing, no tokenizers or weights."""
    found = downloads.configs(model)
    return ModelDir(
        found.folder, found.values.get("config.json", {}), Listing(found.folder, found.listed), found.values
    )


def _refuse_activation_quantized(directory: ModelDir) -> None:
    if directory.config.get("quantize_activations"):
        raise UnsupportedModel(
            "activation-quantized checkpoints need their activation quantizer; only weight quantization is supported"
        )


def _gguf_sizes(named: builds.Build, wanted: tuple[str, ...]) -> list[int]:
    """The bytes of a GGUF build's files: from the Hub cache when they are all there, else from the Hub's listing."""
    cached = [hub.cached_file(named.repo, f, named.revision) for f in wanted]
    if all(c is not None for c in cached):
        return [c.stat().st_size for c in cached if c is not None]
    metadata = downloads.info(f"{named.repo}@{named.revision}")
    listed = {s.rfilename: int(s.size or 0) for s in downloads.listing(metadata, wanted, gguf=True)}
    return [listed.get(f, 0) for f in wanted]


def _remote(model: str, patterns: tuple[str, ...]) -> tuple[list[Tensor], int, int]:
    """The tensors of the safetensors files of `model` (repo@revision) that `patterns` match, and the files' count
    and bytes, without downloading them: each file's header is read on the Hub."""
    from huggingface_hub import HfApi

    metadata = downloads.info(model)
    wanted = [s for s in downloads.listing(metadata, patterns) if s.rfilename.endswith(".safetensors")]
    try:
        headers = [
            HfApi().parse_safetensors_file_metadata(metadata.id, s.rfilename, revision=metadata.sha) for s in wanted
        ]
    except Exception as error:  # the hub raises many types; hub_failure says which
        raise downloads.hub_failure(model, "its files can't be listed", error) from error
    found = [Tensor(name, t.dtype, tuple(t.shape)) for h in headers for name, t in h.tensors.items()]
    return found, len(headers), sum(s.size or 0 for s in wanted)


def _cached(model: str, patterns: tuple[str, ...]) -> tuple[Path, list[str]] | None:
    """The Hub cache's snapshot of `model` at the commit it names and its weight files that `patterns` match, when
    another tool (or an ippon cache since cleared) left them whole there: every shard its index lists, which the Hub
    links into a snapshot only once its download is complete. Trusted as the Hub stored them, so a model already on
    this Mac is checked from its own files rather than read again on the Hub."""
    repo, revision = names.split(model)
    if revision is None or not names.is_commit(revision):
        return None
    folder = hub.snapshot_dir(repo, revision)
    try:
        weights = weight_files(folder) if folder.is_dir() else []
    except UnsupportedModel:  # a shard its index lists isn't there: not whole
        return None
    matched = [w for w in weights if any(fnmatch.fnmatch(w, p) for p in patterns)]
    return (folder, matched) if matched else None


def _stored(model: str, patterns: tuple[str, ...]) -> tuple[list[Tensor], int, int]:
    """`_remote`'s tensors, file count and bytes of `model` (repo@commit), from its files when they are all in the
    Hub cache (`hub.downloaded_files`, `_cached`), so a model on this Mac is checked offline; else from the Hub."""
    local = hub.downloaded_files(model, patterns) or _cached(model, patterns)
    if local is None:
        return _remote(model, patterns)
    path, found = local
    weights = [f for f in found if f.endswith(".safetensors")]
    return tensors(path, weights), len(weights), sum((path / f).stat().st_size for f in weights)


def _local(directory: Path, patterns: tuple[str, ...]) -> list[Tensor]:
    files = [
        p.name for p in sorted(directory.glob("*.safetensors")) if any(fnmatch.fnmatch(p.name, q) for q in patterns)
    ]
    return tensors(directory, files)
