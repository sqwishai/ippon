"""A model's files as an adapter reads them, whatever they were published as.

`ModelDir` is the folder an adapter is handed: its config.json, the JSON files given in code (a bring-your-own model's
`ippon.json`), and a `Source`, which answers for the files and weights in the format the model was published in:

- `Folder`, a folder of safetensors, JSON and text files (a checkpoint, a Hub snapshot);
- `Listing`, a Hub repository's listing and its configs, for a check before any weights are downloaded;
- a GGUF build (`gguf.source.Gguf`): the creator's small files beside one llama.cpp file;
- an Ollama build (`ollama.Build`): a pulled manifest, whose blobs hold the files.

Each source names its files, reads its JSON, says how its packed weights are stored (and counts them by format), says
what identifies the artifact it reads (`Origin`, and an Ollama build's checkpoint), and reads the weights an adapter
asks for (`files` only, or the model's own but those in `exclude`) under the creator's names. Nothing here loads MLX:
locating a model (`ippon check`, `pull`, the MCP server) never does. A source imports what reads tensors (MLX) when
it reads them.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .checkpoint import invalid_file, read_json, weight_files
from .errors import UnsupportedModel
from .quantization import inspect
from .safetensors import Tensor, tensors

if TYPE_CHECKING:
    from .weights import Weights

API_VERSION = 1  # the adapter API's version (adapter.py), which an adapter names as its `api`
SETTINGS = "ippon.json"  # a model's settings file (`ModelDir.settings`)
# what a LoRA adapter's base loads from (Kev's), its tokenizer included for an adapter that ships none
BASE_FILES = (
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",  # its licence (licences.of_folder)
    "LICENSE",
)

Kind = Literal["folder", "listing", "gguf", "ollama"]


@dataclass(frozen=True)
class Origin:
    """What identifies the artifact a source reads (provenance.describe): the file or folder whose path, or the Hub
    cache snapshot it is in, names it (None for a build identified by itself alone, an Ollama build), and what else
    identifies it, as a model's `build` gives it (a GGUF file's name and sha256, an Ollama build's name and
    manifest)."""

    path: Path | None
    identity: Mapping[str, str] = field(default_factory=dict)


class Source(Protocol):
    """Where a model's files and weights are read from."""

    @property
    def kind(self) -> Kind: ...

    @property
    def as_stored(self) -> bool:
        """Whether it is a quantized build, which loads only as it is stored (never quantized again)."""
        ...

    @property
    def checkpoint(self) -> str | None:
        """The identity of the checkpoint it reads, when the source carries one (an Ollama build's: its tensor blobs
        and the assets its adapter reads, ollama.checkpoint_identity); None when its files say what it is (`origin`)."""
        ...

    def origin(self, model: ModelDir) -> Origin:
        """What identifies the artifact it reads (`Origin`)."""
        ...

    def formats(self, model: ModelDir, stored: Sequence[Tensor]) -> dict[str, int]:
        """Its packed matrices counted by format ("affine", "mxfp8", ...), from `stored`, its tensors' headers, which
        quantization.inspect also checks; {} for a format whose file describes its own matrices (GGUF)."""
        ...

    def file(self, name: str) -> Path:
        """Where the file `name` is (it may not exist: `has`)."""
        ...

    def has(self, name: str) -> bool: ...

    def quantization(self, config: Mapping[str, Any]) -> Any:
        """The metadata its packed weights are read with, as stored (`quantization.Quantization.read` checks it)."""
        ...

    def weights(self, model: ModelDir, files: Sequence[str] | None, exclude: Sequence[str]) -> Weights:
        """The tensors of `files` (the creator's file names), else of the model's own weight files but those in
        `exclude`, under the creator's names."""
        ...

    def media_weights(self, model: ModelDir) -> Weights:
        """The weights a picture, video or audio path reads."""
        ...

    def media_names(self, model: ModelDir, weights: Weights) -> Container[str]:
        """The names of the picture and audio tensors there are to load, `weights` being the model's."""
        ...

    def stored(self, model: ModelDir) -> list[Tensor]:
        """Its tensors as they are stored, from their headers (no weights read), as admission counts them."""
        ...


@dataclass(frozen=True)
class ModelDir:
    """A model: its directory, its config.json ({} when it has none), where its files and weights are read from
    (`source`), and JSON files given in code (`extra`, by file name: a bring-your-own model's `ippon.json` passed to
    `Decider.load(settings=...)`), which stand in for its own."""

    path: Path
    config: Mapping[str, Any]
    source: Source
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, path: str | Path, extra: Mapping[str, Any] | None = None) -> ModelDir:
        """A folder of files."""
        path = Path(path).expanduser()
        return cls(path, config_of(path), Folder(path), dict(extra or {}))

    def file(self, name: str) -> Path:
        return self.source.file(name)

    def has(self, name: str) -> bool:
        return name in self.extra or self.source.has(name)

    def json(self, name: str) -> Any:
        """The JSON file `name` (given in code, else the model's own), or None when it is absent."""
        if name in self.extra:
            return self.extra[name]
        if not self.source.has(name):
            return None
        file = self.source.file(name)
        return read_json(file) if file.is_file() else None

    def settings(self) -> Mapping[str, Any]:
        """The model's `ippon.json` (given in code, else in the directory): how a bring-your-own model is asked, or a
        choice among its creator's modes; {} when it has none, UnsupportedModel when it is not an object."""
        found = self.json(SETTINGS)
        if found is None:
            return {}
        if not isinstance(found, Mapping):
            raise UnsupportedModel(f"{self.path}: {SETTINGS} must contain an object")
        return found

    @property
    def quantization(self) -> Any:
        """The metadata its packed weights are read with, as stored (`Source.quantization`)."""
        return self.source.quantization(self.config)


def config_of(path: Path) -> Mapping[str, Any]:
    """The config.json in folder `path`, {} when there is none."""
    config = path / "config.json"
    value = read_json(config) if config.is_file() else {}
    if not isinstance(value, Mapping):
        raise invalid_file(config, "expected a JSON object")
    return value


def configured_quantization(config: Mapping[str, Any]) -> Any:
    """A checkpoint's quantization as its config gives it: "quantization" (MLX's name) or "quantization_config"
    (Hugging Face's); None for neither."""
    return config.get("quantization") or config.get("quantization_config")


@dataclass(frozen=True)
class Folder:
    """A folder of files: safetensors weights, JSON and text."""

    path: Path
    kind: Kind = field(default="folder", init=False)
    as_stored: bool = field(default=False, init=False)
    checkpoint: str | None = field(default=None, init=False)

    def file(self, name: str) -> Path:
        return self.path / name

    def has(self, name: str) -> bool:
        return self.file(name).is_file()

    def quantization(self, config: Mapping[str, Any]) -> Any:
        return configured_quantization(config)

    def weights(self, model: ModelDir, files: Sequence[str] | None, exclude: Sequence[str]) -> Weights:
        """The safetensors `files`, else the model's weight files (its index's, else every *.safetensors), but those
        in `exclude`; quantized ones read as `ModelDir.quantization` describes them."""
        from .weights import Weights  # MLX, loaded when weights are read

        names = [f for f in (weight_files(self.path) if files is None else files) if f not in exclude]
        quantization = model.quantization
        return Weights.from_dir(self.path, names, quantization if isinstance(quantization, dict) else None)

    def media_weights(self, model: ModelDir) -> Weights:
        return self.weights(model, None, ())

    def media_names(self, model: ModelDir, weights: Weights) -> Container[str]:
        return weights.tensors.keys()

    def stored(self, model: ModelDir) -> list[Tensor]:
        return tensors(self.path)

    def origin(self, model: ModelDir) -> Origin:
        return Origin(model.path)

    def formats(self, model: ModelDir, stored: Sequence[Tensor]) -> dict[str, int]:
        return inspect(stored, model.quantization)


@dataclass(frozen=True)
class Listing:
    """A Hub repository before its weights are downloaded: its configs in `path`, and the files its listing names
    (`files`), which a check reads presence from. It holds no weights."""

    path: Path
    files: frozenset[str]
    kind: Kind = field(default="listing", init=False)
    as_stored: bool = field(default=False, init=False)
    checkpoint: str | None = field(default=None, init=False)

    def file(self, name: str) -> Path:
        return self.path / name

    def has(self, name: str) -> bool:
        return name in self.files

    def quantization(self, config: Mapping[str, Any]) -> Any:
        return configured_quantization(config)

    def weights(self, model: ModelDir, files: Sequence[str] | None, exclude: Sequence[str]) -> Weights:
        raise UnsupportedModel(f"{self.path}: a repository's listing holds no weights; download the model to load it")

    def media_weights(self, model: ModelDir) -> Weights:
        return self.weights(model, None, ())

    def media_names(self, model: ModelDir, weights: Weights) -> Container[str]:
        return weights.tensors.keys()

    def stored(self, model: ModelDir) -> list[Tensor]:
        raise UnsupportedModel(f"{self.path}: a repository's listing holds no tensor headers")

    def origin(self, model: ModelDir) -> Origin:
        return Origin(model.path)

    def formats(self, model: ModelDir, stored: Sequence[Tensor]) -> dict[str, int]:
        return inspect(stored, model.quantization)
