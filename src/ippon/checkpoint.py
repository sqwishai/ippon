"""Read checkpoint metadata at the file boundary, with model errors that name what needs repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import UnsupportedModel

# the vocabulary tables, under a model's text root (Weights.text_root): a load reads their rows from the file
# (`Weights.table`), so admission's estimate leaves them out
EMBEDDING, HEAD = "embed_tokens.weight", "lm_head.weight"


def invalid_file(path: Path, reason: object) -> UnsupportedModel:
    """An unreadable or damaged checkpoint file, with a recovery step."""
    return UnsupportedModel(f"{path}: {reason}. Restore a complete, readable copy of this checkpoint from its source.")


def read_json(path: Path) -> Any:
    """A checkpoint's JSON, or UnsupportedModel for unreadable, malformed or excessively nested content."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as error:
        raise invalid_file(path, f"cannot read JSON ({error})") from None


def weight_files(directory: str | Path) -> list[str]:
    """The safetensors shards named by an index, else every safetensors file in the directory."""
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return sorted(path.name for path in directory.glob("*.safetensors"))
    value = read_json(index)
    mapping = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise invalid_file(index, "weight_map must be a non-empty object of tensor names and shard filenames")
    for name in mapping.values():
        if (
            not isinstance(name, str)
            or not name.endswith(".safetensors")
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise invalid_file(index, "shards must be relative .safetensors paths inside the model directory")
        if not (directory / name).is_file():
            raise invalid_file(index, f"missing shard {name!r}")
    return sorted(set(mapping.values()))
