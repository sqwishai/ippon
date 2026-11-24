"""A GGUF build's tensors read into MLX under the creator's names (names.py), llama.cpp's transforms undone, so each
adapter reads them as it reads the creator's checkpoint: Q8_0, Q4_0, Q4_K and Q5_K matrices as MLX's affine groups of
32, Q6_K ones unpacked to fp16 (reader.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path

from ..errors import UnsupportedModel
from ..weights import Weights
from . import names
from .header import GROUP_SIZE, read_header
from .reader import host_rows, load


def weights(path: Path, reads: names.Reads, files: Sequence[str] | None) -> Weights:
    """The tensors of the GGUF file at `path` under the creator's names (`reads`, its adapter's), Q8_0, Q4_0, Q4_K and
    Q5_K matrices quantized as MLX reads them: those of the creator's file in `files` that the build stores as a part
    of its own (a head the creator stores apart), else the model's own, which leave every such part out."""
    header = read_header(path)
    rules, backbone = reads.tensors(header)
    parts = {r.part for r in rules} - {""}
    part = next((f for f in files or () if f in parts), "")
    if files is not None and not part:
        raise UnsupportedModel(f"{path.name} holds the model's weights, not {', '.join(files)}")
    plain = names.plain(header, rules, backbone, part)
    return Weights(
        names.rename(header, load(header), rules, backbone, part),
        str(path),
        quantization={"group_size": GROUP_SIZE, "mode": "affine"},
        hosted={target: partial(host_rows, header, header.tensors[found]) for target, found in plain.items()},
    )


def media_weights(mmproj: Path | None, reads: names.Reads) -> Weights:
    """The picture and audio parts in the mmproj file at `mmproj` under the creator's names (empty without one)."""
    if mmproj is None:
        return Weights({}, "")
    header = read_header(mmproj)
    return Weights(
        names.rename(header, load(header), _media_rules(mmproj, reads), 0),
        str(mmproj),
        quantization={"group_size": GROUP_SIZE, "mode": "affine"},
    )


def media_names(mmproj: Path | None, reads: names.Reads) -> set[str]:
    """The creator's names of the tensors the mmproj file at `mmproj` holds, from its header alone."""
    if mmproj is None:
        return set()
    return names.targets(read_header(mmproj), _media_rules(mmproj, reads))


def _media_rules(mmproj: Path, reads: names.Reads) -> list[names.Rule]:
    if reads.media is None:
        raise UnsupportedModel(f"ippon cannot read the picture parts in {mmproj.name}")
    return reads.media()
