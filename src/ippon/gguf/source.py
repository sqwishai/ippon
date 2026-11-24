"""A GGUF build as a model's source (`model_dir.Source`): the creator's small files at the pinned revision of its base
repository, beside one llama.cpp file whose tensors stand in for the creator's weights, and its mmproj file (the
picture and audio parts), when it is there.

A build is read by the rules of the adapter its creator's files choose (`bound`), once its metadata is checked against
the creator's configuration by that adapter (`gguf_check`, with the architectures' checks here): a build whose
dimensions, heads, RoPE, causality or decision settings differ from its creator's checkpoint is refused rather than
run on the wrong configuration. Reading the tensors is tensors.py's, which loads MLX.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import hashes
from ..errors import UnsupportedModel
from ..model_dir import Kind, ModelDir, Origin, configured_quantization
from ..safetensors import Tensor as Stored
from . import builds
from .header import GROUP_SIZE, QUANTIZED, UNPACKED, Header, close, read_header

if TYPE_CHECKING:
    from ..adapter import Adapter
    from ..weights import Weights
    from .names import Reads


@dataclass(frozen=True)
class Gguf:
    """A GGUF build's files: the creator's small files (`folder`), the build's file (`path`), which build it is, and
    its mmproj file, when it is beside it (None: the build reads text only, as llama.cpp does without --mmproj); and
    how its tensors read, its adapter's `gguf_rules`, once the adapter is chosen (`bound`)."""

    folder: Path
    path: Path
    build: builds.Build
    mmproj: Path | None = None
    reads: Reads | None = None
    kind: Kind = field(default="gguf", init=False)
    as_stored: bool = field(default=True, init=False)
    checkpoint: str | None = field(default=None, init=False)

    def file(self, name: str) -> Path:
        return self.folder / name

    def has(self, name: str) -> bool:
        return self.file(name).is_file()

    def quantization(self, config: Mapping[str, Any]) -> Any:
        """Its config's, as the creator's checkpoint gives it: the GGUF file describes its own matrices
        (tensors.weights)."""
        return configured_quantization(config)

    def weights(self, model: ModelDir, files: Sequence[str] | None, exclude: Sequence[str]) -> Weights:
        from . import tensors  # MLX, loaded when weights are read

        return tensors.weights(self.path, self._reads(), files)

    def media_weights(self, model: ModelDir) -> Weights:
        from . import tensors  # MLX, loaded when weights are read

        return tensors.media_weights(self.mmproj, self._reads())

    def media_names(self, model: ModelDir, weights: Weights) -> Container[str]:
        """The creator's names of the tensors its mmproj file holds, from the file's header alone."""
        from . import tensors  # MLX, loaded when weights are read

        return tensors.media_names(self.mmproj, self._reads())

    def origin(self, model: ModelDir) -> Origin:
        """Its file, by name and sha256: the Hub cache blob's name when it is the pinned hash, else hashed once
        (hashes.py)."""
        return Origin(self.path, {"file": self.path.name, "sha256": hashes.sha256(self.path, self.build.sha256)})

    def formats(self, model: ModelDir, stored: Sequence[Stored]) -> dict[str, int]:
        return {}  # the file describes its own matrices (tensors.weights)

    def _reads(self) -> Reads:
        if self.reads is None:
            raise UnsupportedModel(f"{self.path.name}: no adapter that reads {self.build.repo}'s builds was chosen")
        return self.reads

    def stored(self, model: ModelDir) -> list[Stored]:
        """The tensors the build holds once loaded, as admission counts them: a quantized matrix as MLX keeps it
        (its packed uint32 weight, fp16 scales and biases per group of 32), a Q6_K one as fp16, the rest as stored."""
        out = []
        for t in read_header(self.path).tensors.values():
            if t.type in QUANTIZED:
                rows, cols = t.size // t.shape[-1], t.shape[-1]
                bits = QUANTIZED[t.type]
                out += [
                    Stored(t.name, "U32", (rows, cols * bits // 32)),
                    Stored(t.name + ".scales", "F16", (rows, cols // GROUP_SIZE)),
                    Stored(t.name + ".biases", "F16", (rows, cols // GROUP_SIZE)),
                ]
            else:
                out.append(Stored(t.name, "F16" if t.type in UNPACKED else t.type, t.shape))
        return out


def pinned(path: Path, repo: str, file: str, revision: str, digest: str) -> Header:
    """The header of the GGUF file at `path`, which must be `repo`'s `file` at `revision` (its header `digest`), and of
    types ippon reads; UnsupportedModel otherwise."""
    header = read_header(path)
    if header.digest != digest:
        raise UnsupportedModel(f"{path.name}: its header is not {repo}/{file}'s at {revision[:8]}")
    header.refuse_unread()
    return header


# -- the GGUF's settings against the creator's -----------------------------------------------------------------------


def bound(directory: ModelDir, header: Header, adapter: Adapter) -> ModelDir:
    """`directory`, a GGUF build (`header` its file's), read by `adapter`'s rules (`gguf_rules`) once its metadata
    agrees with the creator's configuration (`adapter.gguf_check`); UnsupportedModel, naming each difference, when it
    doesn't, or when the adapter reads no GGUF build."""
    source = directory.source
    assert isinstance(source, Gguf)
    if adapter.gguf_rules is None:
        raise UnsupportedModel(f"{header.path.name}: {adapter.name} reads no GGUF build")
    differ = adapter.gguf_check(header, directory)
    if differ:
        raise UnsupportedModel(
            f"{header.path.name} is not the model of {directory.path.name}'s config: " + "; ".join(differ)
        )
    return replace(directory, source=replace(source, reads=adapter.gguf_rules))


def lfm3_differences(header: Header, config: Mapping[str, Any]) -> list[str]:
    """How an LFM3 build differs from the creator's text config: its dimensions, layer kinds and both RoPEs (the zero
    tail as rope_freqs' factors)."""
    text = config["text_config"]
    layer_types = list(text["layer_types"])
    ropes = text["rope_parameters"]
    expected = {
        "block_count": len(layer_types),
        "embedding_length": text["hidden_size"],
        "feed_forward_length": text["intermediate_size"],
        "attention.head_count": text["num_attention_heads"],
        "attention.key_length": text["head_dim"],
        "attention.layer_norm_rms_epsilon": text.get("norm_eps", 1e-5),
        "attention.sliding_window": text["sliding_window"],
        "rope.freq_base": ropes["full_attention"]["rope_theta"],
        "rope.freq_base_swa": ropes["sliding_attention"]["rope_theta"],
        "shortconv.l_cache": text["conv_L_cache"],
    }
    differ = [
        f"{key} {header.get(key)} against {want}"
        for key, want in expected.items()
        if header.get(key) is None or not close(header.get(key), want)
    ]
    kv = list(header.get("attention.head_count_kv") or [])
    if kv != [0 if t == "conv" else text["num_key_value_heads"] for t in layer_types]:
        differ.append("its attention layers or key-value heads differ")
    if list(header.get("attention.sliding_window_pattern") or []) != [t == "sliding_attention" for t in layer_types]:
        differ.append("its sliding layers differ")
    pairs = text["head_dim"] // 2
    turned = int(float(ropes["full_attention"].get("partial_rotary_factor", 1.0)) * text["head_dim"]) // 2
    t = header.tensors.get("rope_freqs.weight")
    if t is not None and t.type == "F32":
        import numpy as np

        factors = np.fromfile(header.path, "<f4", pairs, offset=t.offset)
    else:
        factors = None
    if factors is None or list(factors[:turned]) != [1.0] * turned or not (factors[turned:] >= 1e29).all():
        differ.append(f"its rope_freqs are not {turned} of {pairs} pairs turning")
    return differ


def qwen35_differences(header: Header, config: Mapping[str, Any]) -> list[str]:
    """How a Qwen3.5 build differs from the creator's config: its dimensions and layer kinds; the speculative decoding
    layers past the backbone's (nextn_predict_layers) count in block_count and stay out of recurrent_layers'
    comparison."""
    extra_blocks = int(header.get("nextn_predict_layers", 0))
    text = config.get("text_config", config)
    rope = text.get("rope_parameters") or {}
    layer_types = list(text["layer_types"])
    expected = {
        "block_count": len(layer_types) + extra_blocks,
        "embedding_length": text["hidden_size"],
        "feed_forward_length": text["intermediate_size"],
        "attention.head_count": text["num_attention_heads"],
        "attention.head_count_kv": text["num_key_value_heads"],
        "attention.key_length": text["head_dim"],
        "attention.layer_norm_rms_epsilon": text["rms_norm_eps"],
        "rope.freq_base": rope.get("rope_theta", text.get("rope_theta")),
        "rope.dimension_count": int(text["head_dim"] * rope.get("partial_rotary_factor", 0.25)),
        "ssm.group_count": text["linear_num_key_heads"],
        "ssm.time_step_rank": text["linear_num_value_heads"],
        "ssm.state_size": text["linear_key_head_dim"],
        "ssm.inner_size": text["linear_num_value_heads"] * text["linear_value_head_dim"],
        "ssm.conv_kernel": text["linear_conv_kernel_dim"],
    }
    differ = [
        f"{key} {header.get(key)} against {want}"
        for key, want in expected.items()
        if header.get(key) is None or not close(header.get(key), want)
    ]
    recurrent = list(header.get("attention.recurrent_layers") or [])[: len(layer_types)]
    if recurrent != [t == "linear_attention" for t in layer_types]:
        differ.append("its recurrent and attention layers differ")
    return differ
