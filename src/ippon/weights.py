"""A model's weights, from safetensors files only.

Checkpoints of one architecture name their tensors under different roots: `model.` for a causal LM,
`model.language_model.` or `language_model.` for a multimodal checkpoint whose text model is all a decision
model uses. `Weights.text_root()` finds the root that holds the embedding and the first layer; the
architecture then asks for names relative to it. Arrays load lazily (mx.load), so tensors that are never asked
for, a vision tower or a multi-token-prediction head, are never read.

Weights are never read from a pickle: `.bin`, `.pt` and `.pth` files are refused. (A small head an adapter names
exactly, such as Kev's head.pt, is read by ptfile, which runs nothing from it.)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from .checkpoint import EMBEDDING, invalid_file, weight_files
from .errors import UnsupportedModel, as_decision_error, out_of_memory
from .host_rows import MLX, HostRows
from .quant import QuantizedMatrix, Table
from .quantization import Quantization, biases_of, global_scale_names, packed, scales_of

TEXT_ROOTS = ("model.language_model.", "language_model.model.", "language_model.", "model.", "")
# PEFT LoRA settings under which an adapted layer is not W + (alpha / r) * B @ A on every token
LORA_VARIANTS = (
    "use_dora",
    "use_rslora",
    "use_qalora",
    "use_bdlora",
    "fan_in_fan_out",
    "lora_bias",
    "rank_pattern",
    "alpha_pattern",
    "modules_to_save",
    "layer_replication",
    "trainable_token_indices",
    "target_parameters",
    "alora_invocation_tokens",
    "arrow_config",
    "monteclora_config",
)


HEADER_DTYPES = {mlx: stored for stored, mlx in MLX.items()}  # an MLX dtype -> its stored name


def header_dtype(array: mx.array) -> str:
    """An array's dtype as a safetensors header names it."""
    return HEADER_DTYPES.get(array.dtype, str(array.dtype))


@dataclass
class Weights:
    """Named tensors from one or more safetensors files, read lazily, each handed out once (`matrix`, `get`): what
    a loader took is no longer held here, so a tensor's memory goes as soon as the layer made from it is evaluated,
    and a model loading in another dtype than its files' is never in memory twice.

    A checkpoint saved by MLX may hold quantized matrices: a packed uint32 `<name>.weight` with `<name>.scales` and
    `<name>.biases`, the group size in its config's "quantization" (with per-layer entries). `get` returns any
    tensor in full precision, dequantizing it; `matrix` keeps a quantized one quantized."""

    tensors: dict[str, mx.array]
    source: str = ""
    quantization: Mapping[str, Any] | None = None  # the checkpoint's "quantization" config, if any
    taken: set[str] = field(default_factory=set, repr=False, compare=False)  # handed out already
    # each tensor's file and its name there, for those read from a safetensors file as they are stored
    paths: dict[str, tuple[Path, str]] = field(default_factory=dict, repr=False, compare=False)
    # how a tensor of another format is read row by row from its file (a GGUF build's: gguf.reader.host_rows)
    hosted: dict[str, Callable[[mx.Dtype], HostRows | None]] = field(default_factory=dict, repr=False, compare=False)
    # vocabulary tables (`table`) read from their files on the CPU; False for a format that multiplies by a whole one
    host_tables: bool = True

    @classmethod
    def from_dir(
        cls,
        directory: str | Path,
        files: Iterable[str] | None = None,
        quantization: Mapping[str, Any] | None = None,
    ) -> Weights:
        """The safetensors files of a model directory: `files`, else `weight_files(directory)`."""
        directory = Path(directory)
        names = list(weight_files(directory) if files is None else files)
        if not names:
            raise UnsupportedModel(f"{directory}: no safetensors weights (other formats are never loaded)")
        tensors: dict[str, mx.array] = {}
        paths: dict[str, tuple[Path, str]] = {}
        for name in names:
            if not name.endswith(".safetensors"):
                raise UnsupportedModel(f"{directory / name}: only safetensors files are loaded")
            try:
                loaded = mx.load(str(directory / name))
            except (OSError, ValueError, RuntimeError) as error:
                if out_of_memory(error):
                    raise as_decision_error(error) from None
                raise invalid_file(directory / name, f"cannot load safetensors ({error})") from None
            assert isinstance(loaded, dict)
            tensors.update(loaded)
            paths |= {key: (directory / name, key) for key in loaded}
        return cls(tensors, str(directory), quantization=quantization, paths=paths)

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def get(self, name: str, dtype: mx.Dtype | None = None) -> mx.array:
        """The tensor `name` in full precision (dequantized if the checkpoint stores it quantized), cast to `dtype`
        when given; UnsupportedModel naming the tensor when it is missing."""
        found = self.matrix(name, dtype)
        return found.dense(dtype) if isinstance(found, QuantizedMatrix) else found

    def _global_scale(self, name: str) -> tuple[list[str], mx.array | None]:
        """The tensor names of `name`'s global scale (at most one, as `packed` checks) and the scale itself in fp32:
        NVFP4's multiplier, one per tensor or per expert, finite and positive."""
        found = [k for k in global_scale_names(name) if k in self.tensors]
        if not found:
            return found, None
        scale = self.tensors[found[0]].astype(mx.float32)
        if scale.size == 1:
            scale = scale.reshape(())
        if not bool(mx.all(mx.isfinite(scale) & (scale > 0))):
            raise UnsupportedModel(f"{name}: global scale must be finite and positive")
        return found, scale

    def _packing(self, name: str) -> Quantization | None:
        """How the matrix `name` is stored (`quantization.packed`): None in full precision."""
        array = self.tensors[name]
        scales, biases = self.tensors.get(scales_of(name)), self.tensors.get(biases_of(name))
        return packed(
            name,
            header_dtype(array),
            tuple(array.shape),
            None if scales is None else (header_dtype(scales), tuple(scales.shape)),
            None if biases is None else (header_dtype(biases), tuple(biases.shape)),
            [
                (header_dtype(self.tensors[k]), tuple(self.tensors[k].shape))
                for k in global_scale_names(name)
                if k in self.tensors
            ],
            self.quantization,
        )

    def matrix(self, name: str, dtype: mx.Dtype | None = None) -> mx.array | QuantizedMatrix:
        """The tensor `name`, quantized as the checkpoint stores it (its scales and biases in `dtype`) or cast to
        `dtype`, taken out of these weights."""
        if name in self.taken:
            raise ValueError(f"{self.source}: {name!r} was read already (Weights hands each tensor out once)")
        if name not in self.tensors:
            raise UnsupportedModel(f"{self.source}: the checkpoint has no tensor {name!r}")
        array = self.tensors[name]
        scales, biases = self.tensors.get(scales_of(name)), self.tensors.get(biases_of(name))
        spec = self._packing(name)
        if spec is None:
            self.take(name)
            return array if dtype is None else array.astype(dtype)
        assert scales is not None
        scale_names, global_scale = self._global_scale(name)
        q = QuantizedMatrix(
            array,
            scales,
            biases,
            spec.bits,
            spec.group_size,
            mode=spec.mode,
            global_scale=global_scale,
            global_scale_is_amax=spec.global_scale_is_amax(scale_names[0] if scale_names else None),
        )
        for key in (name, scales_of(name), biases_of(name), *scale_names):
            if key in self.tensors:
                self.take(key)
        return q if dtype is None else q.astype(dtype)

    def host_rows(self, name: str, dtype: mx.Dtype) -> HostRows | None:
        """The matrix `name` as rows read from its file on the CPU in `dtype` (`HostRows`), packed as it is stored,
        taken out of these weights; None, leaving it here, when it or a part of it is in no safetensors file of its
        own and no other reader (`hosted`) either: a matrix a LoRA changed."""
        if name in self.hosted and name in self.tensors:
            found = self.hosted[name](dtype)
            if found is not None:
                self.matrix(name)  # checked as a load checks it, and taken: its rows come from the file
            return found
        parts = [name, *(p for p in (scales_of(name), biases_of(name)) if p in self.tensors)]
        if name not in self.tensors or any(p not in self.paths for p in parts):
            return None
        if any(k in self.tensors for k in global_scale_names(name)):
            return None  # an NVFP4 table's multiplier: loaded whole, which applies it
        spec = self._packing(name)  # checked as a load checks it
        packing = None if spec is None else (spec.bits, spec.group_size, spec.mode)
        table = HostRows.from_safetensors([self.paths[p] for p in parts], dtype, packing)
        if table is not None:
            for p in parts:  # taken: its rows come from the file
                self.take(p)
        return table

    def table(self, name: str, dtype: mx.Dtype) -> Table:
        """A vocabulary table, the embedding or the output matrix, of which a decision model reads a few hundred rows
        a prompt: read from its file on the CPU (`host_rows`) with `host_tables` where it can be, else as `matrix`
        loads it. Kettei Core keeps 2.6 GB of its 10.1 GB at bf16 off the GPU so, with its answers unchanged."""
        found = self.host_rows(name, dtype) if self.host_tables else None
        return found if found is not None else self.matrix(name, dtype)

    def take(self, name: str) -> mx.array:
        self.taken.add(name)
        return self.tensors.pop(name)

    def text_root(self) -> str:
        """The root under which the text model's embedding and layers are named."""
        for root in TEXT_ROOTS:
            if root + EMBEDDING in self.tensors and any(k.startswith(f"{root}layers.0.") for k in self.tensors):
                return root
        raise UnsupportedModel(f"{self.source}: no text model (embed_tokens and layers) found in the checkpoint")

    def with_lora(self, adapter: Weights, config: Mapping[str, Any]) -> Weights:
        """These weights with a PEFT LoRA adapter merged in: W + (alpha / r) * B @ A in fp32, for every adapted
        matrix. Variants whose model is not that formula (DoRA, rsLoRA, QA-LoRA, activated LoRA, per-module ranks,
        replicated layers, trained biases, tokens or extra modules) are refused, and so is an adapter tensor that
        names no matrix of these weights, or a quantized one (merge into the full-precision base instead). The
        merged weights keep this checkpoint's quantization config, which its untouched quantized matrices need, and
        the files of the tensors no LoRA matrix changed, so their tables can still be read from them (`table`)."""
        refused = [k for k in LORA_VARIANTS if config.get(k)]
        if config.get("bias", "none") != "none":
            refused.append("bias")
        if refused or config.get("peft_type", "LORA") != "LORA":
            raise UnsupportedModel(
                f"{adapter.source}: this LoRA adapter uses {refused or config.get('peft_type')}, "
                "which ippon does not merge"
            )
        scale = float(config["lora_alpha"]) / float(config["r"])
        root = self.text_root()
        merged = dict(self.tensors)
        pairs: dict[str, dict[str, mx.array]] = {}
        for name, array in adapter.tensors.items():
            module, _, kind = name.rpartition(".lora_")
            if kind not in ("A.weight", "B.weight") or not module.startswith("base_model.model."):
                raise UnsupportedModel(f"{adapter.source}: unexpected adapter tensor {name!r}")
            pairs.setdefault(module.removeprefix("base_model.model."), {})[kind[0]] = array
        for module, ab in pairs.items():
            target = next((t for t in (root + module + ".weight", module + ".weight") if t in self.tensors), None)
            if target is None or set(ab) != {"A", "B"}:
                raise UnsupportedModel(
                    f"{adapter.source}: the adapter's {module!r} matches no matrix of the base model"
                )
            if scales_of(target) in self.tensors:
                raise UnsupportedModel(
                    f"{adapter.source}: {module!r} is quantized in {self.source}; merge the LoRA into the base model "
                    "at full precision"
                )
            base = self.tensors[target]
            delta = ab["B"].astype(mx.float32) @ ab["A"].astype(mx.float32)
            if delta.shape != base.shape:
                raise UnsupportedModel(f"{adapter.source}: {module!r} is {delta.shape}, the base matrix {base.shape}")
            merged[target] = base.astype(mx.float32) + scale * delta
        untouched = {name: path for name, path in self.paths.items() if merged.get(name) is self.tensors.get(name)}
        return Weights(merged, f"{self.source} + {adapter.source}", quantization=self.quantization, paths=untouched)
