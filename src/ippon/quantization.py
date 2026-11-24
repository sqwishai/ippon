"""The precisions ippon runs and the quantized formats it reads, without importing MLX: one table that loading,
preflight, `ippon check`, setup and the names a model is asked for by all read.

A precision (`PRECISIONS`) is what `precision=` names: "auto" (computed in fp16 where a model's activations fit in it,
else bf16, as precision.py measures), a full-precision dtype (bf16, fp16, fp32, or their long names), or q8, q6 and q4,
which quantize a full-precision build's weight matrices here, to MLX's affine groups. Below 8 bits is low-bit: loaded
only when asked for, with a warning of its drift (drift.py). precision.py maps the dtypes to MLX's.

A stored quantized matrix is a packed U32 weight with companions named after it (`scales_of`, `biases_of`), and for
NVFP4 an optional global scale (`global_scale_names`), read and checked by `Quantization.read` and `packed`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .errors import InvalidRequest, UnsupportedModel
from .safetensors import COMPUTED, FLOATS, Tensor

Dtype = Literal["bfloat16", "float16", "float32"]
FLOAT_DTYPES: dict[str, Dtype] = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
FULL = tuple(FLOAT_DTYPES)  # the full-precision names, also the tags of a creator's checkpoint (names.TAGS)
SHORT: dict[Dtype, str] = {dtype: name for name, dtype in FLOAT_DTYPES.items()}  # a dtype's short name ("fp16")
LOW_BIT_BELOW = 8  # fewer bits than this drift further from the creator's answers: loaded only when asked for
QUANTIZE_TO = (8, 6, 4)  # the widths ippon quantizes a full-precision build's weight matrices to as it loads
QUANTIZED = tuple(f"q{bits}" for bits in QUANTIZE_TO)  # their precisions' names
AFFINE_BITS = (4, 5, 6, 8)  # the affine widths ippon loads as stored
GROUP_SIZE = 64  # the affine group ippon quantizes to, and MLX's for a checkpoint that names none
GROUP_SIZES = (GROUP_SIZE, 32)  # a matrix quantized here takes the first that divides its width (`group_for`)
MODES = {"affine": (None, GROUP_SIZE), "mxfp8": (8, 32), "nvfp4": (4, 16), "mxfp4": (4, 32)}  # fixed bits, group
# An NVFP4 global scale's two spellings: MLX's is the amax of the group scales, NVIDIA ModelOpt's a multiplier
GLOBAL_SCALE, SCALE_2 = ".global_scale", "_scale_2"


def low_bit(bits: int | None) -> bool:
    """Whether `bits` (None: full precision) is below 8: loaded only when asked for, with a warning (drift.py)."""
    return bits is not None and bits < LOW_BIT_BELOW


def form_of(mode: str, bits: int) -> str:
    """How a warning or setup names a quantized format: "4-bit" for affine, else the mode ("NVFP4")."""
    return f"{bits}-bit" if mode == "affine" else mode.upper()


def group_for(width: int) -> int | None:
    """The group a matrix `width` columns wide is quantized in here (GROUP_SIZES); None for a width of neither,
    which stays as it is."""
    return next((g for g in GROUP_SIZES if width % g == 0), None)


@dataclass(frozen=True)
class Precision:
    """What a `precision=` name loads: the compute dtype (None: measured, as auto measures it) and the bits the weight
    matrices are quantized to here (None: as stored)."""

    name: str
    dtype: Dtype | None = None
    bits: int | None = None

    @property
    def changes_weights(self) -> bool:
        """The weights quantized here: another model, whose answers `ippon verify` checks."""
        return self.bits is not None

    @property
    def low_bit(self) -> bool:
        """Weights quantized here below 8 bits, which drift further from the creator's answers (drift.py)."""
        return low_bit(self.bits)

    @property
    def width(self) -> int:
        """Bytes per value computed: fp32's 4, else 2 (a measured dtype is fp16 or bf16)."""
        return 4 if self.dtype == "float32" else 2

    @property
    def form(self) -> str:
        """How setup and a warning name it: "full", or "8-bit", "6-bit", "4-bit"."""
        return "full" if self.bits is None else form_of("affine", self.bits)

    @property
    def said(self) -> str:
        """How a message names it: auto is "full precision", the others their names."""
        return "full precision" if self.dtype is None and self.bits is None else self.name


PRECISIONS: dict[str, Precision] = {
    p.name: p
    for p in (
        Precision("auto"),  # fp16 when the model's activations fit in it with room to spare, else bf16
        *(Precision(name, dtype) for name, dtype in FLOAT_DTYPES.items()),
        *(Precision(dtype, dtype) for dtype in FLOAT_DTYPES.values()),
        # weight matrices quantized to these bits, computed in the dtype auto chooses
        *(Precision(name, bits=bits) for name, bits in zip(QUANTIZED, QUANTIZE_TO, strict=True)),
    )
}
NAMES = ("auto", *FULL, *QUANTIZED)  # the names messages list (PRECISIONS also takes the dtypes' long names)
OFFERED = ("fp16", "q8")  # what a model too large at its precision is offered: the smallest full precision, and 8-bit


def listing(items: Sequence[str]) -> str:
    """`items` as a message lists them: "a, b or c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def format_policy(name: str, alternative: str) -> str:
    """The precisions ippon runs, and what to use instead of `name`."""
    return (
        f"{name}: ippon runs decision models at full precision or 8-bit, and at 4 to 6 bits or fp4 only when asked, "
        f"because lower precision changes their answers; use {alternative}"
    )


def unsupported_format(name: str, alternative: str = "an 8-bit or full-precision build") -> UnsupportedModel:
    """A build stored in a format ippon doesn't run, shared by all loading paths."""
    return UnsupportedModel(format_policy(name, alternative))


def get(name: str) -> Precision:
    """Resolve a precision before opening a model or allocating weights."""
    if name not in PRECISIONS:
        # the caller's option, not the model: invalid_request, so an agent fixes the call instead of the model
        raise InvalidRequest(format_policy(f"precision {name!r}", "--precision " + listing(NAMES)))
    return PRECISIONS[name]


def scales_of(name: str) -> str:
    """The name of the matrix `name`'s group scales."""
    return name.removesuffix(".weight") + ".scales"


def biases_of(name: str) -> str:
    """The name of the matrix `name`'s affine biases."""
    return name.removesuffix(".weight") + ".biases"


def matrix_of(name: str) -> str:
    """The matrix a tensor belongs to, without its .weight, .scales or .biases."""
    return name.removesuffix(".scales").removesuffix(".biases").removesuffix(".weight")


def global_scale_names(name: str) -> tuple[str, ...]:
    """Global-scale companion spellings, without duplicates for tensors not named *.weight."""
    base = name.removesuffix(".weight")
    return tuple(dict.fromkeys((base + GLOBAL_SCALE, name + GLOBAL_SCALE, name + SCALE_2, base + SCALE_2)))


def is_global_scale(name: str) -> bool:
    return name.endswith((GLOBAL_SCALE, SCALE_2))


@dataclass(frozen=True)
class Quantization:
    mode: str
    bits: int
    group_size: int
    mlx_global_scale: bool = False
    scale_convention: str | None = None

    def global_scale_is_amax(self, found: str | None) -> bool:
        """Whether the global scale stored as `found` (None: none) is the amax of the group scales, as MLX's is, rather
        than a multiplier: as the metadata says, else by its spelling (GLOBAL_SCALE an amax, SCALE_2 a multiplier)."""
        if self.scale_convention is not None:
            return self.scale_convention == "mlx"
        return self.mlx_global_scale or (found is not None and not found.endswith(SCALE_2))

    @classmethod
    def read(
        cls, config: Mapping[str, Any] | None, name: str, weight: tuple[int, ...], scales: tuple[int, ...]
    ) -> Quantization:
        """Resolve one packed tensor's metadata and check its physical dimensions."""
        config = {} if config is None else config
        if not isinstance(config, Mapping):
            raise UnsupportedModel(f"{name}: quantization metadata must be an object")
        base = name.removesuffix(".weight")
        own = config.get(base, config.get(name, {}))
        if own is False:
            raise UnsupportedModel(f"{name}: marked unquantized but has quantization scales")
        if own is True:  # quantized with the file's global settings
            own = {}
        if not isinstance(own, Mapping):
            raise UnsupportedModel(f"{name}: per-tensor quantization metadata must be an object")
        q = {
            k: config[k]
            for k in ("mode", "bits", "group_size", "global_scale", "global_scale_convention")
            if k in config
        }
        q.update(own)
        mode = q.get("mode", "affine")
        if not isinstance(mode, str) or mode not in MODES:
            raise UnsupportedModel(f"{name}: unsupported quantization mode {mode!r}")
        fixed_bits, default_group = MODES[mode]
        group = q.get("group_size", default_group)
        if type(group) is not int or group not in ((32, 64, 128) if mode == "affine" else (default_group,)):
            raise UnsupportedModel(f"{name}: invalid {mode} group_size {group!r}")
        if len(weight) < 2 or len(weight) != len(scales) or weight[:-1] != scales[:-1] or scales[-1] <= 0:
            raise UnsupportedModel(f"{name}: packed weight {weight} and scales {scales} do not match")
        bits = q.get("bits", fixed_bits or weight[-1] * 32 // (scales[-1] * group))
        if mode == "affine" and type(bits) is int and bits in (2, 3):
            raise unsupported_format(f"{name}: {bits}-bit {mode}")
        if (
            type(bits) is not int
            or bits not in (AFFINE_BITS if mode == "affine" else (fixed_bits,))
            or weight[-1] * 32 != scales[-1] * group * bits
        ):
            raise UnsupportedModel(f"{name}: invalid {mode} bits or packed dimensions ({bits!r}, {weight}, {scales})")
        native_scale = q.get("global_scale", False)
        if type(native_scale) is not bool or (native_scale and mode != "nvfp4"):
            raise UnsupportedModel(f"{name}: global_scale metadata must be an NVFP4 boolean")
        convention = q.get("global_scale_convention")
        if convention is not None and convention not in ("mlx", "multiplier"):
            raise UnsupportedModel(f"{name}: unsupported global_scale_convention {convention!r}")
        return cls(mode, bits, group, native_scale, convention)


def packed(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    scales: tuple[str, tuple[int, ...]] | None,
    biases: tuple[str, tuple[int, ...]] | None,
    global_scales: Sequence[tuple[str, tuple[int, ...]]],
    config: Mapping[str, Any] | None,
) -> Quantization | None:
    """A weight matrix's quantization (None: stored in full precision), from its dtype and shape and those of its
    scales, biases and NVFP4 global scales (safetensors' dtype names), or UnsupportedModel for a layout ippon doesn't
    read: checked from a checkpoint's headers before it loads and from the arrays a loaded one holds."""
    if scales is None:
        if dtype == "U32":
            raise UnsupportedModel(f"{name}: packed U32 weights have no quantization scales")
        if len(shape) >= 2 and dtype not in FLOATS:
            raise UnsupportedModel(f"{name}: unsupported packed layout {dtype}; use an MLX packed checkpoint")
        return None
    if dtype != "U32":
        raise UnsupportedModel(f"{name}: a matrix with quantization scales must be packed U32")
    spec = Quantization.read(config, name, shape, scales[1])
    if spec.mode == "affine":
        if biases is None or biases != scales or scales[0] not in FLOATS:
            raise UnsupportedModel(f"{name}: affine scales need matching floating-point biases")
    elif scales[0] != "U8" or biases is not None:
        raise UnsupportedModel(f"{name}: {spec.mode} requires encoded U8 scales and no biases")
    if spec.mlx_global_scale and not global_scales:
        raise UnsupportedModel(f"{name}: global_scale metadata requires a scale tensor")
    if len(global_scales) > 1:
        raise UnsupportedModel(f"{name}: ambiguous global scales")
    if global_scales:
        kind, size = global_scales[0]
        if spec.mode != "nvfp4" or kind not in COMPUTED or size not in ((), (1,), shape[:-2]):
            raise UnsupportedModel(f"{name}: global scale must be an NVFP4 multiplier per tensor or expert")
    return spec


def inspect(found: Sequence[Tensor], config: Mapping[str, Any] | None) -> dict[str, int]:
    """Validate packed tensor headers and count matrices by format, without loading weights."""
    tensors = {t.name: t for t in found}
    modes: dict[str, int] = {}

    def header(name: str) -> tuple[str, tuple[int, ...]] | None:
        t = tensors.get(name)
        return None if t is None else (t.dtype, t.shape)

    for t in found:
        scales = header(scales_of(t.name))
        if scales is None and not t.name.endswith(".weight"):
            continue  # a buffer (position ids, say), never multiplied
        found_scales = [h for k in global_scale_names(t.name) if (h := header(k)) is not None]
        spec = packed(t.name, t.dtype, t.shape, scales, header(biases_of(t.name)), found_scales, config)
        if spec is not None:
            modes[spec.mode] = modes.get(spec.mode, 0) + 1
    return modes
