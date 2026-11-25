"""Chip profiles: which code path each operation takes on M1 to M5.

A profile is chosen once, at load, from the chip: m5 has neural accelerators in its GPU cores, which MLX's own
matrix multiplies use and which ippon's own matmul kernels target; m1 to m4 run MLX's matmuls (on m1, m2 and m4 an
MXFP8 one unpacked first on long prompts, choices.py) and the paths that need no accelerators. A profile changes which
kernels compute a model, never what a model computes.

A lower profile can be forced, to run an older chip's paths on a newer Mac (testing) or to rule a path out:

    Decider.load(model, profile="m4")      IPPON_PROFILE=m4      ippon run --profile m4 ...

A profile above the chip is refused, since its kernels would not run. Each operation's path can also be set on its
own, over the profile's (`Switches` lists them; "mlx" is always MLX's own operation, and a switch's first path the
reference):

    Decider.load(model, switches="conv=mlx,gemm=mlx")   IPPON_SWITCHES=conv=mlx   ippon run --switch conv=mlx

attention=mlx remains the default. attention=bounded is an explicit memory mode for eligible large unmasked
head-size-256 calls; it evaluates query blocks separately and runs all layers eagerly even when compile=layers
is requested. The full-key softmax remains MLX's, but query-matmul shapes can change rounding.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, get_args, get_type_hints

from .device import Device, detect
from .errors import UnsupportedProfile

PROFILES = ("m1", "m2", "m3", "m4", "m5")
# The MLX ippon's kernel facts were measured on: choices.py's tiles, which head-size-256 attention calls MLX fuses
# (kernels/attention.mlx_fuses) and which Gated DeltaNet head counts its kernels take (arch/ops.FUSED_DELTA), the
# "mlx" scripts/measure_kernels.py records. They move with MLX's own kernels, so under an MLX of another minor
# version (0.33 against 0.32.x) MLX's own operations run in their place (`Profile.mlx_fitted`). Its patch releases
# keep them, since pyproject takes any 0.32.x.
FITTED_MLX = "0.32.4"


def _minor(text: str) -> str:
    """A version's major.minor: "0.32" of "0.32.4"."""
    return ".".join(text.split(".")[:2])


@dataclass(frozen=True)
class Switches:
    """The code path of each operation, the profile's own as defaults: only paths that exist. Each field's first
    path is the reference, MLX's own ("mlx", or "off"). Every Apple GPU runs the fused kernels, and its matmuls as
    choices.py picks them for each shape: on m5 the neural accelerators' tile library, on m1 to m4 MLX's own
    matmuls, with m1, m2 and m4's MXFP8 matrices unpacked on long prompts."""

    add_norm: Literal["mlx", "kernel"] = "kernel"  # residual add + norm fused (kernels/norm.py)
    conv: Literal["mlx", "kernel"] = "kernel"  # the hybrids' short convolutions fused (kernels/conv.py)
    # MLX's own matmuls, or the kernels choices.py picks for each shape: kernels/gemm.py's and kernels/qgemm.py's
    # variants where they fill the GPU
    gemm: Literal["mlx", "kernel"] = "kernel"
    gated_delta: Literal["mlx", "half"] = "mlx"  # its q, k, v in fp32, or in the compute dtype
    # Gated DeltaNet's q, k, v (normalized, fp32) in one kernel. A/B on a 2B Qwen3.5 hybrid (M5 Pro): 2-8% faster at
    # 20 to 2,000 words; exact on the M4's kernels
    delta_inputs: Literal["mlx", "kernel"] = "kernel"
    attention: Literal["mlx", "bounded"] = "mlx"  # MLX's attention, or MLX's in evaluated query blocks (a memory mode)
    # evaluation started every few layers while the rest of the pass is built. A/B (M5 Pro): a 2B Qwen3.5 hybrid 3-8%
    # faster at 20-1,000 words, Qwen3 0.6B 3-15%, Gemma 3 270M 6-21%; d1-omni no difference. Models of fewer than
    # PIPELINE_MIN_LAYERS layers don't hand over (a 6-layer encoder's whole request takes 2.5 ms)
    pipeline: Literal["off", "layers", "early"] = "layers"
    # each layer one replayed mx.compile trace per kind and pass shape (arch/common.py). Main-thread CPU per request
    # against "off" (M5 Pro, 100 words, 1 and 3 questions): d1-omni-600M 31% and 19% less, a ModernBERT-large encoder
    # 20% and 16%, a Qwen3.5-0.8B 33% and 32%, d1-3B 32% and 29%; the same answers, bit for bit
    compile: Literal["off", "layers"] = "layers"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


PATHS: dict[str, tuple[str, ...]] = {name: get_args(kind) for name, kind in get_type_hints(Switches).items()}


def overridden(base: Switches, text: str | None) -> Switches:
    """`base` with IPPON_SWITCHES-style overrides ("add_norm=mlx,conv=mlx") applied; UnsupportedProfile for an
    unknown switch or path."""
    changes: dict[str, str] = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if value not in PATHS.get(key, ()):
            known = "; ".join(f"{k}={'|'.join(v)}" for k, v in sorted(PATHS.items()))
            raise UnsupportedProfile(f"unknown switch {item.strip()!r}; switches are {known}")
        changes[key] = value
    return replace(base, **changes)  # type: ignore[arg-type]  # each value checked against its field's paths above


def installed_mlx() -> str | None:
    """The MLX this process runs, None when it isn't installed."""
    try:
        return version("mlx")
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class Profile:
    name: str
    detected: str | None  # the chip's own profile; None off Apple silicon
    device: Device | None = None
    switches: Switches = Switches()
    # the switches set over the profile's own, and what set each: IPPON_SWITCHES, or `switched` (--switch, switches=)
    sources: Mapping[str, Literal["IPPON_SWITCHES", "--switch"]] = field(default_factory=dict)
    mlx: str | None = FITTED_MLX  # the MLX this process runs (`resolve` reads it)

    @property
    def forced(self) -> bool:
        return self.detected is not None and self.name != self.detected

    @property
    def emulated(self) -> bool:
        """True when MLX also runs this profile's chip's kernels (IPPON_PROFILE set before MLX started), not
        only ippon's own paths."""
        return self.device is not None and f"m{self.device.emulated_generation}" == self.name

    @property
    def mlx_fitted(self) -> bool:
        """Whether the MLX this process runs is of the minor version ippon's kernel facts were measured on
        (FITTED_MLX); with any other, MLX's own operations run in their place."""
        return self.mlx is not None and _minor(self.mlx) == _minor(FITTED_MLX)

    @property
    def mlx_neural_accelerators(self) -> bool:
        """Whether MLX runs its own kernels as a chip with neural accelerators (`Device.mlx_neural_accelerators`)."""
        return self.device is not None and self.device.mlx_neural_accelerators


def native(device: Device) -> str | None:
    """The profile a Mac runs unless told otherwise: its chip generation's, except an M5's before macOS 26.2, whose
    neural accelerators MLX can't reach, which runs the M4's paths; None off Apple silicon."""
    if not device.generation:
        return None
    name = f"m{min(device.generation, 5)}"
    return "m4" if name == "m5" and not device.neural_accelerators else name


def resolve(override: str | None = None, device: Device | None = None, switched: str | None = None) -> Profile:
    """The profile to run: `override`, else IPPON_PROFILE, else the Mac's own (`native`); its switches with
    IPPON_SWITCHES and then `switched` ("add_norm=mlx,gemm=mlx") applied. UnsupportedProfile for an unknown name,
    one above what the Mac runs, or an unknown switch."""
    device = device or detect()
    detected = native(device)
    name = (override or os.environ.get("IPPON_PROFILE") or "").strip().lower() or None
    if name is None:
        name = detected or "m1"
    elif name not in PROFILES:
        raise UnsupportedProfile(f"unknown profile {name!r}; profiles are {', '.join(PROFILES)}")
    elif name == "m5" and not device.neural_accelerators and (device.generation is None or device.generation >= 5):
        # an M5 before macOS 26.2, or no Apple GPU at all; an older chip gets the message below
        raise UnsupportedProfile("profile m5 needs the neural accelerators of an M5 and macOS 26.2 or later")
    elif detected is not None and PROFILES.index(name) > PROFILES.index(detected):
        raise UnsupportedProfile(
            f"profile {name} needs an {name.upper()}-generation chip or later; this Mac is an {device.name or detected}"
        )
    base = Switches()
    environment = overridden(base, os.environ.get("IPPON_SWITCHES"))
    chosen = overridden(environment, switched)
    sources: dict[str, Literal["IPPON_SWITCHES", "--switch"]] = {}
    given, defaults, picked = environment.as_dict(), base.as_dict(), chosen.as_dict()
    for key, value in picked.items():
        if value != given[key]:
            sources[key] = "--switch"
        elif value != defaults[key]:
            sources[key] = "IPPON_SWITCHES"
    return Profile(name, detected, device, chosen, sources, installed_mlx())
