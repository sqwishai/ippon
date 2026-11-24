"""The machine ippon runs on: its chip, its memory and the fast paths it has.

The kernels choose between paths once, at load, from this description: the M5 generation has neural
accelerators in every GPU core, which MLX's matrix multiplies use from macOS 26.2; M1 to M4 run MLX's
other kernels. Nothing here changes a model's numbers, only which kernels compute them.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

# Apple GPU families: "applegpu_g17s" is generation 17 - 12 = 5 (M1 is g13), its letter the part: g, s, c and d the
# base, Pro, Max and Ultra chips, p a phone's. Families with neural accelerators in their cores: g17 and later (the M5
# generation), and g18 and later for the phone parts, from macOS 26.2 on, as MLX's own check has it.
_ARCHITECTURE = re.compile(r"applegpu_g(\d+)([a-z]?)")
_FAMILY_OFFSET = 12
_PARTS = {"base": "g", "Pro": "s", "Max": "c", "Ultra": "d"}
_TIERS = {letter: tier for tier, letter in _PARTS.items()}
_CHIP = re.compile(r"Apple M(\d+)(?: (Pro|Max|Ultra))?")
_NEURAL_ACCELERATOR_MACOS = (26, 2)


def _generation(architecture: str | None) -> int | None:
    """1 for M1 ... 5 for M5 from a GPU family; None for no family or a phone part's."""
    m = _ARCHITECTURE.fullmatch(architecture or "")
    return int(m[1]) - _FAMILY_OFFSET if m and m[2] != "p" else None


def architecture_of(generation: int, tier: str | None = None) -> str:
    """The GPU family of an M-`generation` chip of `tier` ("Pro", "Max", "Ultra"; None the base part):
    applegpu_g17s for an M5 Pro."""
    return f"applegpu_g{generation + _FAMILY_OFFSET}{_PARTS[tier or 'base']}"


def chip(name: str | None) -> tuple[int, str | None] | None:
    """A Mac chip's generation and tier by its name: (5, "Pro") for "Apple M5 Pro"; None for any other name."""
    m = _CHIP.fullmatch(name or "")
    return (int(m[1]), m[2]) if m else None


def chip_architecture(name: str | None) -> str | None:
    """The GPU family of a Mac chip by its name: "Apple M5 Pro" is applegpu_g17s."""
    found = chip(name)
    return architecture_of(*found) if found else None


@dataclass(frozen=True)
class Device:
    name: str | None  # "Apple M5 Pro"
    architecture: str | None  # "applegpu_g17s"
    memory_bytes: int | None
    working_set_bytes: int | None  # what the GPU may keep resident; macOS's default is about 3/4 of memory
    macos: tuple[int, ...]
    gpu: bool
    emulated_architecture: str | None = None  # the older family MLX was told to take (MLX_METAL_GPU_ARCH), if any
    gpu_cores: int | None = None  # the GPU's cores (20 on a 20-core M5 Pro): how many threadgroups run at once

    @property
    def generation(self) -> int | None:
        """1 for M1 ... 5 for M5, from the GPU family (applegpu_g13 is M1); None off Apple silicon or for phone
        parts."""
        return _generation(self.architecture)

    @property
    def emulated_generation(self) -> int | None:
        """The generation whose MLX kernels run, when MLX was told to take an older family than the chip's."""
        return _generation(self.emulated_architecture)

    @property
    def tier(self) -> str | None:
        """ "base", "Pro", "Max" or "Ultra", from the family's suffix letter (detect() takes it from the chip's
        name: MLX reports an M5 Max as applegpu_g17s, like the M5 Pro)."""
        m = _ARCHITECTURE.fullmatch(self.architecture or "")
        return _TIERS.get(m[2]) if m else None

    @property
    def neural_accelerators(self) -> bool:
        m = _ARCHITECTURE.fullmatch(self.architecture or "")
        if not (self.gpu and m) or self.macos < _NEURAL_ACCELERATOR_MACOS:
            return False
        return int(m[1]) >= (18 if m[2] == "p" else 17)

    @property
    def mlx_neural_accelerators(self) -> bool:
        """Whether MLX runs its kernels as a chip with neural accelerators: the family it takes (the chip's, or the
        older one `_emulate` told it to take) is the M5's or later, on macOS 26.2 or later."""
        return replace(self, architecture=self.emulated_architecture or self.architecture).neural_accelerators


def macos_version(text: str | None = None) -> tuple[int, ...]:
    """The macOS version as a tuple, (0,) off macOS."""
    text = platform.mac_ver()[0] if text is None else text
    return tuple(int(part) for part in text.split(".") if part.isdigit()) or (0,)


def max_body(memory: int | None = None) -> int:
    """The most bytes of a request body a server takes by default: a 32nd of this Mac's memory (`memory`, else
    the OS's count), from 64 MiB to 1.5 GiB, since a video sent as base64 is held about four times over while it
    is read (the body, its JSON string, the decoded bytes, their frames): 256 MiB on an 8 GB Mac, 1 GiB on 32 GB."""
    found = memory if memory is not None else physical_memory()
    return min(max((found or 8 << 30) // 32, 64 << 20), 3 << 29)


def physical_memory() -> int | None:
    """The machine's memory in bytes, from the OS (for a machine without an Apple GPU)."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


@cache
def gpu_cores() -> int | None:
    """The GPU's core count, as IOKit reports it (`gpu-core-count`; MLX's device info doesn't): about 20 ms, once."""
    try:
        text = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-k", "gpu-core-count"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'"gpu-core-count" = (\d+)', text)
    return int(m[1]) if m else None


def detect() -> Device:
    """This machine, as MLX's Metal device reports it."""
    import mlx.core as mx  # here: `doctor` reports an MLX that doesn't load

    gpu = mx.metal.is_available()
    info: dict[str, Any] = dict(mx.device_info(mx.gpu)) if gpu else {}  # the GPU's, whatever the default device
    reported = info.get("architecture")  # what MLX runs as: the chip's family, or the one it was told to take
    real = chip_architecture(info.get("device_name")) or reported
    # Only an older generation is emulation: an M5 Max reports applegpu_g17s, the M5 Pro's suffix (MLX #3791).
    return Device(
        name=info.get("device_name"),
        architecture=real,
        emulated_architecture=reported if reported and real and _generation(reported) != _generation(real) else None,
        memory_bytes=info.get("memory_size") or physical_memory(),
        working_set_bytes=info.get("max_recommended_working_set_size"),
        macos=macos_version(),
        gpu=gpu,
        gpu_cores=gpu_cores() if gpu else None,
    )
