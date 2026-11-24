"""Running an older chip's MLX kernels on a newer Mac, for testing: `IPPON_PROFILE=m4 python ...`.

MLX picks some kernels by the GPU family it detects (the M5's neural-accelerator matrix multiplies, for one), and its
MLX_METAL_GPU_ARCH environment variable, read when MLX starts, makes it take another family's. The variable is
undocumented, so ippon sets it only when asked through IPPON_PROFILE, only for a profile below the chip, only if it
isn't set already and only before MLX is imported: neither this module nor device.py, which holds the chip facts,
imports anything that imports MLX. A profile chosen in code (`profile="m4"`, `--profile m4`) switches ippon's own paths,
while MLX keeps the chip's kernels; `Profile.emulated` says which happened.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from .device import architecture_of, chip

VARIABLE = "MLX_METAL_GPU_ARCH"


def architecture_for(profile: str, name: str | None) -> str | None:
    """The family to have MLX take for `profile` ("m1".."m4") on the chip `name`d; None unless the profile is
    below it."""
    found = chip(name)
    wanted = re.fullmatch(r"m([1-4])", profile.strip().lower())
    if found is None or wanted is None or int(wanted[1]) >= found[0]:
        return None
    return architecture_of(int(wanted[1]), found[1])


def chip_name() -> str | None:
    """This Mac's chip as sysctl names it ("Apple M5 Pro"); None when it can't be read."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def apply() -> str | None:
    """Set MLX_METAL_GPU_ARCH for IPPON_PROFILE when that asks for an older chip; the family set, or None."""
    profile = os.environ.get("IPPON_PROFILE", "")
    if not re.fullmatch(r"\s*[mM][1-4]\s*", profile) or VARIABLE in os.environ or "mlx.core" in sys.modules:
        return None
    family = architecture_for(profile, chip_name())
    if family is not None:
        os.environ[VARIABLE] = family
    return family


APPLIED = apply()  # on import, which the package does before anything else
