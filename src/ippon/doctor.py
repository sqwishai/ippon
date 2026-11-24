"""`ippon doctor`: what stands between this Mac and ippon's full speed, and what to change.

Each check is ok, warn (slower than it could be) or fail (won't run as intended), with the fix. The checks take the
machine as data (`Device`, MLX's version), so each is tested without the machine it describes.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, requires, version
from typing import Literal

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from . import native as native_library
from . import terminal
from .admission import budget
from .device import Device, detect
from .errors import RuntimeUnavailable, UnsupportedProfile
from .profiles import Profile, native, resolve

Status = Literal["ok", "warn", "fail"]
# each status in the accent of what it means: settled, to look at, in the way (brand.ACCENTS)
TONES: dict[Status, str] = {"ok": "mint", "warn": "amber", "fail": "red"}


@dataclass(frozen=True)
class Check:
    status: Status
    what: str
    fix: str = ""


def mlx_range() -> SpecifierSet | None:
    """The MLX versions this ippon is tested with, as its package requires them (pyproject.toml); None when the
    installed package doesn't say."""
    try:
        return next(r.specifier for r in map(Requirement, requires("ippon") or ()) if r.name == "mlx" and r.specifier)
    except (PackageNotFoundError, StopIteration):
        return None


def checks(
    device: Device,
    profile: Profile,
    mlx_version: str | None,
    broken: Mapping[str, str],
) -> list[Check]:
    out: list[Check] = []
    if not device.gpu:
        out.append(Check("fail", "no Apple GPU: MLX runs on the CPU, many times slower", "run on an Apple-silicon Mac"))
    else:
        out.append(Check("ok", f"{device.name} ({device.architecture}), profile {profile.name}"))
    if device.generation == 5 and not device.neural_accelerators:
        out.append(
            Check(
                "warn",
                f"macOS {'.'.join(map(str, device.macos))}: the M5's neural accelerators are unused",
                "update to macOS 26.2 or later",
            )
        )
    elif device.neural_accelerators:
        out.append(Check("ok", "the GPU's neural accelerators are used"))
    if profile.forced:
        kernels = "ippon's and MLX's" if profile.emulated else "ippon's (MLX keeps this chip's)"
        out.append(
            Check(
                "warn",
                f"profile {profile.name} is forced on an {profile.detected} chip: {kernels} kernels for an older chip",
                "unset IPPON_PROFILE and drop profile= / --profile for this chip's paths",
            )
        )
    tested = mlx_range()
    if mlx_version is None:
        out.append(Check("fail", "MLX is not installed", "pip install 'ippon' (it depends on mlx)"))
    elif tested is not None and not tested.contains(mlx_version, prereleases=True):
        out.append(Check("warn", f"MLX {mlx_version} is untested here", f"pip install 'mlx{tested}'"))
    else:
        out.append(Check("ok", f"MLX {mlx_version}"))
    if device.working_set_bytes:
        params = budget(device.working_set_bytes) / 2 / 1e9  # what check and pull admit, at 2 bytes a value
        out.append(
            Check("ok", f"GPU working set {device.working_set_bytes / 2**30:.0f} GiB: up to ~{params:.0f}B in bf16")
        )
    for name, error in sorted(broken.items()):
        out.append(Check("fail", f"adapter {name!r} failed to load: {error}", "update or uninstall its package"))
    return out


def run() -> list[Check]:
    """The checks on this Mac."""
    first: list[Check] = []
    try:
        device = detect()
    except Exception as error:  # MLX doesn't import, or can't open the GPU: nothing else can be checked
        return [mlx_failure(error)]
    from .registry import broken  # the adapters import MLX

    try:
        profile = resolve(device=device)
    except UnsupportedProfile as error:  # IPPON_PROFILE asks for one this Mac can't run: this Mac's own
        first.append(Check("fail", str(error), "unset IPPON_PROFILE or set it to this chip's or a lower one"))
        profile = resolve(native(device) or "m1", device=device)
    try:
        mlx_version: str | None = version("mlx")
    except PackageNotFoundError:
        mlx_version = None
    try:
        native_library.check()
    except RuntimeUnavailable as error:
        first.append(
            Check("fail", str(error), "reinstall ippon, or run `uv sync --reinstall-package ippon` in a checkout")
        )
    return first + checks(device, profile, mlx_version, broken())


def mlx_failure(error: BaseException) -> Check:
    """The check for an MLX that doesn't load: under Rosetta (an x86_64 Python, which has no MLX), or broken."""
    machine = platform.machine()
    if machine != "arm64":
        fix = f"this Python runs as {machine} (Rosetta): use a native arm64 Python, as `platform.machine()` says"
    else:
        tested = mlx_range()
        fix = f"pip install --force-reinstall 'mlx{tested}'" if tested else "reinstall mlx"
    return Check("fail", f"MLX doesn't load: {type(error).__name__}: {error}", fix)


def report(found: list[Check], colour: bool = False) -> str:
    """The checks, a line each, with what to change under one that isn't ok. With `colour` each status is in its
    accent (TONES) and the commands a fix names in the accent of what to do next."""
    lines = []
    for c in found:
        lines.append(f"{terminal.styled(f'{c.status:5s}', colour, TONES[c.status])} {c.what}")
        if c.fix and c.status != "ok":
            lines.append(terminal.commands(f"      -> {c.fix}", colour))
    return "\n".join(lines)
