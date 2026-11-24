"""The Mac a measurement runs on, as every command that times or records one sees it (bench, ab, compare, suite,
profile, doctor): other processes busy beside it, the ippon revision being run, and one description of it for
results, which also records its energy mode and power source."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

from . import __version__
from .device import detect

BUSY_CPU = 50.0  # a process above this share of one core while timing is reported as busy


def power() -> tuple[str | None, bool]:
    """(energy mode, on AC power) from pmset; (None, True) where pmset is unavailable."""
    try:
        settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True, check=True).stdout
        battery = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, True
    mode = next((line.split()[-1] for line in settings.splitlines() if line.strip().startswith("powermode")), None)
    return mode, "AC Power" in battery


def busy() -> list[str]:
    """Other processes using more than BUSY_CPU% of a core now (their commands, shortened): timings taken beside them
    are noisier."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,pcpu=,args="], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    mine = {os.getpid(), os.getppid()}
    found = []
    for line in out.splitlines():
        pid, cpu, *args = line.split()
        if not args or int(pid) in mine or float(cpu) <= BUSY_CPU or args[0].endswith(("WindowServer", "kernel_task")):
            continue
        name = Path(args[0]).name
        script = next((Path(a).name for a in args[1:3] if a.endswith(".py") or a == "-m"), None)
        found.append(f"{name}{' ' + script if script and script != '-m' else ''} ({float(cpu):.0f}%)")
    return found


def revision() -> str | None:
    """The git revision of the ippon source being run, "+" when it has uncommitted changes: a regular install's
    as its wheel recorded it (hatch_build.py), else its checkout's; None when neither says."""
    here = Path(__file__).parent
    if (here / "REVISION").is_file():
        return (here / "REVISION").read_text().strip() or None
    try:
        rev = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--show-prefix", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True, check=False
        )
    except OSError:
        return None
    prefix, _, short = rev.stdout.partition("\n")
    if rev.returncode != 0 or prefix != "src/ippon/":  # not ippon's own checkout (another repository's)
        return None
    return short.strip() + ("+" if dirty.stdout.strip() else "")


SECRET_WORDS = ("KEY", "TOKEN", "SECRET")  # a variable naming one holds a credential: recorded as "<set>"


def environment() -> dict[str, str]:
    """The MLX_ and IPPON_ variables that change how ippon runs, as `info` shows them and results record them, a
    credential's value (IPPON_API_KEY) replaced by "<set>"."""
    return {
        name: "<set>" if any(word in name for word in SECRET_WORDS) else value
        for name, value in sorted(os.environ.items())
        if name.startswith(("MLX_", "IPPON_"))
    }


def describe() -> dict[str, Any]:
    """This Mac and what runs on it, as results record them."""
    d = detect()
    mode, ac = power()
    return {
        "chip": d.name,
        "architecture": d.architecture,
        "memory_gib": round((d.memory_bytes or 0) / 2**30),
        "macos": platform.mac_ver()[0],
        "mlx": version("mlx"),
        "ippon": __version__,
        "revision": revision(),
        "power": {"mode": mode, "ac": ac},
    }
