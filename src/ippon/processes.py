"""The fresh processes measurements run in (ab's and compare's arms, verify's --against arm, suite's steps): a command
run with its output captured, and one that writes its result as JSON to a file."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ._portable import PRIVATE_REPORT_ENV
from .errors import DecisionError


def command() -> list[str]:
    """How to run ippon in another process: this Python, this installation. -P keeps the folder it starts in off the
    import path: an agent starts ippon's MCP server, and the server, in whatever repository it has open, where a
    json.py or the like would otherwise run as the user."""
    return [sys.executable, "-P", "-m", "ippon"]


def spawn(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, env=env, check=False)


def run_json(command: list[str], out: Path, name: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    """`command`'s result, which it writes to `out` as JSON; DecisionError naming `name` with the end of its stderr
    when it fails."""
    done = spawn(command, {**(os.environ if env is None else env), PRIVATE_REPORT_ENV: str(out.absolute())})
    if done.returncode != 0:
        raise DecisionError(f"{name} failed:\n{done.stderr.strip()[-2000:]}")
    return dict(json.loads(out.read_text()))
