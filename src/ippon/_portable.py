"""Portable paths in published evidence; never use these records to locate or execute files.

This module uses only the standard library so independent reference runners can load it without importing
ippon or any inference runtime. Transform strings only: measurements and executable inputs stay intact.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_START = r"(?<![\w.~])"
_END = r"(?=/|$|[\s\\\"'`<>),;:\]])"
_HOME = re.compile(_START + r"/(?:Users|home)/[^/\s\\\"'`<>]+")
_MAC_TEMP = re.compile(_START + r"/(?:private/)?var/folders/[^/\s]+/[^/\s]+/([TC])" + _END)
_TEMP = re.compile(_START + r"/(?:private/)?(?:tmp|var/tmp)" + _END)
_CACHE = re.compile(r"~/(?:\.cache|Library/Caches)" + _END)
_PRIVATE = re.compile(_START + r"/private" + _END)
_PROCESS = re.compile(r".+ \((\d+)%\)$")
PRIVATE_REPORT_ENV = "IPPON_PRIVATE_REPORT"


def private_report(path: str | Path | None = None) -> bool:
    """Whether this is a measurement child's private result for its parent, rather than a published copy."""
    target = os.environ.get(PRIVATE_REPORT_ENV)
    return bool(target) and (path is None or Path(path).absolute() == Path(str(target)).absolute())


def text(value: str) -> str:
    """Replace home, cache and temporary roots, including paths inside commands or embedded JSON strings."""
    home = str(Path.home())
    if home != "/":
        value = re.sub(_START + re.escape(home) + _END, "~", value)
    value = _HOME.sub("~", value)
    value = _MAC_TEMP.sub(lambda m: "<tmp>" if m[1] == "T" else "<cache>", value)
    value = _TEMP.sub("<tmp>", value)
    value = _CACHE.sub("<cache>", value)
    return _PRIVATE.sub("<private>", value)


def transcript(value: str) -> str:
    """A saved command transcript with portable paths and anonymous busy-process summaries."""
    return re.sub(
        r"(?m)^(busy while timing: )([^\n]*\(\d+%\)[^\n]*)$",
        lambda m: m[1] + ", ".join(f"process ({share}%)" for share in re.findall(r"\((\d+)%\)", m[2])),
        text(value),
    )


def _value(value: Any, process_names: bool = False, key: str = "") -> Any:
    if isinstance(value, str):
        if not process_names and key in ("busy", "activity") and (found := _PROCESS.fullmatch(value)):
            return f"process ({found[1]}%)"
        return text(value) if process_names else transcript(value)
    if isinstance(value, dict):
        return {text(k) if isinstance(k, str) else k: _value(v, process_names, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(v, process_names, key) for v in value]
    return value


def record(value: dict[str, Any], *, process_names: bool = False) -> dict[str, Any]:
    """A portable copy, anonymizing busy processes unless it is private transport for live command output."""
    return {text(k): _value(v, process_names, k) for k, v in value.items()}


def dumps(value: Any, *, process_names: bool = False, **options: Any) -> str:
    """Encode publishable JSON with the same numeric values and options as json.dumps."""
    return json.dumps(_value(value, process_names), **options)
