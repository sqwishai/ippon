"""Where ippon keeps what it makes on this Mac: the records of its servers, of complete downloads and of files'
content hashes, under one folder (IPPON_CACHE, else ~/Library/Caches/ippon), and the user's choices (setup's, the
licences accepted) under another (IPPON_CONFIG); and how a record there is written. Each module that writes there
names its own files (onboarding.CONFIG_FILES and CACHE_FILES collect them)."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# the variables that move ippon's folders and the Hub's: what a process started apart from this shell (the login job,
# Hermes's MCP server) is told, so it reads the same records and models. None of them holds a key or a token
FOLDER_ENV = ("IPPON_CONFIG", "IPPON_CACHE", "HF_HOME", "HF_HUB_CACHE")


def moved_folders() -> dict[str, str]:
    """The FOLDER_ENV variables this environment sets, by name."""
    return {name: os.environ[name] for name in FOLDER_ENV if os.environ.get(name)}


def cache_dir() -> Path:
    return Path(os.environ.get("IPPON_CACHE") or Path.home() / "Library" / "Caches" / "ippon").expanduser()


def config_dir() -> Path:
    """Durable user choices; separate from the model caches that macOS may reclaim."""
    return Path(
        os.environ.get("IPPON_CONFIG") or Path.home() / "Library" / "Application Support" / "ippon"
    ).expanduser()


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Held while a record ippon keeps is read, changed and written again, across processes: writing it whole
    (`write_json_atomic`) can't stop two at once each reading the old one and the last write dropping the other's
    change. The lock is the file `path` with ".lock" added."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_record(path: Path) -> dict[str, Any] | None:
    """A JSON record ippon keeps: {} when there is none, None when it can't be read or isn't an object."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def unreadable(name: str) -> str:
    """Where an unreadable record named `name` is set aside, beside it: downloads.json -> downloads.unreadable.json."""
    return name.removesuffix(".json") + ".unreadable.json"


def set_aside(path: Path) -> None:
    """An unreadable record moved aside unchanged (`unreadable`) before a fresh one is written over it, so what it
    named (the snapshots or agents ippon changed) can still be found."""
    if read_record(path) is None:
        path.replace(path.with_name(unreadable(path.name)))


def write_json_atomic(path: Path, data: Any, *, indent: int | None = None, mode: int | None = None) -> None:
    """`data` as JSON at `path`, whole or not at all: written to a file of its own beside it, then moved over it, so
    a reader, or another process writing at once, never sees half a file. `mode`: the file's permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as out:
            if mode is not None:
                temporary.chmod(mode)
            json.dump(data, out, indent=indent)
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
