"""A file's sha256, hashed once per version of the file.

A pinned GGUF build, an Ollama blob and a build's provenance each need a file's sha256. Hashing takes seconds per GB,
so a file is hashed the first time a version of it is seen and remembered by its real path and version in the ippon
cache (hashes.json). The Hub cache names a blob by its etag, which is the file's sha256 for files stored with LFS but
not for those stored with Xet: a blob named by the sha256 expected of it is taken as having it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .storage import cache_dir, read_record, write_json_atomic

HASHES = "hashes.json"  # in the ippon cache: content hashes by file version (`content_sha256`)


def _version(real: Path) -> list[int]:
    """A version of a file: its device, inode, size, and modification and change times. Writing a file changes its
    change time, which no tool sets back, so a file whose version is unchanged has the content it had."""
    st = real.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def content_sha256(path: Path) -> str:
    """The sha256 of the content of the file at `path`: hashed the first time a version of it is seen (seconds per
    GB), then remembered by its real path and version in the ippon cache (hashes.json)."""
    real = path.resolve()
    version = _version(real)
    record = cache_dir() / HASHES
    known = read_record(record) or {}  # remembered hashes only: an unreadable record is hashed afresh
    entry = known.get(str(real))
    if isinstance(entry, list) and entry[:-1] == version:
        return str(entry[-1])
    with real.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if _version(real) == version:  # not changed while it was read
        known[str(real)] = [*version, sha]
        write_json_atomic(record, known)
    return sha


def sha256(path: Path, expected: str = "") -> str:
    """The sha256 of the file at `path`: `expected` when the file is a Hub cache blob named by it, else its content's
    (`content_sha256`)."""
    return expected if expected and path.resolve().name == expected else content_sha256(path)


def verified(path: Path, expected: str) -> bool:
    """Whether the file at `path` has content `expected` (a sha256)."""
    return sha256(path, expected) == expected
