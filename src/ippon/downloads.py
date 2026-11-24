"""Pinned Hub transfers with licence acceptance, disk admission and reusable progress events.

Only metadata is read before admission. Pinned HTTP ranges and Hub-compatible file locks preserve partial
downloads across cancellation. Cached weights need no new acceptance and load offline.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import Any

from . import names, registry
from .errors import (
    DecisionError,
    DownloadFailed,
    InsufficientDisk,
    InvalidRequest,
    UnsupportedModel,
)
from .licences import Licence, of_metadata

# The Hub answers a request without a token with an X-HF-Warning header asking for one, which huggingface_hub logs on
# first use. ippon's public downloads need no token, so the note only alarms a new user. Filtered on import, since
# every module that asks the Hub imports this one first, and naming the logger doesn't import huggingface_hub.
logging.getLogger("huggingface_hub.utils._http").addFilter(lambda record: "unauthenticated" not in record.getMessage())

# never downloaded, whatever a pattern asks: a repository's code, pickles, and GGUF files but a pinned build's, but
# for a pickle an adapter names exactly (`registry.read_safely`), which only ptfile reads, running nothing from it
NEVER = ["*.py", "*.pyc", "*.bin", "*.pt", "*.pth", "*.pkl", "*.ckpt", "*.h5", "*.msgpack", "*.onnx", "*.gguf"]


def hub_failure(subject: str, what: str, error: Exception) -> DecisionError:
    """What a failed Hub request means: UnsupportedModel when the Hub answered that the repository, its revision or
    the file isn't there, or is closed to this user (another build is the way on), DownloadFailed when the network,
    offline mode or the Hub itself failed (sending the command again is the way on, so an agent never takes a network
    blip for a model it can't run)."""
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        HFValidationError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    # the Hub's own text for these runs to several lines of advice about tokens: one line says what to do
    if isinstance(error, GatedRepoError):  # a RepositoryNotFoundError too, so first
        return UnsupportedModel(
            f"{subject} is gated on Hugging Face: accept its terms on huggingface.co, then log in with `hf auth login`"
        )
    if isinstance(error, RevisionNotFoundError):
        return UnsupportedModel(f"{subject} has no such revision on Hugging Face")
    if isinstance(error, RepositoryNotFoundError):
        return UnsupportedModel(
            f"no model {subject} on Hugging Face, or it is private (if it is yours, log in with `hf auth login`)"
        )
    said = f"{subject}: {what} ({type(error).__name__}: {error})"
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(error, HFValidationError) or (
        isinstance(error, HfHubHTTPError) and status is not None and 400 <= status < 500 and status not in (408, 429)
    ):
        return UnsupportedModel(said)
    return DownloadFailed(f"{said}: check the network and run the command again")


def info(model: str) -> Any:
    """The Hub's metadata for `model` (org/name[@revision]), under the name it was asked by."""
    from huggingface_hub import HfApi

    repo, revision = names.split(model)
    try:
        found = HfApi().model_info(repo, revision=revision, files_metadata=True)
    except Exception as error:
        raise hub_failure(repo, "could not read model metadata", error) from error
    # A renamed repository answers under its new name, while the Hub cache keeps what is fetched by the name asked
    # (huggingface_hub follows the redirect): keep the name asked, so the configs and the weights share one folder
    found.id = repo
    return found


@dataclass(frozen=True)
class Configs:
    """A Hub repository's configs, downloaded before its weights: the folder they are in, each by file name, and every
    file the repository's listing names."""

    folder: Path
    values: dict[str, Any]
    listed: frozenset[str]


def configs(model: str) -> Configs:
    """`model`'s configs (its *config.json files and ippon.json) at the commit its metadata resolves to, without
    tokenizers or weights, and the files its listing names."""
    from huggingface_hub import hf_hub_download

    metadata = info(model)
    available = frozenset(s.rfilename for s in metadata.siblings or ())
    if any(Path(f).is_absolute() or ".." in Path(f).parts for f in available):
        raise UnsupportedModel(f"{model}: unsafe repository filename; no weights downloaded")
    wanted = [f for f in available if f.endswith("config.json") or f == "ippon.json"]
    values: dict[str, Any] = {}
    folder: Path | None = None
    sizes = {s.rfilename: s.size for s in metadata.siblings or ()}
    for file in sorted(wanted):
        if (sizes[file] or 0) > 2 << 20:
            raise UnsupportedModel(f"{model}: configuration {file} exceeds 2 MiB; no weights downloaded")
        try:
            path = Path(hf_hub_download(metadata.id, file, revision=metadata.sha, tqdm_class=quiet()))
            values[file] = json.loads(path.read_text())
            folder = path.parents[len(Path(file).parts) - 1]
        except json.JSONDecodeError as error:
            raise UnsupportedModel(f"{model}: cannot read configuration {file}: {error}") from error
        except Exception as error:
            raise hub_failure(model, f"cannot read configuration {file}", error) from error
    for file in sorted(available.intersection(registry.read_safely())):  # a pickle an adapter matches by, read safely
        hf_hub_download(metadata.id, file, revision=metadata.sha, tqdm_class=quiet())
    if folder is None:
        if any(f.endswith(".gguf") for f in available):  # a GGUF repository: its builds are named one by one
            raise InvalidRequest(
                f"{model} holds GGUF files, one build each: name the one to load, {model}:Q8_0 (ippon runs Q8_0 "
                "builds, and `ippon pull --list` lists them)"
            )
        raise UnsupportedModel(f"{model}: no supported model configuration; no weights downloaded")
    return Configs(folder, values, available)


def hub_cache() -> Path:
    """The Hub cache's root, read when asked: what ippon records downloads under and scans to delete them, so both
    name the same folder (the Hub's own scan reads a copy of the setting taken when it was first imported)."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def cache_folder(repo: str) -> Path:
    """The Hub cache's folder of model repository `repo` (its blobs, snapshots and refs)."""
    from huggingface_hub.file_download import repo_folder_name

    return hub_cache() / repo_folder_name(repo_id=repo, repo_type="model")


@cache
def quiet() -> type[Any]:
    """The Hub's own progress bar, every one of it disabled: the tqdm_class for its downloads ippon shows no progress
    of, without turning the Hub's bars off for anything else in the process. Made on first use, so that importing
    this module doesn't import huggingface_hub."""
    from huggingface_hub.utils.tqdm import tqdm

    class Quiet(tqdm):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **{**kwargs, "disable": True})  # type: ignore[no-untyped-call]

    return Quiet


@dataclass(frozen=True)
class File:
    repo: str
    revision: str
    name: str
    size: int
    cached: bool
    partial: Path

    @property
    def remaining(self) -> int:
        if self.cached or self.partial.with_suffix("").is_file():
            return 0
        try:
            # Sparse preallocation is not downloaded data. Only allocated blocks count against free disk space.
            allocated = self.partial.stat().st_blocks * 512
        except FileNotFoundError:
            allocated = 0
        return max(0, self.size - allocated)


def listing(metadata: Any, patterns: Sequence[str], *, gguf: bool = False) -> list[Any]:
    """The files of a repository's Hub listing (`info`) that `patterns` match, NEVER's left out but for a `gguf`
    build's own file."""
    safe = registry.read_safely()
    return [
        s
        for s in metadata.siblings or ()
        if any(fnmatch(s.rfilename, p) for p in patterns)
        and (
            gguf or (s.rfilename in safe and s.rfilename in patterns) or not any(fnmatch(s.rfilename, n) for n in NEVER)
        )
    ]


def files(model: str, patterns: Sequence[str], *, gguf: bool = False) -> tuple[Licence, list[File]]:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, try_to_load_from_cache

    metadata = info(model)
    result = []
    if not names.is_commit(str(metadata.sha)):
        raise UnsupportedModel(f"{model}: metadata did not resolve to an immutable commit")
    if any(Path(s.rfilename).is_absolute() or ".." in Path(s.rfilename).parts for s in metadata.siblings or ()):
        raise UnsupportedModel(f"{model}: unsafe repository filename; no weights downloaded")
    for sibling in listing(metadata, patterns, gguf=gguf):
        name = sibling.rfilename
        found = try_to_load_from_cache(metadata.id, name, revision=metadata.sha)
        cached = isinstance(found, str) and Path(found).is_file()
        if cached:
            assert isinstance(found, str)
            partial, size = Path(found), Path(found).stat().st_size
        else:
            try:
                detail = get_hf_file_metadata(hf_hub_url(metadata.id, name, revision=metadata.sha))
            except Exception as error:
                raise hub_failure(f"{model}/{name}", "could not read file metadata", error) from error
            if (
                detail.size is None
                or detail.etag is None
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", detail.etag)
            ):
                raise UnsupportedModel(f"{model}/{name}: missing or invalid file metadata; no weights downloaded")
            size = detail.size
            partial = cache_folder(str(metadata.id)) / "blobs" / (detail.etag + ".incomplete")
        result.append(File(str(metadata.id), str(metadata.sha), name, size, cached, partial))
    return of_metadata(metadata), result


def disk_space(items: Sequence[File]) -> tuple[int, int]:
    from huggingface_hub import constants

    target = Path(constants.HF_HUB_CACHE).expanduser()
    while not target.exists():
        target = target.parent
    return sum(f.remaining for f in items), shutil.disk_usage(target).free


def admit_disk(items: Sequence[File]) -> tuple[int, int]:
    needed, free = disk_space(items)
    if needed > free:
        raise InsufficientDisk(f"download needs {needed:,} more bytes; only {free:,} bytes are free on the cache disk")
    return needed, free


@dataclass(frozen=True)
class Progress:
    phase: str
    done: int
    total: int
    bytes_per_second: float
    seconds_left: float | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _download(item: File, progress: Callable[[int], None]) -> Path:
    """Resume pinned bytes under the Hub blob lock; publish only a complete, hash-checked file.

    Hub 2.2 discards its unique temporary files on cancellation. Own the partial file here so resuming does not
    depend on that transfer implementation; metadata, authentication, timeouts and retries still use Hub APIs."""
    from huggingface_hub import constants, hf_hub_url
    from huggingface_hub.utils import build_hf_headers, http_stream_backoff  # type: ignore[attr-defined]

    if item.cached:
        if not item.partial.is_file():
            raise DownloadFailed(f"{item.name}: cached file disappeared; repeat the pull to inspect it again")
        return item.partial
    blob = item.partial.with_suffix("")
    root = blob.parent.parent
    target = root / "snapshots" / item.revision / item.name
    lock_path = root.parent / ".locks" / root.name / (blob.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not blob.is_file():
            blob.parent.mkdir(parents=True, exist_ok=True)
            offset = item.partial.stat().st_size if item.partial.exists() else 0
            if offset > item.size:
                item.partial.unlink()
                raise DownloadFailed(
                    f"{item.name}: partial download exceeds the pinned size; corrupt partial removed, repeat to retry"
                )
            progress(offset)
            if offset < item.size:
                headers = build_hf_headers(library_name="ippon", headers={"Accept-Encoding": "identity"})
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                url = hf_hub_url(item.repo, item.name, revision=item.revision)
                # the Hub's own transfer settings: its read timeout (HF_HUB_DOWNLOAD_TIMEOUT) and retries on 408/429
                with http_stream_backoff(
                    "GET",
                    url,
                    headers=headers,
                    follow_redirects=True,
                    timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
                    retry_on_exceptions=(),
                    retry_on_status_codes=(408, 429),
                ) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        expected = f"bytes {offset}-{item.size - 1}/{item.size}"
                        if response.headers.get("Content-Range") != expected:
                            raise DownloadFailed(f"{item.name}: server returned an unexpected byte range")
                    elif response.status_code == 200:
                        offset = 0  # A server may ignore Range. Restart instead of appending duplicate bytes.
                    else:
                        raise DownloadFailed(f"{item.name}: unexpected download status {response.status_code}")
                    with item.partial.open("ab" if offset else "wb") as output:
                        progress(offset)
                        for chunk in response.iter_bytes(chunk_size=1 << 20):
                            if offset + len(chunk) > item.size:
                                raise DownloadFailed(f"{item.name}: response exceeds the pinned size")
                            output.write(chunk)
                            output.flush()
                            offset += len(chunk)
                            progress(offset)
                        os.fsync(output.fileno())
                if offset != item.size:
                    raise DownloadFailed(f"{item.name}: received {offset:,} of {item.size:,} bytes")
            digest = hashlib.sha256() if len(blob.name) == 64 else hashlib.sha1()
            if len(blob.name) == 40:
                digest.update(f"blob {item.size}\0".encode())  # a Git blob's id: its header, then its content
            with item.partial.open("rb") as downloaded:
                digest = hashlib.file_digest(downloaded, lambda: digest)
            if digest.hexdigest() != blob.name:
                item.partial.unlink()
                raise DownloadFailed(f"{item.name}: content hash mismatch; corrupt partial removed, repeat to retry")
            item.partial.replace(blob)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
            try:
                temporary.symlink_to(os.path.relpath(blob, target.parent))
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return target


def transfer(items: Sequence[File], progress: Callable[[Progress], None] | None = None) -> Path:
    """Download pinned files with aggregate progress. Cancellation retains their reusable HTTP partials. The caller
    admits the disk (`admit_disk`) and then requires the licence first, so a download the disk can't hold records no
    acceptance (`fetch_weights`, `pull.Plan.run`)."""
    total = sum(f.size for f in items)
    completed = sum(f.size for f in items if f.cached)
    started, received, current = time.monotonic(), 0, 0
    path: Path | None = None

    def emit(phase: str) -> None:
        done = min(total, completed + current)
        rate = received / max(time.monotonic() - started, 0.001)
        if progress:
            remaining = 0.0 if done == total else (total - done) / rate if rate else None
            progress(Progress(phase, done, total, rate, remaining))

    emit("downloading")
    try:
        for item in items:
            previous: int | None = None

            def advance(done: int) -> None:
                nonlocal current, received, previous
                if previous is not None:
                    received += max(0, done - previous)
                current, previous = done, done
                emit("downloading")

            path = _download(item, advance)
            if not item.cached:
                completed += item.size
            current = 0
            emit("downloading")
    except KeyboardInterrupt:
        emit("cancelled")
        raise
    except Exception as error:
        emit("failed")
        if isinstance(error, DecisionError):
            raise
        if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
            raise InsufficientDisk(
                "the cache disk ran out of space or reached its quota: free disk space, then repeat the command "
                "to resume. Partial files kept."
            ) from None
        raise DownloadFailed(f"download failed: {error}. Partial files kept; repeat the command to resume.") from error
    if path is None:
        raise UnsupportedModel("no files matched this download")
    emit("complete")
    return path.parents[len(Path(items[-1].name).parts) - 1]


def fetch_weights(model: str, patterns: Sequence[str], *, gguf: bool = False) -> Path:
    """`model`'s files that `patterns` match, its weights among them (with `gguf`, a GGUF build's file), downloaded
    once the disk is admitted and the licence accepted: the boundary every load of missing weights crosses
    (`hub.fetch`, `hub.pinned_file`), a LoRA's base included.

    A load asks for the licence only when it downloads weights it doesn't have: small files, and weights already in
    the cache, load without it. `ippon pull` (`pull.Plan.run`) asks for every licence of what it pulls, cached or not,
    since a pull is the step where a person reads and accepts a model's terms."""
    license_, items = files(model, patterns, gguf=gguf)
    admit_disk(items)
    if any(not f.cached and (gguf or f.name.endswith(".safetensors")) for f in items):
        license_.require(model)
    return transfer(items)
