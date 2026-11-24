"""Models by Hugging Face Hub id, or a local directory.

`Decider.load("LiquidAI/d1-2B")` downloads in two steps: the small JSON and text files first, so the adapter
can be chosen, then the files that adapter lists, from the commit the first step read (`pinned`): a branch that moves
in between can't mix two commits' files. Only safetensors, JSON and text files are ever downloaded: a model
repository's code and pickles never reach the disk, but for a pickle an adapter names exactly (its `read_safely`),
which only ptfile reads, running nothing from it. `repo@revision` pins a commit or tag; HF_HUB_OFFLINE=1 uses the
local cache only.

A model downloaded before loads again without asking the Hub: each complete download is recorded (DOWNLOADS, in the
ippon cache: the commit a branch or tag resolved to, and the files each set of patterns fetched at that commit),
and a load whose files are all still in the Hub cache reads them from there, at the commit it was downloaded at. That
spares a load the Hub's round trips (about 0.1 s for a small model on the M5 Pro, seconds on a slow network) and lets it
run offline (`ippon pull` asks the Hub for a branch's newest commit, and records that download). Every download, a
load's, a pull's and a pinned file's, ends in `complete`, which checks a pinned file against its sha256 (hashed once
per version of the file, hashes.py) and records it, with one rule for which snapshots are ippon's to delete.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import os
import uuid
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import downloads, registry
from .downloads import NEVER, cache_folder
from .errors import DecisionError, UnsupportedModel
from .hashes import verified
from .names import is_commit, split
from .storage import cache_dir, locked, read_record, set_aside, unreadable, write_json_atomic

# what ippon downloads: weights as safetensors, small text files, a model card and its licence (and a pickle only when
# an adapter names it exactly, `registry.read_safely`, which only ptfile reads)
ALLOWED = (".safetensors", ".json", ".txt", ".model", ".jinja", ".md", "LICENSE")
DOWNLOADS = "downloads.json"  # in the ippon cache: complete downloads (`fetch`), by repository and commit
LOCK = DOWNLOADS + ".lock"  # beside it: held while one download's record is added (`storage.locked`)
UNREADABLE = unreadable(DOWNLOADS)  # beside it: the last record that couldn't be read, set aside unchanged


def probe() -> tuple[str, ...]:
    """What a model's first, small download fetches to choose its adapter: its configs and texts, its card and
    licence, and the pickles adapters match their models by (`registry.read_safely`), a few MB each, read safely."""
    return ("*.json", "*.txt", "*.jinja", "README.md", "LICENSE", *sorted(registry.read_safely()))


def pinned(model: str, snapshot: Path) -> str:
    """`model` at the commit of `snapshot`, a Hub cache snapshot directory (named by its commit), so that the next
    download of the same model reads that commit, whatever its branch points to by then."""
    repo, _ = split(model)
    return f"{repo}@{snapshot.name}" if is_commit(snapshot.name) else model


def snapshot_dir(repo: str, revision: str) -> Path:
    """The Hub cache's folder of `repo` at commit `revision`: its files, as links into the cache's blobs."""
    return cache_folder(repo) / "snapshots" / revision


def parse_snapshot(path: Path | str) -> tuple[str, str] | None:
    """(repository, commit) of a Hub cache snapshot folder, or of a file in one (`snapshot_dir`); None for any other
    path. The path is read as given, not resolved: a snapshot's files are links into the cache's blobs."""
    parts = Path(path).expanduser().parts
    for i in range(len(parts) - 2):
        if parts[i].startswith("models--") and parts[i + 1] == "snapshots" and is_commit(parts[i + 2]):
            return parts[i].removeprefix("models--").replace("--", "/", 1), parts[i + 2]
    return None


def cached_file(repo: str, file: str, revision: str) -> Path | None:
    """`repo`'s `file` at commit `revision` in the Hub cache, if it was downloaded."""
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(repo, file, revision=revision)
    return Path(found) if isinstance(found, str) and Path(found).is_file() else None


def pinned_file(repo: str, revision: str, file: str, sha256: str) -> Path:
    """One file of `repo` at commit `revision` whose content's sha256 is pinned: a GGUF build's file (verified.py,
    its licence accepted first) or a small file ippon reads beside a model, from the Hub cache or downloaded there;
    refused unless its content is the pinned one. The Hub cache names a blob by its etag, which is that sha256 for
    files stored with LFS but not for those stored with Xet: those are hashed once (seconds per GB) and remembered by
    path, size and time (`hashes.verified`)."""
    download = Download(f"{repo}@{revision}", (file,), pins=(sha256,))
    path = cached_file(repo, file, revision)
    if path is not None:
        _check_pins(download, path.parent)
        return path
    before = blobs(repo)
    try:
        folder = downloads.fetch_weights(download.model, [file], gguf=download.gguf)
    except DecisionError:
        raise
    except Exception as error:  # the hub raises many types; the message says which
        raise downloads.hub_failure(f"{repo}/{file}", "could not download", error) from error
    return complete(download, folder, before) / file


@dataclass(frozen=True)
class Download:
    """Files downloaded together: those of repository `model` (repo@revision) that `patterns` match, recorded for the
    model as it was asked too (`requested`), and for a pinned download (a GGUF build's file, then its mmproj file),
    the sha256 each file named in `patterns` must have, in order (`pins`)."""

    model: str
    patterns: tuple[str, ...]
    requested: str | None = None
    pins: tuple[str, ...] = ()

    @property
    def gguf(self) -> bool:
        return any(p.endswith(".gguf") for p in self.patterns)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"model": self.model}
        if self.requested is not None:
            out["requested"] = self.requested
        out["patterns"] = list(self.patterns)
        if self.gguf:
            out |= {"gguf": True, "sha256": self.pins[0]}
        return out


def complete(download: Download, folder: Path, before: set[str]) -> Path:
    """What every download ends with, once its files are all in the snapshot `folder`: each pinned file checked
    against its sha256, then the download recorded (`record_download`), for `model` and as it was `requested`. The
    snapshot is ippon's to delete (`delete_downloads`) when one of its own files links to a blob the Hub cache didn't
    hold in `before` (`blobs`, read before it started): a snapshot whose files were all there already, or that another
    tool downloaded, never becomes ippon's, even when another tool adds a file to another revision meanwhile.
    `folder`, returned."""
    _check_pins(download, folder)
    fetched = any(f.resolve().name not in before for f in folder.rglob("*") if f.is_symlink())
    for model in (download.model, download.requested):
        if model is not None:
            record_download(model, download.patterns, folder, fetched=fetched)
    return folder


def _check_pins(download: Download, folder: Path) -> None:
    for file, sha256 in zip(download.patterns, download.pins, strict=False):
        if not verified(folder / file, sha256):
            raise UnsupportedModel(
                f"{split(download.model)[0]}/{file}: the downloaded file is not the pinned one (sha256 {sha256[:12]})"
            )


def fetch(model: str, patterns: Sequence[str]) -> Path:
    """The local snapshot directory of `model` holding its files that match `patterns` (and are safe to have): from
    an earlier complete download when every file of it is still in the Hub cache (`downloaded`), else downloaded."""
    safe = registry.read_safely()
    unsafe = [p for p in patterns if not p.endswith(ALLOWED) and p not in safe]
    if unsafe:
        raise UnsupportedModel(f"refusing to download {unsafe}: only {', '.join(ALLOWED)} files are fetched")
    repo, revision = split(model)
    found = downloaded(model, patterns)
    if found is not None:
        return found
    from huggingface_hub import constants, hf_hub_download, snapshot_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    if constants.HF_HUB_OFFLINE and revision is None:
        # The Hub cache's refs/main can name a snapshot that holds a stray file or two (another tool fetched only an
        # index there), and offline that folder is all snapshot_download returns. A complete download ippon recorded
        # of the same repo is the model as it was last read whole.
        found = recorded_offline(repo, patterns)
        if found is not None:
            return found
    before = blobs(repo)
    try:
        if any(p.endswith(".safetensors") for p in patterns) and not constants.HF_HUB_OFFLINE:
            path = downloads.fetch_weights(model, patterns)
        else:
            path = Path(
                snapshot_download(
                    repo,
                    revision=revision,
                    allow_patterns=list(patterns),
                    ignore_patterns=NEVER,
                    tqdm_class=downloads.quiet(),
                )
            )
            # NEVER drops every pickle, so one an adapter names exactly comes on its own, at the snapshot's commit,
            # where the repository has one: the probe asks for every adapter's, and a model has only its own
            for name in (p for p in patterns if p in safe):
                with suppress(RemoteEntryNotFoundError):
                    hf_hub_download(repo, name, revision=path.name)
    except DecisionError:
        raise
    except Exception as error:  # the hub raises many types; the message says which
        raise downloads.hub_failure(model, "could not download", error) from error
    if constants.HF_HUB_OFFLINE:  # offline, the cache's folder is all there is: nothing was checked complete
        return path
    return complete(Download(model, tuple(patterns)), path, before)


def blobs(repo: str) -> set[str]:
    """The files the Hub cache holds for `repo` (its blobs), to tell a download from a model another tool fetched."""
    folder = cache_folder(repo) / "blobs"
    return {entry.name for entry in folder.iterdir()} if folder.is_dir() else set()


def delete_downloads() -> list[str]:
    """`ippon remove all --models`: the snapshots ippon downloaded deleted from the Hub cache, by revision, as the
    Hub's own cache tools delete them (a file another revision still uses stays); never a revision ippon only reused,
    never one that holds a file ippon's downloads don't name (another app's quantization of the same commit), and
    never another revision of the same repository. What it did, a line each."""
    from huggingface_hub import CachedRevisionInfo, scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        cache = scan_cache_dir(downloads.hub_cache())
    except CacheNotFound:  # no Hub cache on this Mac: nothing was downloaded into it
        return []
    # a snapshot's folder names its repository and its commit: two repositories that share a commit (a fork) are
    # told apart by it, where the commit alone would make the other one's revision look like ippon's
    ours = {
        Path(entry["path"]).resolve(): {file for files in entry["fetched"].values() for file in files}
        for entry in _read_downloads()["snapshots"].values()
        if entry.get("path") and entry.get("downloaded")
    }
    done = []
    for repo in cache.repos:
        revisions: list[CachedRevisionInfo] = []
        shared: list[CachedRevisionInfo] = []
        for r in repo.revisions:
            if (files := ours.get(r.snapshot_path.resolve())) is not None:
                held = {f.file_path.relative_to(r.snapshot_path).as_posix() for f in r.files}
                (revisions if held <= files else shared).append(r)
        done += [
            f"{repo.repo_id}: revision {r.commit_hash[:12]} kept, it holds files another app downloaded" for r in shared
        ]
        if revisions:
            # the Hub picks revisions by commit in every repository it is given: given only this one, a fork that
            # holds the same commit keeps it
            alone = dataclasses.replace(cache, repos=frozenset({repo}))
            strategy = alone.delete_revisions(*(r.commit_hash for r in revisions))
            strategy.execute()
            done.append(
                f"{repo.repo_id}: {len(revisions)} revision(s) deleted ({strategy.expected_freed_size / 1e9:.1f} GB)"
            )
    return done


def known(model: str) -> str | None:
    """The commit `model` (org/name, or org/name@branch-or-tag) resolved to when it was last downloaded; None when it
    never was (or names a commit)."""
    repo, revision = split(model)
    commit = _read_downloads()["resolved"].get(_ref(repo, revision))
    return str(commit) if commit else None


def downloaded(model: str, patterns: Sequence[str]) -> Path | None:
    """The snapshot directory an earlier complete download of `model`'s `patterns` left, at the commit `model` names
    or, for a branch or tag, the one it resolved to then; None when there was none, or a file of it is gone from the
    Hub cache."""
    found = downloaded_files(model, patterns)
    return found[0] if found else None


def downloaded_files(model: str, patterns: Sequence[str]) -> tuple[Path, list[str]] | None:
    """`downloaded`'s snapshot directory and the files the download fetched there (relative paths)."""
    snapshot = _snapshot(model)
    files = snapshot["fetched"].get(_patterns(patterns)) if snapshot else None
    if snapshot is None or files is None:
        return None
    path = Path(snapshot["path"])
    return (path, list(files)) if all((path / f).exists() for f in files) else None  # exists() follows the links


def recorded_offline(repo: str, patterns: Sequence[str]) -> Path | None:
    """The newest snapshot of `repo` where a complete download of `patterns` was recorded and every file of it is
    still in the Hub cache, at any commit; None when there is none."""
    whole = []
    for key, snapshot in _read_downloads()["snapshots"].items():
        files = snapshot["fetched"].get(_patterns(patterns))
        path = Path(snapshot["path"])
        if key.rpartition("@")[0] == repo and files is not None and all((path / f).exists() for f in files):
            whole.append(path)
    return max(whole, key=lambda path: path.stat().st_mtime, default=None)


def downloaded_weights(model: str) -> Path | None:
    """The snapshot folder of `model`'s weights (safetensors files) when a load downloaded them whole, at the commit
    it names or its branch resolved to then, and every file of that download is still in the Hub cache: read from the
    record alone. None otherwise."""
    snapshot = _snapshot(model)
    if snapshot is None:
        return None
    path = Path(snapshot["path"])
    whole = any(
        any(f.endswith(".safetensors") for f in files) and all((path / f).exists() for f in files)
        for files in snapshot["fetched"].values()
    )
    return path if whole else None


def _snapshot(model: str) -> dict[str, Any] | None:
    """The record of `model`'s snapshot: at the commit it names, or the one its branch or tag resolved to when it
    was last downloaded."""
    repo, revision = split(model)
    record = _read_downloads()
    commit = revision if is_commit(revision or "") else record["resolved"].get(_ref(repo, revision))
    snapshot = record["snapshots"].get(f"{repo}@{commit}") if commit else None
    return dict(snapshot) if snapshot else None


def _ref(repo: str, revision: str | None) -> str:
    return repo + (f"@{revision}" if revision else "")


def _patterns(patterns: Sequence[str]) -> str:
    return "\n".join(sorted(patterns))


def _read_downloads(rewriting: bool = False) -> dict[str, Any]:
    """The record of complete downloads (DOWNLOADS); empty when there is none or it can't be read, which only makes a
    load ask the Hub again. Before `rewriting` it, an unreadable one is moved aside to UNREADABLE as it is: it names
    the snapshots `ippon remove all --models` may delete, which a record started afresh would forget."""
    path = cache_dir() / DOWNLOADS
    if rewriting:
        set_aside(path)
    record = read_record(path) or {}
    return {"resolved": dict(record.get("resolved") or {}), "snapshots": dict(record.get("snapshots") or {})}


def record_download(model: str, patterns: Sequence[str], path: Path, *, fetched: bool = True) -> None:
    """Record a complete download (`complete` is the one caller outside tests): every file of the snapshot `path`
    that `patterns` match (NEVER's only when a pattern names it exactly, as a pinned GGUF file's does), and, for a
    branch or tag, the commit it resolved to. `fetched`: whether this brought files the Hub cache didn't hold, which
    makes the snapshot ippon's to delete (`delete_downloads`), once and for good."""
    if not is_commit(path.name):
        return
    repo, revision = split(model)
    files = sorted(
        relative
        for folder, _, names in os.walk(path)
        for name in names
        if (relative := (Path(folder) / name).relative_to(path).as_posix())
        and any(fnmatch.fnmatch(relative, p) for p in patterns)
        and (relative in patterns or not any(fnmatch.fnmatch(relative, n) for n in NEVER))  # a file named exactly
    )
    with locked(cache_dir() / DOWNLOADS):  # two downloads finishing at once each keep their record
        record = _read_downloads(rewriting=True)
        if not is_commit(revision or ""):
            record["resolved"][_ref(repo, revision)] = path.name
            # Publish the Hub's branch reference too: offline loads may ask for a different set of small files.
            ref = revision or "main"
            if Path(ref).is_absolute() or ".." in Path(ref).parts:
                raise UnsupportedModel(f"{model}: unsafe revision name")
            pointer = path.parent.parent / "refs" / ref
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pending = pointer.with_name(pointer.name + f".{uuid.uuid4().hex}.tmp")
            try:
                pending.write_text(path.name)
                pending.replace(pointer)
            finally:
                pending.unlink(missing_ok=True)
        key = f"{repo}@{path.name}"
        snapshot = record["snapshots"].get(key) or {"path": str(path), "fetched": {}}
        downloaded = bool(snapshot.get("downloaded")) or fetched
        snapshot = {
            "path": str(path),
            "fetched": {**snapshot["fetched"], _patterns(patterns): files},
            "downloaded": downloaded,
        }
        record["snapshots"][key] = snapshot
        write_json_atomic(cache_dir() / DOWNLOADS, record, indent=1)
