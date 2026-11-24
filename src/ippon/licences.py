"""A model's licence: what it is called, where to read it at the revision that runs, and whether running the model
asks the user to accept it, as one resolver (`resolve`) that a download (pull, from the Hub's metadata) and a loaded
model's provenance (`Result.diagnostics`, `/v1/models`, from its folder) both read; and the acceptances a person
gave, in licenses.json in ippon's settings.

The catalogue's licence names it, for a repository one of its builds is in (verified.py); else the model card's
`license` (Hugging Face's identifier, shown as verified.LICENCES names it), or for a custom licence (`other`) its
`license_name`. Standard licences share one acceptance; a custom one is accepted per URL it links, or per repository
and revision when it links a file of its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import time
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import LicenseNotAccepted
from .names import in_repository
from .storage import config_dir, locked, read_record, write_json_atomic
from .verified import LICENCES, OPEN

ACCEPTED = "licenses.json"  # in ippon's settings: the licences the user accepted, by key
LOCK = ACCEPTED + ".lock"  # beside it: held while one is added, so two acceptances at once keep both (`locked`)
WRITES = (ACCEPTED, LOCK)  # what this module writes in ippon's settings (`ippon remove all` takes them out)
FILES = frozenset({"license", "license.md", "license.txt", "licence"})  # a repository's own licence file, any case
UNSTATED = "not stated: read the model's page"
URL = re.compile(r"[a-z][a-z0-9+.-]*://")
# a model card's front matter, as huggingface_hub's repocard finds it (REGEX_YAML_BLOCK); that module isn't imported
# for it, as it takes 230 ms where yaml takes 17 (M5 Pro)
FRONT_MATTER = re.compile(r"^(\s*---(?:\r\n|\r|\n))([\S\s]*?)((?:\r\n|\r|\n)---[ \t]*(\r\n|\n|$))")
ACCEPT_ENV = "IPPON_ACCEPT_LICENSE"  # 1 accepts every licence a download needs, for CI, as conda's auto-accept does
_ACCEPT: ContextVar[bool] = ContextVar("accept_license", default=False)
_ASKED: ContextVar[str | None] = ContextVar("asked", default=None)  # the model as the caller named it, for hints
_RERUN: ContextVar[str | None] = ContextVar("rerun", default=None)  # the command typed, with --accept-license
_APPROVE: ContextVar[Callable[[Licence], bool] | None] = ContextVar("approve_license", default=None)


@contextmanager
def accepting(
    accept: bool = False,
    asked: str | None = None,
    rerun: str | None = None,
    approve: Callable[[Licence], bool] | None = None,
) -> Iterator[None]:
    """Licences accepted (`accept`) for what runs inside, and the model as its caller named it (`asked`: a short name
    resolves to a repository whose name the caller never typed), which a licence's hint repeats. `rerun` is the
    command a person typed with --accept-license added, which the hint gives instead of `ippon pull`. `approve` is an
    interactive terminal callback asked before downloading a custom-licence model when none was passed directly."""
    token, named = _ACCEPT.set(accept or _ACCEPT.get()), _ASKED.set(_ASKED.get() or asked)  # the outermost name
    typed = _RERUN.set(_RERUN.get() or rerun)
    asker = _APPROVE.set(_APPROVE.get() if approve is None else approve)
    try:
        yield
    finally:
        _APPROVE.reset(asker)
        _RERUN.reset(typed)
        _ASKED.reset(named)
        _ACCEPT.reset(token)


@dataclass(frozen=True)
class Licence:
    name: str
    url: str
    key: str  # what an acceptance is remembered by in licenses.json: the sha256 of the licence's identity
    repo: str
    open: bool = False  # a permissive open-source licence (verified.OPEN), which asks nothing of the user

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "url": self.url}

    @property
    def accepted(self) -> bool:
        try:
            record = json.loads((config_dir() / ACCEPTED).read_text())
            return isinstance(record, dict) and self.key in record
        except (OSError, ValueError):
            return False

    def accept(self) -> None:
        folder = config_dir()
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = folder / ACCEPTED
        with locked(path):
            record = read_record(path) or {}
            record[self.key] = {"name": self.name, "url": self.url, "accepted_at": time.time()}
            write_json_atomic(path, record, indent=2, mode=0o600)

    def require(self, model: str, accept: bool = False, approve: Callable[[Licence], bool] | None = None) -> None:
        """Nothing for an open licence or one accepted before; else accepted when the caller says so (the argument,
        IPPON_ACCEPT_LICENSE=1 or the user's answer to `approve`), and remembered. Otherwise LicenseNotAccepted."""
        if self.open or self.accepted:
            return
        said = os.environ.get(ACCEPT_ENV, "").strip().lower() in ("1", "true", "yes")
        asker = approve if approve is not None else _APPROVE.get()
        if accept or said or _ACCEPT.get() or (asker is not None and asker(self)):
            self.accept()
            return
        asked = _ASKED.get() or model
        rerun = _RERUN.get()
        if rerun:
            hint = f"to accept, run `{rerun}` (or set {ACCEPT_ENV}=1)"
        else:
            hint = (
                f"pass accept_license=True (for example `Decider.load({asked!r}, accept_license=True)`), "
                f"set {ACCEPT_ENV}=1, or run `ippon pull {shlex.quote(asked)} --accept-license`"
            )
        raise LicenseNotAccepted(
            f"{self.name}: acceptance is required before downloading weights. Read {self.url}; {hint}. "
            "No weights downloaded."
        )


def resolve(
    repo: str | None,
    revision: str | None,
    card: Mapping[str, Any],
    files: Collection[str],
    *,
    known: str = "",
    folder: Path | None = None,
) -> Licence:
    """The licence of Hub repository `repo` at `revision` (or of a folder of no repository), from its model card's
    front matter (`card`) and the files it holds (`files`, by path in it): `known` names it when the caller knows the
    build, else the catalogue's (`names.in_repository`), else the card. Its URL is the repository's own licence file at
    `revision`, else the card's license_link, else the repository's page; for a folder of no repository, its
    licence file or model card."""
    listed = in_repository(repo)
    known = known or (listed.licence if listed else "")
    published = str(card.get("license") or "")
    custom = str(card.get("license_name") or "") if published in ("", "other") else ""
    if custom and " " not in custom:  # a slug (kettei-community-model-license-1.0): its words, versions kept
        custom = " ".join(w if any(c.isdigit() for c in w) else w.capitalize() for w in custom.split("-"))
    name = known or custom or LICENCES.get(published, published) or UNSTATED
    link = str(card.get("license_link") or "")
    if known or (published and published != "other"):  # a standard licence, one acceptance wherever it is
        identity = (known or published).casefold().strip()
    else:  # a link to a file in the repository (LICENSE) is that repository's own licence: a URL can be shared
        identity = link if URL.match(link) else f"{repo}@{revision}/{link}"
    own = next((f for f in sorted(files) if Path(f).name.lower() in FILES), None) or (
        link if link and not URL.match(link) else None
    )
    page = f"https://huggingface.co/{repo}" if repo else ""
    if own and page:
        url = f"{page}/blob/{revision or 'main'}/{own}"
    elif own and folder is not None:
        url = str(folder / own)
    elif link:
        url = link
    elif page:
        url = f"{page}#license"
    else:
        url = str(folder / "README.md") if folder is not None and (folder / "README.md").is_file() else ""
    return Licence(name, url, hashlib.sha256(identity.encode()).hexdigest(), repo or "", identity in OPEN)


def of_metadata(metadata: Any) -> Licence:
    """The licence of a Hub repository at the commit its metadata (huggingface_hub's ModelInfo) resolved to."""
    raw = metadata.card_data
    card = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw or {})
    files = [s.rfilename for s in metadata.siblings or ()]
    return resolve(str(metadata.id), str(metadata.sha), card, files)


def card(folder: Path) -> dict[str, Any]:
    """A model card's front matter (the YAML at the top of README.md, between `---` lines), found and read as
    huggingface_hub's repocard reads it for the Hub's metadata, so a loaded model's licence is the one its download
    asked about; {} for a card without front matter, or with front matter that isn't a YAML mapping."""
    import yaml

    try:
        found = FRONT_MATTER.search((folder / "README.md").read_text(encoding="utf-8-sig", errors="replace"))
        read = yaml.safe_load(found[2]) if found else None
    except (OSError, yaml.YAMLError):
        return {}
    return read if isinstance(read, dict) else {}


def of_folder(folder: Path, repo: str | None, revision: str | None, known: str = "") -> Licence:
    """The licence of a model read from `folder`, the snapshot of Hub repository `repo` at `revision` when it is one;
    `known` as `resolve` takes it."""
    files = [p.name for p in folder.iterdir()] if folder.is_dir() else []
    return resolve(repo, revision, card(folder), files, known=known, folder=folder)
