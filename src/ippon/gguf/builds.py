"""The GGUF builds ippon loads, as the catalogue pins them (verified.py: each build's `gguf`): the file (repository,
revision, name, sha256 and header digest) and the creator's repository at the revision whose small files (config,
tokenizer, templates; never weights) describe it.

A GGUF carries a tokenizer and some settings of its own, but not in the form the creator's code reads them, and its
base-model metadata names the pre-trained model it was fine-tuned from, not the decision model. So each build is
tied to the creator's checkpoint by hand, and its metadata is checked against that checkpoint's configuration when it
loads (`source.bound`). A local .gguf file is recognized by its header digest.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import UnsupportedModel
from ..names import parse, split
from ..quantization import unsupported_format
from ..verified import MODELS

SMALL_FILES = ("*.json", "*.txt", "*.jinja")  # the creator's files a build is read with, from its base repository


@dataclass(frozen=True)
class Build:
    """A GGUF build as its readers take it: its file, and the creator's repository and revision it is read with."""

    repo: str
    revision: str
    file: str
    sha256: str
    header: str  # sha256 of the header's bytes (`header.Header.digest`)
    base: str  # the creator's repository and revision: config.json, tokenizer, templates
    base_revision: str
    mmproj: str = ""  # its picture and audio parts, a file of their own beside it (llama.cpp's --mmproj), if any
    mmproj_sha256: str = ""
    mmproj_header: str = ""

    @property
    def quant(self) -> str:
        return self.file.removesuffix(".gguf").rsplit("-", 1)[-1].upper()


def _catalogued() -> tuple[Build, ...]:
    out = []
    for model in MODELS:
        base, pinned = split(model.builds[0].load)  # the creator's checkpoint, at the revision it is checked at
        for b in model.builds:
            if (g := b.gguf) is not None:
                out.append(
                    Build(
                        parse(b.load).repo,
                        g.revision,
                        g.file,
                        g.sha256,
                        g.header,
                        base,
                        g.base_revision or str(pinned),
                        g.mmproj,
                        g.mmproj_sha256,
                        g.mmproj_header,
                    )
                )
    return tuple(out)


BUILDS = _catalogued()
READ = tuple(sorted({b.quant for b in BUILDS}))  # the file quantizations ippon reads: those of its builds


def find(spec: str) -> Build:
    """The build a "repo:QUANT" or "repo@revision:QUANT" spec names (QUANT as in the file's name: Q8_0)."""
    reference = parse(spec)
    if reference.form != "gguf":
        raise UnsupportedModel(f"{spec!r} is not a GGUF build: name one as repo:Q8_0")
    repo, revision, quant = reference.repo, reference.revision or "", reference.quant
    if quant.upper() not in READ:
        alternative = next(
            (f"{b.repo}:Q8_0" for b in BUILDS if b.repo.lower() == repo.lower()), "a Q8_0 or full-precision build"
        )
        raise unsupported_format(spec, alternative)
    matches = [
        b
        for b in BUILDS
        if b.repo.lower() == repo.lower() and b.quant == quant.upper() and revision in ("", b.revision)
    ]
    if not matches:
        known = ", ".join(f"{b.repo}:{b.quant}" for b in BUILDS)
        raise UnsupportedModel(f"{spec!r} is not a GGUF build ippon knows: {known}")
    return matches[0]


def by_header(digest: str) -> Build:
    """The build whose header is `digest` (a local file)."""
    for b in BUILDS:
        if b.header == digest:
            return b
    known = ", ".join(f"{b.repo}/{b.file}" for b in BUILDS)
    raise UnsupportedModel(
        f"a GGUF file ippon doesn't know (header {digest[:12]}): it loads these builds, as published: {known}"
    )
