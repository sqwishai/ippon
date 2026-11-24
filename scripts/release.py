"""Set ippon's version everywhere it is written, for a release or for the development that follows one.

    uv run python scripts/release.py 0.1.0rc1       # a pre-release: the version and uv.lock; CHANGELOG.md unchanged
    uv run python scripts/release.py 0.1.0          # the release: the version, uv.lock, and ## Unreleased dated as it
    uv run python scripts/release.py 0.2.0.dev0     # after it: the next .dev0 version and a new, empty ## Unreleased
    uv run python scripts/release.py --notes 0.1.0  # print a version's notes: its section, or ## Unreleased's for a
                                                    # pre-release (the release workflow's GitHub release text)

The version lives in pyproject.toml and src/ippon/__init__.py (a test holds them equal) and in uv.lock, which
`uv lock` rewrites. A release needs a non-empty ## Unreleased section; docs/releasing.md has the rest.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?")  # PEP 440's forms this project uses
UNRELEASED = "## Unreleased\n"
PRERELEASE = re.compile(r"(?:a|b|rc)\d+$")


def replace_once(path: Path, pattern: str, new: str) -> None:
    text = path.read_text()
    changed, count = re.subn(pattern, new, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"{path.relative_to(ROOT)}: no line matches {pattern!r}")
    path.write_text(changed)


def section(text: str, heading: str) -> str:
    """The body of the CHANGELOG section whose heading line starts with `heading`, or "" when there is none."""
    for part in text.split("\n## ")[1:]:
        if part.startswith(heading.removeprefix("## ")):
            return part.partition("\n")[2].strip()
    return ""


def notes(version: str) -> str:
    """A version's release notes: its dated section, or ## Unreleased's for a pre-release."""
    text = (ROOT / "CHANGELOG.md").read_text()
    body = section(text, UNRELEASED.strip() if PRERELEASE.search(version) else f"## {version} (")
    if not body:
        raise SystemExit(f"CHANGELOG.md has no notes for {version}")
    return body


def changelog(version: str, today: dt.date) -> None:
    path = ROOT / "CHANGELOG.md"
    text = path.read_text()
    if PRERELEASE.search(version):  # a pre-release of the coming version: its notes stay under ## Unreleased
        notes(version)
        return
    if ".dev" in version:  # the development after a release: a new, empty section on top
        if UNRELEASED in text:
            raise SystemExit("CHANGELOG.md already has ## Unreleased")
        path.write_text(text.replace("# Changelog\n\n", f"# Changelog\n\n{UNRELEASED}\n", 1))
        return
    head, found, rest = text.partition(UNRELEASED)
    body = rest.split("\n## ", 1)[0]
    if not found or not body.strip():
        raise SystemExit("CHANGELOG.md has no ## Unreleased section with entries to release")
    path.write_text(f"{head}## {version} ({today.day} {today:%B %Y})\n{rest}")


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--notes" and VERSION.fullmatch(args[1]):
        print(notes(args[1]))
        return
    if len(args) != 1 or not VERSION.fullmatch(args[0]):
        raise SystemExit(__doc__)
    version = args[0]
    replace_once(ROOT / "pyproject.toml", r'^version = "[^"]+"$', f'version = "{version}"')
    replace_once(ROOT / "src/ippon/__init__.py", r'^__version__ = "[^"]+"$', f'__version__ = "{version}"')
    changelog(version, dt.date.today())
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    print(f"ippon {version}: pyproject.toml, src/ippon/__init__.py, uv.lock and CHANGELOG.md updated")


if __name__ == "__main__":
    main()
