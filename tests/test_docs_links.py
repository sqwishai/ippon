"""Every relative link and #anchor in the Markdown docs reaches a file and a heading that exist: the docs move
with the code, and a broken link on GitHub or PyPI is the first thing a reader meets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)\s]+)\)|(?:src|srcset|href)=\"([^\"]+)\"")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces as hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def prose(path: Path) -> str:
    return re.sub(r"```.*?```", "", path.read_text(), flags=re.S)


def test_relative_links_and_anchors_resolve() -> None:
    listed = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True)
    pages = [ROOT / f for f in listed.stdout.split() if not f.startswith("benchmarks/results/")]
    anchors = {p.resolve(): {slug(h) for h in re.findall(r"^#+ (.+)$", prose(p), flags=re.M)} for p in pages}
    broken = []
    for page in pages:
        for match in LINK.finditer(prose(page)):
            target = match.group(1) or match.group(2)
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                continue
            path, _, anchor = target.partition("#")
            resolved = (page.parent / path).resolve() if path else page.resolve()
            if not resolved.exists() or (anchor and resolved in anchors and anchor not in anchors[resolved]):
                broken.append(f"{page.relative_to(ROOT)}: {target}")
    assert not broken, "\n".join(broken)
