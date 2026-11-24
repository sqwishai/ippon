"""docs/models.md's section for each model and the builds ippon loads by name, from their one record
(src/ippon/verified.py), written between its markers.

    uv run python scripts/models_table.py docs/models.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from ippon.verified import MODELS

START, END = "<!-- builds: scripts/models_table.py -->", "<!-- /builds -->"


def table() -> str:
    """A section per model, so that each has its own link (docs/models.md#d1-2b): how to pull it, where its creator
    publishes it, its licence, and every build with what was checked of it."""
    sections = []
    for model in MODELS:
        first = model.builds[0]
        family, repo = first.name.partition(":")[0], first.load.partition("@")[0]
        rows = [f"| `{b.load}` | {b.weights} | {b.answers} |" for b in model.builds]
        sections.append(
            f"### {model.name}\n\n"
            f"`ippon pull {family}` downloads the build that `ippon pull --list` stars. Its creator publishes it at\n"
            f"[{repo}](https://huggingface.co/{repo}), and its licence is {model.licence}.\n\n"
            "| Load | Weights | Checked |\n|---|---:|---|\n" + "\n".join(rows)
        )
    return "\n\n".join(sections)


def main() -> None:
    path = Path(sys.argv[1])
    page = path.read_text()
    head, _, rest = page.partition(START)
    _, _, tail = rest.partition(END)
    if not rest:
        raise SystemExit(f"{path}: no {START} ... {END} block")
    path.write_text(head + START + "\n" + table() + "\n" + END + tail)


if __name__ == "__main__":
    main()
