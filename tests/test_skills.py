"""The agent skills shipped in the package (src/ippon/skills) stay true to the command line they teach: every `ippon`
command they name exists, every reference a skill points to is there, and `ippon skills` prints each of them."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SKILLS = Path(__file__).resolve().parents[1] / "src/ippon/skills"


def ippon(*args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "ippon", *args], capture_output=True, text=True, timeout=60, check=True
    )
    return done.stdout


def test_the_skills_name_only_real_commands_and_references() -> None:
    listing = ippon("--help").split("commands:")[1].split("start here:")[0]
    commands = {name for pair in re.findall(r"^    ([a-z][\w-]*)(?: \(([\w-]+)\))?", listing, re.M) for name in pair}
    commands.discard("")
    assert {"decide", "run", "pull", "serve"} <= commands
    folders = sorted(p for p in SKILLS.iterdir() if (p / "SKILL.md").is_file())
    assert folders
    for folder in folders:
        skill = (folder / "SKILL.md").read_text()
        front = skill.split("---")[1]
        assert re.search(rf"^name: {folder.name}$", front, re.M), folder.name
        assert "description:" in front, folder.name
        assert ippon("skills", folder.name) == skill
        for file in [folder / "SKILL.md", *sorted((folder / "references").glob("*.md"))]:
            text = file.read_text()
            for command in set(re.findall(r"`ippon ([a-z-]+)", text)) - commands:
                ippon(command, "--help")  # one that runs, unlisted (check, prompt, ...), may be named too
            for name, reference in re.findall(r"`ippon skills ([a-z-]+) ([a-z-]+)`", text):
                assert (SKILLS / name / "references" / f"{reference}.md").is_file(), f"{file.name}: {reference}"
