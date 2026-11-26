"""`ippon setup` (onboarding.py, terminal.py): the model recommended for a Mac, what setup keeps for the commands after
it, what it adds to an agent's own settings and takes back out, and its questions where the cursor can't move."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import huggingface_hub.constants
import pytest

from ippon import agents, drift, names, onboarding, processes
from ippon.agents import AGENTS
from ippon.device import Device
from ippon.errors import RuntimeUnavailable
from ippon.terminal import Line, Terminal

GB = 2**30


def mac(memory: int, gpu: float) -> Device:
    return Device("Apple M2 Pro", "applegpu_g14s", memory * GB, int(gpu * GB), (27, 0), True, gpu_cores=16)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty HOME and ippon settings folder, and fake `claude`, `codex` and `pi` commands that log their
    arguments."""
    for folder in ("home", "config", "bin"):
        (tmp_path / folder).mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("IPPON_CONFIG", str(tmp_path / "config"))
    for agent in ("claude", "codex", "pi"):
        fake = tmp_path / "bin" / agent
        log = f"open({str(tmp_path / 'calls')!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')"
        fake.write_text(f"#!{sys.executable}\nimport sys\n{log}\n")
        fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin")
    return tmp_path / "home"


@pytest.mark.parametrize(
    ("memory", "gpu", "start", "fits"),
    [
        (48, 37.4, "kettei-core:bf16", "fits"),  # an M5 Pro: Kettei Core as published, with room to spare
        (16, 12, "kettei-core:bf16", "tight"),  # a 16 GB Mac: Kettei Core only just fits, and its description says so
        (8, 5.33, "kettei-dot:bf16", "tight"),  # an 8 GB M1 Air: Kettei Core is too big, so Dot, beside a reading
        (8, 4.6, "kettei-core:bf16", "too big"),  # less still: nothing fits, and the list starts on Core, saying so
    ],
)
def test_the_list_keeps_one_order_and_starts_on_the_first_model_this_mac_holds(
    memory: int, gpu: float, start: str, fits: str
) -> None:
    """Sqwish's models lead on every Mac, each at the precision it is published at, with no model marked as the one
    to take; the cursor starts on the first that isn't too big here, and its description says how it fits."""
    device = mac(memory, gpu)
    listed = onboarding.ordered(onboarding.choices(device))
    assert [c.named.name for c in listed[:2]] == ["kettei-core:bf16", "kettei-dot:bf16"]
    assert all(c.precision is None for c in listed)  # the user turns a precision, setup doesn't
    at = onboarding.first_usable(listed, device)
    assert listed[at].named.name == start
    assert onboarding.fit(listed[at], device) == fits
    said = {"fits": "with room to spare", "tight": "only just fits", "too big": "too big for this Mac"}
    assert said[fits] in onboarding.describe(listed[at], device)[-1]
    if memory == 8:
        assert said["too big"] in onboarding.describe(listed[0], device)[-1]  # Core leads, and says it won't fit
    assert not any(line.note == "recommended" for line in onboarding.lines(listed, device))
    assert len(listed) == len(onboarding.choices(device))  # the ones too big for this Mac are listed all the same


def test_setup_without_a_terminal_takes_core_as_published_and_leaves_agents_alone(home: Path) -> None:
    """With nobody to turn a precision (--yes, an agent), setup takes the first model that fits as its creator
    published it: Kettei Core at full precision on a 16 GB Mac, where it only just fits. Never 8-bit unasked."""
    pulled: list[tuple[str, object]] = []
    sampled: list[tuple[str, Sequence[str]]] = []

    def pull(name: str, approve: object, progress: object) -> int:
        pulled.append((name, approve))
        return 0

    def sample(name: str, options: Sequence[str]) -> str:
        sampled.append((name, options))
        return "answered"

    steps = onboarding.Steps(pull, sample)
    settings = onboarding.run(None, mac(16, 12), steps)
    assert pulled == [("kettei-core:bf16", None)]  # no terminal: no licence question, the licence must be accepted
    assert sampled == [("kettei-core:bf16", [])]
    assert (settings.model, settings.precision, settings.serving, settings.everywhere) == (
        "kettei-core:bf16",
        None,
        "on demand",
        [],
    )
    assert onboarding.chosen_model() == ("kettei-core:bf16", None)  # what ippon claude and start take
    assert not any(home.iterdir()), "wrote into HOME"


def test_every_session_comes_out_as_it_went_in(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every-session setup adds ippon's MCP server and skill beside what the user has, and `remove` takes out only
    those, never a skill of the user's own."""
    opencode = home / ".config" / "opencode" / "opencode.json"
    opencode.parent.mkdir(parents=True)
    theirs = {"theme": "dark", "mcp": {"other": {"type": "remote", "url": "https://example.com"}}}
    opencode.write_text(json.dumps(theirs))
    own_skill = home / ".agents" / "skills" / "ippon"
    own_skill.mkdir(parents=True)
    opencode_command = tmp_path / "bin" / "opencode"  # OpenCode installed too
    opencode_command.write_text("#!/bin/sh\n")
    opencode_command.chmod(0o755)
    for agent in AGENTS:
        agents.add_everywhere(agent, "kettei-core", ["--precision", "q8"])
    added = json.loads(opencode.read_text())["mcp"]["ippon"]
    assert added["command"][-3:] == ["kettei-core", "--precision", "q8"]
    pi = json.loads((home / ".pi" / "agent" / "mcp.json").read_text())["mcpServers"]["ippon"]
    assert pi["args"][-3:] == ["kettei-core", "--precision", "q8"]
    assert (home / ".pi" / "agent" / "extensions" / "ippon.ts").read_text().startswith("// ippon in Pi")
    assert (home / ".claude" / "skills" / "ippon" / "SKILL.md").is_file()
    assert not own_skill.is_symlink()  # the user's own, left in place
    calls = (tmp_path / "calls").read_text().splitlines()
    assert any(
        c.startswith("mcp add --scope user ippon -- ") and c.endswith("mcp kettei-core --precision q8") for c in calls
    )
    removed = {agent: agents.remove_everywhere(agent) for agent in AGENTS}  # what each really had from ippon
    taken = "ippon's MCP server taken out"
    assert removed == {
        "claude": "ippon's MCP server and skill taken out",
        "codex": taken,
        "opencode": taken,
        "pi": "ippon's MCP server, skill and extension taken out",
    }
    assert not (home / ".pi" / "agent" / "extensions" / "ippon.ts").exists()
    agents.add_everywhere("claude", "kettei-core", [])
    (tmp_path / "bin" / "claude").unlink()  # Claude Code gone: only its own command edits its MCP servers
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin")
    said = str(agents.remove_everywhere("claude"))
    assert said.startswith("ippon's skill taken out")
    assert "MCP server stays" in said
    assert json.loads(opencode.read_text()) == theirs
    assert not (home / ".claude" / "skills" / "ippon").exists()
    assert own_skill.is_dir()
    assert "mcp remove ippon" in (tmp_path / "calls").read_text()


def test_left_and_right_turn_a_model_through_its_precisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-precision build turns through full, 8, 6 and 4 bits and stops at either end, its memory falling with
    the bits and a tag on how far its answers move, a warning from 6 bits down; a published quantized build has only
    itself. In a terminal 100 columns wide every row fits."""
    device = mac(48, 37.4)
    listed = onboarding.ordered(onboarding.choices(device))
    at = onboarding.first_usable(listed, device)
    monkeypatch.setenv("COLUMNS", "100")
    drawn = Terminal(io.StringIO(), io.StringIO())._rows(onboarding.lines(listed, device), at, None, 0, len(listed))
    rows = [re.sub(r"\x1b\[[0-9;]*m", "", row) for row in drawn]
    assert max(len(row) for row in rows) < 100
    core = next(c for c in onboarding.choices(device) if c.named.name == "kettei-core:bf16")
    assert core.precision is None  # full precision, with room to spare
    turned = [core.turned(-1)]
    for _ in range(4):
        turned.append(turned[-1].turned(1))
    assert [c.precision for c in turned] == [None, "q8", "q6", "q4", "q4"]
    assert [round(c.gigabytes, 1) for c in turned[:4]] == [7.6, 4.0, 3.1, 2.1]  # as `ippon pull --dry-run` gives
    tags = [onboarding.tag(c) for c in turned[:4]]  # as measured: Core moves noticeably at 6 bits, a lot at 4
    assert tags == [None, drift.TAG["close"], drift.TAG["noticeable"], drift.TAG["a lot"]]
    assert any(line.startswith("⚠") and drift.SAYS["a lot"] in line for line in onboarding.describe(turned[3], device))
    gguf = onboarding.Choice(names.find("d1-2b:q8_0")[0])
    assert gguf.steps == [gguf]  # a published build alone has only itself
    assert gguf.turned(1) == gguf
    # a family with another verified build steps on to it in memory's order: d1-2B's checkpoint, Liquid's Q8_0 in
    # place of the checkpoint at 8 bits (`d1-2b:q8` loads that file), then 6 and 4 bits, each reading what d1 reads
    d1 = next(c for c in onboarding.choices(device) if c.named.family == "d1-2b")
    shown = [(c.name, onboarding.form(c), round(c.gigabytes, 1)) for c in d1.steps]
    assert shown == [
        ("d1-2b:bf16", "◂ full ▸", 3.8),
        ("d1-2b:q8_0", "◂ Q8_0 ▸", 1.8),
        ("d1-2b:q6", "◂ 6-bit ▸", 1.5),
        ("d1-2b:q4", "◂ 4-bit ▸", 1.1),
    ]
    assert {" · ".join(c.reads) for c in d1.steps} == {"text · images · video"}


def test_questions_are_numbered_where_the_cursor_cannot_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """On TERM=dumb a question is a numbered list: Enter takes the default, a number another line, and a
    checklist takes the numbers to tick."""
    monkeypatch.setenv("TERM", "dumb")
    lines = [Line("Kettei Core"), Line("Kettei Dot"), Line("d1-2B")]
    asked = Terminal(io.StringIO("\n3\nx\n9\n2\n1 3\n"), io.StringIO())
    assert asked.pick("Choose", lines, 0) == 0
    assert asked.pick("Choose", lines, 0) == 2
    assert asked.pick("Choose", lines, 1, keys=(("x", "every build"),)) == "x"
    assert asked.pick("Choose", lines, 1) == 1  # 9 is no line: asked again, then 2
    assert asked.checklist("Use it with", lines, [False, True, False]) == [True, False, True]


def test_setup_goes_back_to_the_question_before(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """b goes back a question: out of "Use it with" to the model's choice (downloaded again only if needed), and out
    of "Keep ippon running?" to how the agents get it, whose new answer is the one setup keeps."""
    monkeypatch.setenv("TERM", "dumb")
    pulled: list[str] = []

    def pull(name: str, approve: object, progress: object) -> int:
        pulled.append(name)
        return 0

    # model: default · uses: back · model: default · uses: as ticked · how: default · serving: back · how: 2 · serving
    asked = Terminal(io.StringIO("\nb\n\n\n\nb\n2\n\n"), io.StringIO())
    settings = onboarding.run(asked, mac(16, 12), onboarding.Steps(pull, lambda name, options: "answered"))
    assert pulled == ["kettei-core:bf16", "kettei-core:bf16"]  # as published: a person turns the precision
    assert sorted(settings.everywhere) == ["claude", "codex", "pi"]
    assert settings.serving == "on demand"


def test_ippon_started_in_a_repository_never_runs_its_code(tmp_path: Path) -> None:
    """An agent starts ippon's MCP server, and the server, in whatever folder it has open: a json.py there must not
    run, neither when ippon starts nor when the server asks a child process about this Mac."""
    planted = tmp_path / "ran"
    (tmp_path / "json.py").write_text(f"open({str(planted)!r}, 'w').write('ran')\n")
    subprocess.run([*processes.command(), "--version"], cwd=tmp_path, capture_output=True, timeout=120, check=True)
    probe = "import ippon.mcp as m; m._device()"
    subprocess.run([sys.executable, "-P", "-c", probe], cwd=tmp_path, capture_output=True, timeout=120, check=False)
    assert not planted.exists()


def test_remove_all_takes_out_everything_ippon_added_and_nothing_else(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ippon remove all --models`: ippon out of every agent's settings, and of folders other programs share only what
    is ippon's: its own files where IPPON_CACHE and IPPON_CONFIG point, and of the Hub cache the revisions ippon
    downloaded, never another revision of the same repository, a model ippon only reused or another tool's model."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("IPPON_CACHE", str(cache))
    hub_cache = tmp_path / "hub"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub_cache))

    def snapshot(repo: str, revision: str, content: str) -> Path:
        folder = hub_cache / f"models--{repo.replace('/', '--')}"
        blob = folder / "blobs" / content
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(content)
        path = folder / "snapshots" / revision
        path.mkdir(parents=True)
        (path / "config.json").symlink_to(blob)
        (folder / "refs").mkdir(exist_ok=True)
        return path

    ours = snapshot("Sqwish/Kettei-Core", "a" * 40, "ours")
    older = snapshot("Sqwish/Kettei-Core", "c" * 40, "older")  # another revision another tool fetched
    reused = snapshot("Sqwish/Kettei-Dot", "d" * 40, "reused")  # here before ippon, which only read it
    theirs = snapshot("someone/other-model", "b" * 40, "theirs")
    cache.mkdir()
    (cache / "other-tool.db").write_text("not ippon's")
    (home.parent / "config" / "app-settings.json").write_text("not ippon's")
    snapshots = {
        "Sqwish/Kettei-Core@" + "a" * 40: {
            "path": str(ours),
            "fetched": {"*.json": ["config.json"]},
            "downloaded": True,
        },
        "Sqwish/Kettei-Dot@" + "d" * 40: {
            "path": str(reused),
            "fetched": {"*.json": ["config.json"]},
            "downloaded": False,
        },
    }
    (cache / "downloads.json").write_text(json.dumps({"snapshots": snapshots}))
    opencode_command = tmp_path / "bin" / "opencode"  # OpenCode installed too
    opencode_command.write_text("#!/bin/sh\n")
    opencode_command.chmod(0o755)
    for agent in AGENTS:
        agents.add_everywhere(agent, "kettei-core", [])
    onboarding.Settings.path().write_text('{"everywhere": ["claude"')  # cut short: read as nothing chosen
    assert onboarding.Settings.load() == onboarding.Settings()
    onboarding.Settings(model="kettei-core:q8").save()  # set aside first, as it was, then written afresh
    unreadable = onboarding.Settings.path().with_name(onboarding.SETUP_UNREADABLE)
    assert unreadable.read_text() == '{"everywhere": ["claude"'
    done = onboarding.remove_all(models=True)
    assert not unreadable.exists()
    assert not (home / ".claude" / "skills" / "ippon").exists()
    assert "ippon" not in json.loads((home / ".config" / "opencode" / "opencode.json").read_text()).get("mcp", {})
    assert not ours.exists()
    assert (older / "config.json").read_text() == "older"
    assert reused.exists()
    assert theirs.exists()
    assert not (cache / "downloads.json").exists()
    assert (cache / "other-tool.db").exists()
    assert not onboarding.Settings.path().exists()
    assert (home.parent / "config" / "app-settings.json").exists()
    assert any(line.startswith("Sqwish/Kettei-Core: 1 revision(s) deleted") for line in done)


def test_setup_refuses_an_agent_that_is_not_installed(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`setup --use opencode` without OpenCode says how to install it, and writes none of its settings."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    with pytest.raises(RuntimeUnavailable, match="OpenCode isn't installed"):
        agents.add_everywhere("opencode", "kettei-core", [])
    assert not (home / ".config" / "opencode").exists()
