"""`ippon claude`, `ippon codex`, `ippon opencode` and `ippon pi`: a coding agent for one session with the local
decision model beside it, through ippon's MCP server, and nothing written to the agent's own configuration.

Each agent takes the session's additions its own way, as its documentation gives them.

- Claude Code reads `--mcp-config FILE`, and `--plugin-dir DIR` for a plugin holding ippon's skill (when a decision
  model fits and how to ask it).
- Codex reads `-c key=value` overrides of its config.toml: `mcp_servers.ippon`.
- OpenCode reads a whole config from `OPENCODE_CONFIG_CONTENT`, merged over the user's: the MCP server.
- Pi loads an extension with `-e FILE` (ippon's, `pi/ippon.ts`: the model as a Pi classifier, which codemode and
  extensions ask, and the MCP server) and a skill with `--skill DIR`.

`ippon setup` can instead give an agent the model in every session (`add_everywhere`, undone by `remove_everywhere`):
ippon's MCP server by the agent's own `mcp add` (OpenCode's and Pi's in their config files), ippon's skill linked into
its skills folder, and Pi's extension into its extensions folder. Each agent's session plan and these come from one
table (`KINDS`).

Codex and OpenCode get ippon's guidance as the MCP server's instructions. The server they reach is the shared one,
named by `IPPON_URL` and `IPPON_MODEL` in the agent's environment, which its MCP server inherits.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, get_args

from . import __version__, daemon, processes
from .errors import DecisionError, RuntimeUnavailable
from .served import CLIENT_ENV
from .storage import write_json_atomic

Agent = Literal["claude", "codex", "opencode", "pi"]
AGENTS: tuple[Agent, ...] = get_args(Agent)
# what a Codex MCP server is given of the environment beside Codex's own short list (HOME, PATH, ...): where ippon's
# cache and the Hub's are, and the server's key
PASSED_PREFIXES = ("IPPON_", "HF_", "HUGGINGFACE_")


@dataclass
class Plan:
    """How to start an agent for one session: its arguments, what its environment adds, and its files by path in the
    session's temporary directory, which the plan was made for (`run`)."""

    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    programs: dict[str, str] = field(default_factory=dict)  # scripts by name, first on the agent's PATH (`run`)


def _skill() -> dict[str, str]:
    """ippon's skill's files, by their paths in its folder."""
    root = files("ippon").joinpath("skills", "ippon")
    found = {"SKILL.md": root.joinpath("SKILL.md")}
    found |= {f"references/{f.name}": f for f in root.joinpath("references").iterdir() if f.name.endswith(".md")}
    return {path: f.read_text(encoding="utf-8") for path, f in found.items()}


def skill_files() -> dict[str, str]:
    """ippon's skill as a Claude Code plugin's files, by their paths in the plugin."""
    plugin = {
        "name": "ippon",
        "version": __version__,
        "description": "Ask a decision model on this Mac, through ippon",
        "author": {"name": "Harsh Pratap Singh"},
    }
    out = {".claude-plugin/plugin.json": json.dumps(plugin, indent=2)}
    return out | {f"skills/ippon/{path}": text for path, text in _skill().items()}


def _pi_extension() -> Path:
    return Path(str(files("ippon").joinpath("pi", "ippon.ts")))


def _toml(value: str | list[str]) -> str:
    """A string or a list of strings as an inline TOML value, as Codex's `-c` reads one. JSON's string escapes are
    TOML's, but for the surrogate pairs ASCII-only JSON writes for characters past U+FFFF, so those stay as they
    are."""
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")  # TOML's strings hold no raw DEL


def _opencode_server(mcp: list[str]) -> dict[str, Any]:
    """ippon's MCP server as an OpenCode config gives it, for a session and in the user's own config alike."""
    return {"type": "local", "command": mcp, "enabled": True}


def _pi_server(mcp: list[str]) -> dict[str, Any]:
    """ippon's MCP server as Pi's mcp.json gives it (Claude Code's `mcpServers` shape)."""
    return {"command": mcp[0], "args": mcp[1:]}


def _claude(mcp: list[str], args: Sequence[str], env: Mapping[str, str], folder: Path) -> Plan:
    servers = {"mcpServers": {"ippon": {"type": "stdio", "command": mcp[0], "args": mcp[1:]}}}
    made = Plan(
        ["claude", "--mcp-config", str(folder / "mcp.json"), "--plugin-dir", str(folder / "ippon-plugin")],
        files={"mcp.json": json.dumps(servers, indent=2)} | {f"ippon-plugin/{p}": t for p, t in skill_files().items()},
    )
    made.argv += args
    return made


def _codex(mcp: list[str], args: Sequence[str], env: Mapping[str, str], folder: Path) -> Plan:
    # Codex gives its MCP servers only the variables it is told to pass on: the session's own (the server's URL,
    # model and API key) as well as this environment's
    passed = sorted(name for name in {**os.environ, **env} if name.startswith(PASSED_PREFIXES))
    overrides: dict[str, str | list[str]] = {
        "mcp_servers.ippon.command": mcp[0],
        "mcp_servers.ippon.args": mcp[1:],
        "mcp_servers.ippon.env_vars": passed,
    }
    argv = ["codex", *(word for key, value in overrides.items() for word in ("-c", f"{key}={_toml(value)}")), *args]
    return Plan(argv)


def _opencode(mcp: list[str], args: Sequence[str], env: Mapping[str, str], folder: Path) -> Plan:
    # OpenCode 1.x reads `mcp.NAME` and `plugin` and its later versions still do, migrating them
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {"ippon": _opencode_server(mcp)},
    }
    made = Plan(["opencode", *args])
    made.env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    return made


def _pi(mcp: list[str], args: Sequence[str], env: Mapping[str, str], folder: Path) -> Plan:
    return Plan(
        ["pi", "-e", str(folder / "ippon.ts"), "--skill", str(folder / "skills" / "ippon"), *args],
        env={"IPPON_MCP": json.dumps(mcp)},  # the extension registers the MCP server with it
        files={"ippon.ts": _pi_extension().read_text(encoding="utf-8")}
        | {f"skills/ippon/{path}": text for path, text in _skill().items()},
    )


@dataclass(frozen=True)
class Config:
    """An agent's own JSON config file under HOME (`path`), whose `key` holds its MCP servers by name, each as
    `server` gives one."""

    path: str
    key: str
    server: Callable[[list[str]], dict[str, Any]]


@dataclass(frozen=True)
class Kind:
    """An agent ippon works with: its name, how to install it, how it gets the decision model for one session
    (`session`, the plan `ippon AGENT` runs it with), and how it gets it in every session (`add_everywhere`): ippon's
    MCP server added by the agent's own `mcp add` with `scope`, or written into its `config` file, ippon's skill
    linked into `skills`, a folder under HOME (None: it gets ippon's guidance as the MCP server's instructions), and
    Pi's extension linked into `extensions`."""

    name: str
    install: str
    session: Callable[[list[str], Sequence[str], Mapping[str, str], Path], Plan]
    scope: tuple[str, ...] = ()
    config: Config | None = None
    skills: str | None = None
    extensions: str | None = None


KINDS: dict[Agent, Kind] = {
    "claude": Kind(
        "Claude Code",
        "curl -fsSL https://claude.ai/install.sh | bash",
        _claude,
        scope=("--scope", "user"),
        skills=".claude/skills",
    ),
    "codex": Kind("Codex", "npm install -g @openai/codex", _codex, skills=".agents/skills"),
    "opencode": Kind(
        "OpenCode",
        "curl -fsSL https://opencode.ai/install | bash",
        _opencode,
        config=Config(".config/opencode/opencode.json", "mcp", _opencode_server),
    ),
    "pi": Kind(
        "Pi",
        "npm install -g @earendil-works/pi-coding-agent",
        _pi,
        config=Config(".pi/agent/mcp.json", "mcpServers", _pi_server),
        skills=".pi/agent/skills",
        extensions=".pi/agent/extensions",
    ),
}


def plan(
    agent: Agent,
    mcp: list[str],
    url: str,
    model: str,
    key: str | None,
    args: Sequence[str],
    folder: Path,
) -> Plan:
    """The plan that starts `agent` with `mcp` (the command of ippon's MCP server) asking the server at `url` about
    `model`, its files in `folder`, and `args` passed on to the agent."""
    env = {"IPPON_URL": url, "IPPON_MODEL": model, CLIENT_ENV: agent} | ({daemon.API_KEY_ENV: key} if key else {})
    made = KINDS[agent].session(mcp, args, env, folder)
    made.env |= env
    # the agent's own `ippon` commands run this installation, as its MCP server does, not another on PATH
    made.programs["ippon"] = f'#!/bin/sh\nexec {shlex.join(processes.command())} "$@"\n'
    return made


def mcp_command(model: str, options: Sequence[str]) -> list[str]:
    """The command that starts ippon's MCP server for an agent, asking `model` loaded with `options`."""
    return [*processes.command(), "mcp", model, *options]


def installed(agent: Agent) -> str:
    """The agent's command on PATH; RuntimeUnavailable saying how to install it when it is missing."""
    found, kind = shutil.which(agent), KINDS[agent]
    if found is None:
        raise RuntimeUnavailable(f"{kind.name} isn't installed ({agent} is not on PATH): {kind.install}")
    return found


def add_everywhere(agent: Agent, model: str, options: Sequence[str]) -> list[str]:
    """Give `agent` the decision model in every session: ippon's MCP server and ippon's skill, in its own settings
    (the user's), as its `Kind` says. What was added, as lines to show; RuntimeUnavailable, saying how to install it,
    for an agent that isn't installed, whose settings would be written for nothing."""
    installed(agent)
    kind, mcp = KINDS[agent], mcp_command(model, options)
    if kind.config is not None:
        path = Path.home() / kind.config.path
        config = _read_json(path)
        config.setdefault(kind.config.key, {})["ippon"] = kind.config.server(mcp)
        _write_json(path, config)
        added = [f"ippon's MCP server in ~/{kind.config.path}"]
    else:
        _run([agent, "mcp", "remove", *kind.scope, "ippon"])
        _run([agent, "mcp", "add", *kind.scope, "ippon", "--", *mcp], must=True)
        added = [f"ippon's MCP server ({agent} mcp list)"]
    if kind.skills is not None:
        _link(Path.home() / kind.skills / "ippon", _skill_folder())
        added.append(f"ippon's skill in ~/{kind.skills}")
    if kind.extensions is not None:
        _link(Path.home() / kind.extensions / "ippon.ts", _pi_extension())
        added.append(f"ippon's extension in ~/{kind.extensions}")
    return added


def remove_everywhere(agent: Agent) -> str | None:
    """Undo `add_everywhere`: ippon's MCP server, skill and extension out of the agent's settings, the rest
    untouched. What was taken out, as a line to show ("ippon's MCP server and skill taken out"), and what stayed when
    the agent's command, which alone edits its MCP servers, isn't on PATH; None when nothing was there."""
    kind = KINDS[agent]
    remove = None if kind.config is not None else [agent, "mcp", "remove", *kind.scope, "ippon"]
    if remove is None:
        server = kind.config is not None and _remove_from_config(kind.config)
    else:  # an agent's `mcp remove` succeeds for a server it doesn't have (Codex 0.154): only one it lists goes
        server = _run([agent, "mcp", "get", "ippon"]) and _run(remove)
    skill = kind.skills is not None and _unlink(Path.home() / kind.skills / "ippon", _skill_folder())
    extension = kind.extensions is not None and _unlink(Path.home() / kind.extensions / "ippon.ts", _pi_extension())
    parts = [part for part, done in (("MCP server", server), ("skill", skill), ("extension", extension)) if done]
    gone = " and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 2 else parts)
    if remove is not None and shutil.which(agent) is None:  # its MCP server, if added, is still in its settings
        kept = f"its MCP server stays, since `{agent}` isn't on PATH: run `{shlex.join(remove)}` once it is"
        return f"ippon's {gone} taken out, and {kept}" if gone else kept[0].upper() + kept[1:]
    return f"ippon's {gone} taken out" if gone else None


def _remove_from_config(config: Config) -> bool:
    """Whether ippon's MCP server was in the agent's config file, and is now taken out of it."""
    path = Path.home() / config.path
    if not path.exists():
        return False
    data = _read_json(path)
    if data.get(config.key, {}).pop("ippon", None) is None:
        return False
    _write_json(path, data)
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} isn't a JSON object, so ippon leaves it alone")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, data, indent=2)


def _skill_folder() -> Path:
    return Path(str(files("ippon").joinpath("skills", "ippon")))


def _link(link: Path, target: Path) -> None:
    """`link` made to point at ippon's `target` (its skill, Pi's extension), never over a file of the user's own."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to(target)


def _unlink(link: Path, target: Path) -> bool:
    """Whether `link` pointed at ippon's `target` (and is now gone); a file of the user's own is left alone."""
    if link.is_symlink() and Path(os.readlink(link)) == target:
        link.unlink()
        return True
    return False


def _run(argv: Sequence[str], must: bool = False) -> bool:
    """Whether an agent's command succeeded (False when it isn't installed). With `must`, a failure is an error that
    gives the command's own output, so setup never says it added what the agent refused."""
    try:
        done = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    except FileNotFoundError:
        if must:
            raise RuntimeUnavailable(
                f"`{argv[0]}` is not on this Mac's PATH, so ippon can't change its settings"
            ) from None
        return False
    if must and done.returncode:
        said = (done.stderr or done.stdout).strip()[-400:]
        raise DecisionError(f"`{' '.join(argv[:3])}` failed ({done.returncode}): {said}")
    return done.returncode == 0


def _at_terminal() -> bool:
    """Whether this process is in its terminal's foreground group, where Ctrl-C reaches the agent itself."""
    try:
        return os.tcgetpgrp(0) == os.getpgrp()
    except OSError:
        return False


def run(planned: Callable[[Path], Plan]) -> int:
    """Run the agent `planned` makes for a new temporary directory until it exits, its files there removed
    afterwards; its exit status (128 + n for an agent a signal n ended, as a shell reports it).

    Ctrl-C at a terminal reaches the agent directly, since it shares ippon's foreground group, so ippon ignores it and
    the agent decides what it means. SIGINT from elsewhere, SIGTERM and SIGHUP are passed on to the agent."""
    agent: list[subprocess.Popen[bytes]] = []
    received: list[int] = []

    def forward(signum: int, _frame: Any) -> None:
        if signum == signal.SIGINT and _at_terminal():
            return
        received.append(signum)
        if agent:
            agent[0].send_signal(signum)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with tempfile.TemporaryDirectory(prefix="ippon-agent-") as temporary:
            folder = Path(temporary).resolve()
            made = planned(folder)
            for name, text in made.files.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(text, encoding="utf-8")
            env = os.environ | made.env
            if made.programs:
                (folder / "bin").mkdir()
                for name, text in made.programs.items():
                    (folder / "bin" / name).write_text(text, encoding="utf-8")
                    (folder / "bin" / name).chmod(0o755)
                env["PATH"] = os.pathsep.join([str(folder / "bin"), env.get("PATH", "")])
            if received:  # stopped before the agent started
                return 128 + received[0]
            child = subprocess.Popen(made.argv, env=env)
            agent.append(child)
            if received:
                child.send_signal(received[0])
            status = child.wait()
            return 128 - status if status < 0 else status
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
