from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shlex
import shutil
import sys
from importlib.resources import files
from pathlib import Path

import pytest

from ippon import cli, names
from ippon.cli import main, readable


def test_check_reports_metal_initialization_failure_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ippon import check, cli
    from ippon.errors import RuntimeUnavailable

    def unavailable(*args: object, **options: object) -> dict[str, object]:
        raise ImportError("[metal::load_device] No Metal device available")

    monkeypatch.setattr(check, "check", unavailable)
    assert cli.main(["check", "ollama://d3:4b-mxfp8", "--json"]) == RuntimeUnavailable.exit_code
    captured = capsys.readouterr()
    assert "ippon doctor" in captured.err
    assert json.loads(captured.out)["error"]["type"] == "runtime_unavailable"


ALL_COMMON = [
    "--profile", "m4", "--switch", "add_norm=mlx", "--precision", "fp32", "--chunk", "512", "--keep-awake", "2",
    "--prefix-budget", "1G", "--literal-state", "--truncation", "error",
    "--window", "65536", "--answer-cache", "64",
]  # fmt: skip


def test_every_common_option_reaches_the_load_here_and_in_another_process() -> None:
    """Each common option is one of Decider.load's, set from the command line, and forwarded to another process
    (ippon's arm of a comparison) it loads the same: a command that loads a model can't drop one silently
    (verify once ignored --chunk and --prefix-budget)."""
    import argparse
    import inspect

    from ippon.cli import common_options, forward_options, load_options
    from ippon.decider import Decider

    parser = argparse.ArgumentParser(parents=[common_options()])
    args = parser.parse_args(ALL_COMMON)
    options = load_options(args)
    assert set(options) <= set(inspect.signature(Decider.load).parameters)
    assert options.pop("freeze_objects") is True  # not an option: the command line owns its process
    assert options == {
        "profile": "m4",
        "switches": "add_norm=mlx",
        "precision": "fp32",
        "chunk": 512,
        "keep_awake": 2.0,
        "prefix_budget": 1 << 30,
        "literal_state": True,
        "truncation": "error",
        "window": 65536,
        "accept_license": False,
        "answer_cache": 64,
    }
    dests = {a.dest for a in common_options()._actions if a.option_strings} - {"settings"}
    assert len(dests) == len(options)  # every option of the parser has its entry
    assert load_options(parser.parse_args(forward_options(args))) == {**options, "freeze_objects": True}
    assert forward_options(parser.parse_args([])) == []  # defaults are not repeated
    assert load_options(parser.parse_args(["--truncation", "creator"]))["truncation"] == "creator"


@pytest.mark.parametrize(
    ("given", "code", "says"),
    [
        ("request.json", 4, "no installed adapter matches"),  # read from the file, then the model refused
        ('{"state": "x", "questions": {"q": {"type": "noul"}}}', 4, "no installed adapter matches"),  # JSON text
        ('{"state": "x", "questions": ', 3, "the request given is not JSON"),
        ("missing.json", 3, "can't read missing.json"),
        ("empty.json", 3, "no request in empty.json"),
        ("binary.json", 3, "UTF-8"),
    ],
)
def test_run_reads_a_request_from_a_file_or_json_text_and_exits_with_the_errors_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], given: str, code: int,
    says: str,
) -> None:  # fmt: skip
    """A request that isn't one is refused (InvalidRequest, 3) before any model loads, saying what to give; a model
    no adapter runs is UnsupportedModel (4)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "request.json").write_text(json.dumps({"questions": {"q": {"type": "noul"}}}))
    (tmp_path / "empty.json").write_text("  \n")
    (tmp_path / "binary.json").write_bytes(b"\xff\xfe")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unknown"}))
    assert main(["run", str(tmp_path), given]) == code
    assert says in capsys.readouterr().err


def test_decide_adds_the_files_its_media_flags_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--image, --video and --audio add each file's base64 after the request's own media; a file that can't be read
    is refused (InvalidRequest, 3) before any model loads."""
    import base64

    from ippon.cli import with_files

    photo, clip = tmp_path / "photo.png", tmp_path / "clip.wav"
    photo.write_bytes(b"png bytes")
    clip.write_bytes(b"wav bytes")
    args = argparse.Namespace(image=[str(photo)], video=None, audio=[str(clip), str(clip)])
    body = with_files({"state": "s", "images": ["first"], "audio": "held"}, args)
    assert body["images"] == ["first", base64.b64encode(b"png bytes").decode()]
    assert body["audio"] == ["held", *[base64.b64encode(b"wav bytes").decode()] * 2]
    assert "videos" not in body
    request = json.dumps({"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    assert main(["decide", str(tmp_path), request, "--image", str(tmp_path / "missing.png")]) == 3
    assert "can't read" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command",
    [
        ["verify", "MODEL", "--reference", "missing.json"],
        ["verify", "MODEL", "--reference", "request.json"],  # JSON, not a reference
        ["verify", "MODEL", "--invariance", "--requests", "broken.json"],
        ["verify", "MODEL"],
        ["bench", "MODEL", "--sizes", "100,many"],
        ["decide", "MODEL", "request.json", "--chunk", "0"],
        ["decide", "MODEL", "request.json", "--window", "-1"],
        ["decide", "MODEL", "request.json", "--answer-cache", "-1"],
        ["decide", "MODEL", "request.json", "--prefix-budget=-1"],
        ["decide", "MODEL", "request.json", "--keep-awake", "nan"],
        ["serve", "MODEL", "--detach", "--keep-awake", "inf"],
        ["serve", "MODEL", "--keep-awake", "-1"],
        ["decide", "MODEL", "request.json", "--states", "binary.jsonl"],
        ["eval", "missing.jsonl", "--model", "MODEL"],
        ["eval", "binary.jsonl", "--model", "MODEL"],
    ],
)
def test_a_misused_command_exits_3_with_invalid_request_before_any_model_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: list[str]
) -> None:
    """Options that can't be used are the caller's to change (InvalidRequest), not a crash or a failure of ippon's."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "request.json").write_text(json.dumps({"state": "x", "questions": {"q": {"type": "noul"}}}))
    (tmp_path / "broken.json").write_text("[{")
    (tmp_path / "binary.jsonl").write_bytes(b"\xff\xfe")
    command = [str(tmp_path) if part == "MODEL" else part for part in command]
    as_json = command[0] != "bench"
    assert main(command + (["--json"] if as_json else [])) == 3
    captured = capsys.readouterr()
    assert captured.err.startswith("ippon: ")  # what to change, not a traceback
    if as_json:
        assert json.loads(captured.out)["error"]["type"] == "invalid_request"


def test_every_listed_command_explains_itself_and_its_examples_use_options_it_takes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`ippon COMMAND --help` says what the command does and shows it in use, and an example can't outlive an option:
    each is a use of its command with only options that help lists. An option ippon doesn't take is named in the
    error without the words typed beside it, since a prompt among them can hold a secret."""
    for command in cli.HELP_ORDER:
        with pytest.raises(SystemExit):
            main([command, "--help"])
        text = capsys.readouterr().out
        assert text.split("\n\n")[1][0].isupper(), command  # what it does, under its usage
        for typed, _ in cli.EXAMPLES[command]:
            assert typed in text, typed
            ours = typed[typed.index(f"ippon {command}") :].split(" -- ")[0]  # what follows -- is the agent's own
            for option in (word for word in shlex.split(ours) if re.fullmatch(r"--?[a-z][a-z-]*", word)):
                assert re.search(rf"(?<![\w-]){option}(?![\w-])", text), typed
    with pytest.raises(SystemExit) as misuse:
        main(["status", "--bogus", "a secret"])
    error = capsys.readouterr().err
    assert misuse.value.code == 2
    assert "--bogus isn't an option it takes" in error
    assert "secret" not in error


def test_info_and_doctor_print_json_for_scripts(capsys: pytest.CaptureFixture[str]) -> None:
    """info's text and --json report the same facts (built once); doctor's --json lists its checks."""
    assert main(["info", "--json"]) == 0
    facts = json.loads(capsys.readouterr().out)
    assert {"ippon", "mlx", "chip", "profile", "switches", "kernels", "adapters"} <= set(facts)
    assert main(["info"]) == 0
    text = capsys.readouterr().out
    for shown in (facts["mlx"], facts["macos"], f"profile: {facts['profile']['name']}", *facts["adapters"]):
        assert shown in text
    code = main(["doctor", "--json"])
    checks = json.loads(capsys.readouterr().out)
    assert all(set(c) == {"status", "what", "fix"} for c in checks)
    assert code == (1 if any(c["status"] == "fail" for c in checks) else 0)


def test_each_kind_of_error_has_its_own_exit_code_and_the_documentation_says_which() -> None:
    """Distinct codes, so a script tells the kinds apart, never 0 (success) or 2 (argparse's misused command line);
    errors.py and docs/errors.md name each kind with the code the command line returns."""
    import re

    from ippon import errors

    kinds: list[type[errors.DecisionError]] = [  # each that declares a code: the others are a server's only
        c
        for c in vars(errors).values()
        if isinstance(c, type) and issubclass(c, errors.DecisionError) and "exit_code" in vars(c)
    ]
    codes = {c.exit_code for c in kinds}
    assert len(codes) == len(kinds)
    assert not codes & {0, 2}
    assert all(re.search(rf"^ +{c.exit_code} +{c.kind} ", errors.__doc__ or "", re.MULTILINE) for c in kinds)
    documented = (Path(__file__).resolve().parents[1] / "docs/errors.md").read_text()
    exit_codes = documented.split("## CLI exit codes\n", 1)[1].split("\n## ", 1)[0]
    rows = dict(re.findall(r"^\| (\d+) \| `(\w+)` \|", exit_codes, re.MULTILINE))
    assert {c.kind: str(c.exit_code) for c in kinds} == {kind: code for code, kind in rows.items()}


def test_the_package_and_its_metadata_name_one_version() -> None:
    """`ippon --version` and the version a release is tagged and published as are written in two places."""
    import tomllib

    from ippon import __version__

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == __version__


def test_a_short_name_loads_its_build_and_run_is_decide(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`decide` (and `run`, the same command) and Decider.load take a model's short name for its build (verified.py),
    a bare family name for its recommended build; a Hub id loads as given; a name that is none is refused, naming
    the closest."""
    from ippon import decider, hub, locate
    from ippon.errors import UnsupportedModel
    from test_decider import REQUEST
    from test_serve import toy

    loaded: list[str] = []

    def load(model: str, **options: object) -> decider.Decider:
        loaded.append(model)
        return toy()

    monkeypatch.setattr(decider.Decider, "load", load)
    assert main(["decide", "d1-2b", json.dumps(REQUEST)]) == 0
    answered = capsys.readouterr().out
    assert main(["run", "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4", json.dumps(REQUEST)]) == 0
    assert capsys.readouterr().out == answered
    assert loaded == [
        "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4",
        "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4",
    ]

    def offline(model: str, patterns: object) -> Path:
        raise UnsupportedModel(f"fetched {model}")

    monkeypatch.setattr(hub, "fetch", offline)
    with pytest.raises(UnsupportedModel, match="fetched LiquidAI/d1-2B"):
        locate.open_model("d1-2b:bf16")  # Decider.load's way in
    assert main(["check", "clef-flsh"]) == UnsupportedModel.exit_code
    assert "did you mean clef-flash" in capsys.readouterr().err
    assert main(["check", "d1-2b:4bit"]) == UnsupportedModel.exit_code  # a tag the family doesn't have
    assert "d1-2b:q8_0" in capsys.readouterr().err


def test_a_bare_name_is_its_familys_smallest_verified_build() -> None:
    """A family's recommended build is the smallest verified one of those that read the most media, or a verified
    safetensors one (the creator's checkpoint or an MLX build) at most 1.5 times its size, else the smallest whose
    prompts are checked; and every tag says what its build is, in one vocabulary."""
    for family, builds in names.families().items():
        assert all(names.TAGS[n.tag].kind == n.kind for n in builds), family
    assert (
        names.parse(str(names.resolve("clef-flash"))).unrevised == "mlx-community/clef-flash-8bit"
    )  # 10.7 GB, within 1.5x of the q8_0 9.7 GB
    assert (
        names.resolve("d1-2b") == "LiquidAI/d1-2B@10e7671d44e4b37700020b1eed3f594c5f2af0b4"
    )  # pictures and video, where its 1.8 GB Q8_0 reads text only
    assert (
        names.parse(str(names.resolve("d1-omni-2b"))).unrevised == "LiquidAI/d1-Omni-2B"
    )  # its q8_0 is past its bound


def test_decide_reads_for_a_person_at_a_terminal() -> None:
    """A line per question, in the request's order: a yes/no as its likelier side, a choice with up to three
    runners-up and how many more, each by its probability, a score by its nearest level's name (an expected level
    like Kettei's too); then the tokens read, and whether a part was cut."""
    answer = {
        "answers": {
            "refund": {"type": "noul", "noul": 0.9984},
            "angry": {"type": "noul", "noul": 0.12},
            "team": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.93,  # a creator's own confidence (Kettei's is rescaled for chance)
                "probabilities": {"billing": 0.95, "shipping": 0.03, "app": 0.015, "sales": 0.0049, "other": 0.0001},
            },
            "urgency": {"type": "score", "score": 1.0, "confidence": 0.941, "legend": {"0": "Can wait", "1": "Today"}},
            "level": {"type": "score", "score": 0.95, "confidence": 0.708, "legend": {"0": "Can wait", "1": "Today"}},
        },
        "usage": {"input_tokens": 1234},
        "truncated": [{"question": "team"}],
    }
    assert readable(answer).splitlines() == [
        "refund   yes 99.8%",
        "angry    no 88.0%",
        "team     billing 95.0%  (shipping 3.0%, app 1.5%, sales 0.5%, 1 more)",
        "urgency  Today (1) 94.1%",
        "level    Today (0.95) 70.8%",
        "(1,234 tokens read, part of it cut to fit, see --json)",
    ]


def test_every_named_hub_build_is_pinned_to_the_commit_it_was_verified_at() -> None:
    """A named safetensors build loads org/name@commit, never the Hub's main, so what `verified` describes is what
    `ippon pull` downloads; asking for it by its bare Hub id still finds it."""
    for n in names.named():
        found = names.parse(n.build.load)
        if found.form == "hub":
            assert found.revision is not None, n.name
            assert names.is_commit(found.revision), n.name
            assert names.of_load(found.repo) == n, n.name


def test_status_refreshes_at_a_terminal_until_ctrl_c_and_prints_once_elsewhere(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["status"]) == 0  # not a terminal: one snapshot, as an agent's shell reads it
    once = capsys.readouterr().out
    assert "\x1b[?1049h" not in once
    assert once.count("ippon status --json") == 1
    naps = iter([None, KeyboardInterrupt()])

    def nap(_: float) -> None:
        if (raised := next(naps)) is not None:
            raise raised

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("time.sleep", nap)
    assert cli.main(["status"]) == 0
    live = capsys.readouterr().out
    assert live.startswith("\x1b[?1049h")  # the other screen, redrawn each second
    assert live.count("\x1b[H\x1b[J") == 2
    assert "Ctrl-C stops" in live
    assert live.split("\x1b[?1049l")[1].count("ippon status --json") == 1  # the last state left on the user's screen


def test_ippon_alone_shows_inkys_sticker_where_the_terminal_draws_pictures_and_words_alone_elsewhere(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """iTerm2 and Ghostty get Inky's sticker, ippon's own picture whole, with the name and its lines written beside
    it after it is drawn. Apple's Terminal, and iTerm2 inside tmux, get the name and its lines alone on one left
    edge, with no picture sequence a terminal could print as text."""
    from ippon import __version__

    for key in ("TERM_PROGRAM", "LC_TERMINAL", "KITTY_WINDOW_ID", "TMUX", "NO_COLOR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    sticker = (files("ippon") / "inky.png").read_bytes()

    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert main([]) == 0
    shown = capsys.readouterr().out
    drawn = re.search(r"\x1b\]1337;File=inline=1;size=(\d+);[^:]*:([A-Za-z0-9+/=]+)\x07", shown)
    assert drawn is not None
    assert base64.b64decode(drawn[2]) == sticker
    assert int(drawn[1]) == len(sticker)
    assert shown[drawn.end() : drawn.end() + 2] == "\x1b8"  # the cursor back where it was, for the words beside
    assert re.search(r"\x1b\[\d+Grun decision models on your mac", shown)

    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    assert main([]) == 0
    chunks = re.findall(r"\x1b_G([^;]*);([A-Za-z0-9+/=]*)\x1b\\", capsys.readouterr().out)
    assert "q=2" in chunks[0][0]  # no reply, which would reach the shell's prompt as text
    assert all(len(data) <= 4096 for _, data in chunks)
    assert chunks[-1][0].endswith("m=0")
    assert base64.b64decode("".join(data for _, data in chunks)) == sticker

    # Apple's Terminal (one opened from iTerm2 too, which inherits LC_TERMINAL) and iTerm2 inside tmux: Inky in
    # pixels beside the name in an 80-column window, and no picture sequence a terminal could print as text
    for program, inherited, tmux in (
        ("Apple_Terminal", None, None),
        ("Apple_Terminal", "iTerm2", None),
        ("iTerm.app", None, "/tmp/tmux-501/default,1,0"),
    ):
        monkeypatch.setenv("TERM_PROGRAM", program)
        if inherited:
            monkeypatch.setenv("LC_TERMINAL", inherited)
        if tmux:
            monkeypatch.setenv("TMUX", tmux)
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a: os.terminal_size((80, 24)))
        assert main([]) == 0
        shown = capsys.readouterr().out
        assert "\x1b]1337" not in shown
        assert "\x1b_G" not in shown
        words = re.sub(r"\x1b\[[0-9;]*m", "", shown).splitlines()
        beside = [line for line in words if line.endswith("run decision models on your mac")]
        assert len(beside) == 1
        assert len(beside[0]) <= 79  # beside Inky, in the window
        assert shown.count("\x1b[48;") > 100  # Inky's cells, each a colour
        assert any(line.endswith(f"ippon {__version__}") for line in words)


def test_a_request_is_read_from_stdin_only_when_dash_asks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Open:  # an agent's shell: stdin open, nothing coming
        def read(self) -> str:
            pytest.fail("read stdin unasked")

        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", Open())
    assert main(["lint"]) != 0
    assert "give the request" in capsys.readouterr().err
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"state": "s", "questions": {}})))
    main(["lint", "-"])  # asked for: read
    assert "not a System One request" in capsys.readouterr().out


def test_lint_accepts_media_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`ippon lint` accepts --image/--video/--audio just like `ippon decide`."""
    import base64

    photo = tmp_path / "dot.gif"
    photo.write_bytes(base64.b64decode("R0lGODlhAQABAAAAACw="))
    req = tmp_path / "req.json"
    req.write_text(
        json.dumps(
            {
                "state": "Check photo",
                "questions": {"ok": {"type": "noul", "instructions": "Is the photo clearly in focus?"}},
            }
        )
    )
    assert main(["lint", str(req), "--image", str(photo), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []
