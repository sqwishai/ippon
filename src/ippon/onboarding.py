"""`ippon setup`: from a fresh install to a decision model answering on this Mac, and the commands that use it.

Five steps, each one question with its answer already chosen, so Enter alone goes through them:

1. this Mac (its chip and memory) and the model for it: every model listed in one order, Sqwish's first (`ordered`),
   each at the precision it is published at with what it reads and whether it fits here, the cursor on the first
   that isn't too big (`first_usable`) and the highlighted one described below the list, how it fits included;
2. its licence, where it has one, then the download and a first answer, which proves it runs here;
3. what to use it with: the coding agents found on this Mac, and the user's own code (an API on port 8000);
4. for the agents ticked, whether they get it only when started with `ippon claude` and the like (the default, which
   writes nothing to their settings) or in every session, which `ippon remove AGENT` undoes;
5. how the server runs: started by whatever needs it (an agent's MCP server starts it) and its memory given back when
   idle (the default), or from login, which suits the user's own code, since a program calling port 8000 can't start
   a server that isn't there.

Nothing is refused: a model that needs more memory than this Mac has is listed with how much it needs, and the
download step says so before anything is fetched. The choice is kept in setup.json in ippon's settings, where the
agent commands and `ippon start` read the model from. Without a terminal to ask (an agent's shell, CI) the defaults are
taken, the agents' settings are left alone, and what was chosen is printed with how to change it.
"""

from __future__ import annotations

import functools
import os
import plistlib
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from . import __version__, brand, daemon, drift, hashes, hub, licences, names, processes, recommend
from .admission import budget
from .agents import AGENTS, KINDS, Agent, add_everywhere, remove_everywhere
from .device import Device
from .errors import DecisionError, InvalidRequest
from .locate import downloaded
from .options import precision_argv
from .quantization import FULL, PRECISIONS, QUANTIZED, Precision, form_of
from .schema import MEDIA
from .served import DEFAULT_PORT, url_of
from .storage import cache_dir, config_dir, read_record, set_aside, unreadable, write_json_atomic
from .terminal import Back, Line, Terminal, commands, styled
from .verified import DEFAULT_MODEL

Serving = Literal["on demand", "login"]
# The precisions ← and → turn a full-precision build through, quantized as it loads (None: as published)
TURNED = (None, *QUANTIZED)
SETUP = "setup.json"  # in ippon's settings: what setup chose (`Settings`)
SETUP_UNREADABLE = unreadable(SETUP)  # beside it: the last one that couldn't be read, set aside unchanged
LOGIN_LABEL = "ai.sqwish.ippon"  # the launchd job that serves the model from login
LOGIN_LOG = "login.log"  # its log, in the servers' folder (daemon.folder), which `ippon status` tells apart


@dataclass
class Settings:
    """What setup chose, in setup.json in ippon's settings."""

    model: str | None = None  # a short name (kettei-core:bf16), or a family name with `precision`
    precision: str | None = None  # "q8" for a model ippon quantizes as it loads; None as published
    serving: Serving = "on demand"
    everywhere: list[Agent] = field(default_factory=list)  # agents given ippon in every session

    @staticmethod
    def path() -> Path:
        return config_dir() / SETUP

    @classmethod
    def load(cls) -> Settings:
        """What setup.json holds; the defaults when there is none or it can't be read (`_read`)."""
        data = read_record(cls.path()) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self) -> None:
        """Written to setup.json. An unreadable one is moved aside to SETUP_UNREADABLE as it is first: it names the
        agents given ippon in every session, which `ippon remove` takes it out of, and a fresh one would forget."""
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        set_aside(path)
        write_json_atomic(path, asdict(self), indent=2)

    def leave(self, agent: Agent) -> str | None:
        """ippon taken out of `agent`'s own settings (`remove_everywhere`) and off the agents it is in every session
        of, saved: `ippon remove AGENT`, and setup for an agent no longer ticked. What was taken out."""
        gone = remove_everywhere(agent)
        self.everywhere = [a for a in self.everywhere if a != agent]
        self.save()
        return gone

    def serve(self, serving: Serving) -> None:
        """The chosen model served from login (`serve_from_login`) or on demand (no login job), saved: setup's
        answer, and `ippon remove login`."""
        if serving == "login":
            assert self.model is not None
            serve_from_login(self.model, self.served_at())
        elif from_login():
            stop_from_login()
        self.serving = serving
        self.save()

    def served_at(self) -> str | None:
        """The precision the chosen model is served at beside its name: None when its name carries it (kettei-core:q8)
        or it runs as published, the precision for a choice saved with its precision apart from its name."""
        return self.precision if self.precision and not names.precision_of(self.model or "") else None

    def options(self) -> list[str]:
        """The loading options the chosen model is served with (`served_at`), as arguments."""
        return precision_argv(self.served_at())


def chosen_model() -> tuple[str, str | None]:
    """The model the agent commands and `ippon start` use when none is named, and the precision it is served at
    beside its name (`Settings.served_at`): setup's choice, else verified.DEFAULT_MODEL."""
    settings = Settings.load()
    if settings.model:
        return settings.model, settings.served_at()
    return DEFAULT_MODEL, None


# -- the models ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """A model as the list shows it: what to pull and load it with, the memory it holds, and its build."""

    named: names.Named
    precision: str | None = None
    builds: tuple[
        names.Named, ...
    ] = ()  # the family's builds ← and → step between, `named` among them; () for it alone

    @property
    def name(self) -> str:
        return f"{self.named.family}:{self.precision}" if self.precision else self.named.name

    @property
    def label(self) -> str:
        return f"{self.named.model} ({self.at.form})" if self.precision else self.named.model

    @property
    def at(self) -> Precision:
        """The precision it loads at: the one it is turned to, else auto (as published)."""
        return PRECISIONS[self.precision or "auto"]

    @property
    def steps(self) -> list[Choice]:
        """What ← and → step through, from the most memory to the least: a full-precision checkpoint at each of
        TURNED its name loads it at, a quantized build (a creator's GGUF file) as published. `d1-2b:q8` is Liquid's
        own Q8_0 file, so d1-2B's checkpoint isn't also offered at 8 bits."""
        out = [
            replace(self, named=n, precision=p)
            for n in self.builds or (self.named,)
            for p in (TURNED if n.kind == "safetensors" and n.tag in FULL else (None,))
            if p is None or names.find(f"{n.family}:{p}")[0] == n
        ]
        return sorted(out, key=lambda c: -c.gigabytes)

    @property
    def gigabytes(self) -> float:
        """About what its weights hold in memory at the precision it loads at (`recommend.gigabytes`)."""
        return recommend.gigabytes(self.named, self.at)

    def turned(self, step: int) -> Choice:
        """The step `step` (-1 or +1) away, stopping at either end."""
        steps = self.steps
        return steps[max(0, min(len(steps) - 1, steps.index(self) + step))]

    @property
    def reads(self) -> list[str]:
        return ["text", *(m.shown for m in MEDIA if m.field in self.named.media)]


def choices(device: Device, every_build: bool = False) -> list[Choice]:
    """The models for this Mac: each family's build as `ippon recommend` gives it, each at the precision it is
    published at, with the family's other verified builds from the Hub to step to (every named build alone, with
    `every_build`): the user turns it."""
    if every_build:
        return [Choice(n) for n in names.named()]
    families = names.families()
    return [
        Choice(p.named, builds=tuple(n for n in families[p.named.family] if n == p.named or steppable(n)))
        for p in recommend.recommend("text", device)
    ]


def steppable(n: names.Named) -> bool:
    """Whether setup's ← and → reach a build besides the family's own: verified, from the Hub (an Ollama build is
    in Ollama's store) and at 8 bits or more."""
    return n.build.status == "verified" and n.kind in ("safetensors", "gguf") and not n.low_bit


def fit(choice: Choice, device: Device) -> recommend.Fit:
    """Whether the model, at the precision it is turned to, fits this Mac (`recommend.fit`)."""
    return recommend.fit(choice.gigabytes, device)


def first_usable(listed: Sequence[Choice], device: Device) -> int:
    """Where the cursor starts: the first model in the list that isn't too big for this Mac at its precision, has
    its prompts checked and isn't below 8 bits; else the first. The list's order never changes for it."""
    for i, c in enumerate(listed):
        if not c.named.low_bit and fit(c, device) != "too big" and c.named.build.status in recommend.CHECKED:
            return i
    return 0


def mac(device: Device) -> str:
    memory = f"{(device.memory_bytes or 0) / 2**30:.0f} GB memory"
    gpu = f" · {device.working_set_bytes / 2**30:.0f} GB for the GPU" if device.working_set_bytes else ""
    return f"{device.name or 'This Mac'} · {memory}{gpu}"


def ordered(listed: Sequence[Choice]) -> list[Choice]:
    """One order on every Mac: Sqwish's models first (verified.py's `starter`: Kettei Core, then Kettei Dot), then
    the rest in their model's `place`, a family's builds together. How each fits this Mac is on its row and in its
    description, never in where it sits."""

    def place(c: Choice) -> tuple[bool, int, int, str]:
        entry = c.named.entry
        return (entry.starter is None, entry.starter or 0, entry.place, c.named.name)

    return sorted(listed, key=place)


def form(c: Choice) -> str:
    """The precision column: the precision a checkpoint is turned to, else the published build's own format, between
    arrows when ← and → have somewhere to go."""
    published = {"8bit": "8-bit", **dict.fromkeys(FULL, "full")}.get(c.named.tag, c.named.tag.upper())
    shown = c.at.form if c.precision else published
    return f"◂ {shown} ▸" if len(c.steps) > 1 else shown


def lines(listed: Sequence[Choice], device: Device) -> list[Line]:
    width = max(len(c.named.model) for c in listed)
    forms = max(len(form(c)) for c in listed)
    reads = max(len(" · ".join(c.reads)) for c in listed)
    notes = {"too big": "too big for this Mac"}
    out = []
    for c in listed:
        fitting = fit(c, device)
        text = f"{c.named.model.ljust(width)}  {form(c).ljust(forms)}  {c.gigabytes:>5.1f} GB  {fitting:<8}  "
        text += " · ".join(c.reads).ljust(reads)
        tagged = tag(c)
        note, tone = (notes[fitting], "2") if fitting in notes else tagged or ("", "2")
        out.append(Line(text, note, fitting != "too big", tone, c.label))
    return out


def tag(c: Choice) -> tuple[str, str] | None:
    """The tag on a model's row and its accent: how far its answers move at the precision it is turned to
    (drift.TAG). None at full precision."""
    if c.at.changes_weights and not c.at.low_bit:
        return drift.TAG["close"]
    if c.at.changes_weights:
        return drift.TAG[drift.level(c.name, c.at.form)]
    if c.named.low_bit:  # a published build below 8 bits, unmeasured unless drift.MEASURED names it
        return drift.TAG[drift.level(c.named.name, form_of("affine", 4))]
    return None


def describe(c: Choice, device: Device) -> list[str]:
    """The highlighted model, below the list: what it is, what it reads, what the precision it is turned to does to
    its answers and what it takes."""
    n = c.named
    said = ", ".join(c.reads[:-1]) + " and " + c.reads[-1] if len(c.reads) > 1 else c.reads[0]
    out = [f"{c.label} · {n.licence}", f"Answers questions about {said}: a probability for every option."]
    if c.at.changes_weights and not c.at.low_bit:
        out.append("✓ Half the memory, and its answers stay close to full precision.")
    elif c.at.changes_weights:
        level = drift.level(c.name, c.at.form)
        out.append(f"{drift.TAG[level][0][0]} {drift.SAYS[level]}")
    fetched = "already downloaded" if downloaded(str(names.resolve(n.name))) else f"a {n.build.weights} download"
    out.append(f"It holds about {c.gigabytes:.1f} GB in memory ({fetched}).")  # the row's figure, and what it fetches
    room = budget(device.working_set_bytes) / 1e9 if device.working_set_bytes else None
    says = {
        "fits": f"It fits this Mac with room to spare ({room or 0:.1f} GB for models).",
        "tight": f"It only just fits the {room or 0:.1f} GB this Mac gives models, so other apps may slow it.",
        "too big": f"It is too big for this Mac, which gives models {room or 0:.1f} GB.",
        "unknown": "This Mac didn't say how much memory it gives models, so whether it fits is unknown.",
    }
    out.append(says[fit(c, device)])
    return out


# -- the agents ------------------------------------------------------------------------------------------------------


def found_agents() -> dict[Agent, bool]:
    return {agent: shutil.which(agent) is not None for agent in AGENTS}


# -- serving from login ----------------------------------------------------------------------------------------------


def _login_plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LOGIN_LABEL}.plist"


def serve_from_login(model: str, precision: str | None) -> None:
    """A launchd job that serves `model` (at `precision`, beside its name) on port 8000 from login (and now), giving
    its memory back after half an hour without a request, as a server a client starts; `ippon stop` stops it until
    the next login."""
    log = daemon.folder() / LOGIN_LOG
    serving = daemon.Wanted((model,), precision or "auto").argv()
    job = {
        "Label": LOGIN_LABEL,
        "ProgramArguments": [*processes.command(), "serve", *serving, "--record-log", str(log)],
        "RunAtLoad": True,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
    }
    path = _login_plist()
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_from_login()
    path.write_bytes(plistlib.dumps(job))
    _launchctl("bootstrap", f"gui/{os.getuid()}", str(path))


def remove_all(models: bool) -> list[str]:
    """Everything ippon added to this Mac besides its own command, taken out, for `ippon remove all`: its MCP server
    and skill from every agent's settings, its login job, its running servers, its settings and caches, and with
    `models` the model folders it downloaded. What it took out, a line each."""
    done = [f"{KINDS[a].name}: {gone}" for a in AGENTS if (gone := remove_everywhere(a))]
    if from_login():
        stop_from_login()
        done.append("the login job that started ippon's server: removed")
    for server in daemon.running():
        daemon.stop(server)
        done.append(f"the server at {server.url}: stopped")
    if models:
        done += hub.delete_downloads()
    for folder, own, named in (
        (config_dir(), CONFIG_FILES, "IPPON_CONFIG"),
        (cache_dir(), CACHE_FILES, "IPPON_CACHE"),
    ):
        if not folder.exists():
            continue
        if not os.environ.get(named):  # ippon's own folder: all of it
            shutil.rmtree(folder)
            done.append(f"{folder}: deleted")
            continue
        # a folder IPPON_CONFIG or IPPON_CACHE named, which may be shared: ippon's own files in it, nothing else
        taken = [path for path in (folder / name for name in own) if path.exists()]
        for path in taken:
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
        if taken:
            done.append(f"{folder}: ippon's {', '.join(p.name for p in taken)} deleted, the rest left")
    return done


# what ippon writes in its settings and cache folders, each module's own, which is all `remove all` deletes from one
# that the environment names (IPPON_CONFIG, IPPON_CACHE): it may be a folder other programs keep files in too
CONFIG_FILES = (SETUP, SETUP_UNREADABLE, *licences.WRITES)
CACHE_FILES = (
    daemon.SERVERS,
    hub.DOWNLOADS,
    hub.LOCK,
    hub.UNREADABLE,
    hashes.HASHES,
)


def _launchctl(*argv: str) -> None:
    """launchctl run with `argv`, its output and failure (a job already gone) left unsaid."""
    subprocess.run(["launchctl", *argv], check=False, capture_output=True)


def stop_from_login() -> None:
    _launchctl("bootout", f"gui/{os.getuid()}/{LOGIN_LABEL}")
    _login_plist().unlink(missing_ok=True)


def from_login() -> bool:
    return _login_plist().exists()


# -- the flow --------------------------------------------------------------------------------------------------------


@dataclass
class Steps:
    """What setup does besides asking, given by the command line: download a model, its licences asked about and
    its progress shown with the functions given (`pull`: the bytes it fetched; a DecisionError when it can't), and
    load it and answer a first question with it (`sample`, a line saying what it answered)."""

    pull: Callable[[str, Callable[[Any], bool] | None, Callable[[Any], None] | None], int]
    sample: Callable[[str, Sequence[str]], str]


@dataclass(frozen=True)
class Answers:
    """Setup's questions answered on the command line, each None to ask it: the model's short name, what to use it
    with (agents' names and "code" for the user's own), every session or not, and from login or not."""

    model: str | None = None
    use: tuple[str, ...] | None = None
    every_session: bool | None = None
    from_login: bool | None = None


def run(term: Terminal | None, device: Device, steps: Steps, answers: Answers | None = None) -> Settings:
    """Setup, asking `term` what `answers` leaves open (None: the defaults, nothing asked)."""
    answers = answers or Answers()
    say = term.say if term else functools.partial(print, flush=True)  # a log someone tails, line by line
    settings = Settings.load()
    if term:
        say("\n".join(["", *brand.banner(term.colour, [f"ippon {__version__}", mac(device)]), ""]))
    else:
        say(f"ippon {__version__} · {mac(device)}")
    found = found_agents()
    start = term.mark() if term else 0
    choice: Choice | None = None
    while True:  # the model, then the questions after it; going back from the first of those chooses again
        while True:
            choice = _choose_model(term, device, answers.model, choice)
            say(f"\n  {term.style('2/4  Download', '1') if term else '2/4  Download'} · {choice.label}")
            try:
                fetched = steps.pull(
                    choice.name, None if term is None else licence_question(term), term.downloading() if term else None
                )
                break
            except DecisionError as error:  # too big for this Mac, a licence declined, no network: choose again
                if term is None or answers.model is not None:
                    raise
                say(f"\n  {commands(str(error), term.colour)}\n")
        tick = term.style("✓", "mint") if term else "✓"
        say(f"  {tick} {f'{fetched / 1e9:.1f} GB downloaded' if fetched else 'Already on this Mac'}")
        if term:
            with term.working(f"Loading {choice.label} and asking it a first question"):
                answered = steps.sample(choice.name, [])
        else:
            answered = steps.sample(choice.name, [])
        say(f"  {tick} {answered}\n")
        try:
            use, own, everywhere, serving = _questions(term, found, answers)
            break
        except Back:
            assert term is not None
            term.rewind(start)
    settings.model, settings.precision = choice.name, choice.precision
    for agent in sorted(set(settings.everywhere) - set(everywhere)):
        settings.leave(agent)
    written = {agent: add_everywhere(agent, settings.model, settings.options()) for agent in everywhere}
    settings.everywhere = list(everywhere)
    settings.serve(serving)
    say(done(choice, use, everywhere, own, serving, found, written, term))
    return settings


def _questions(
    term: Terminal | None, found: dict[Agent, bool], answers: Answers
) -> tuple[list[Agent], bool, list[Agent], Serving]:
    """What to use the model with, how the agents get it and how ippon keeps running. ← on a question goes back to
    the one asked before it, and out of the first one asked to the model's choice (Back), when that was asked."""
    use: list[Agent] = []
    everywhere: list[Agent] = []
    serving: Serving = "on demand"
    own = False
    given = {"uses": answers.use, "everywhere": answers.every_session, "serving": answers.from_login}
    order, at = ("uses", "everywhere", "serving"), 0
    asked: list[str] = []
    marks: dict[str, int] = {}
    while at < len(order):
        step = order[at]
        back = bool(asked) or answers.model is None
        if term is not None:
            marks[step] = term.mark()
        try:
            if step == "uses":
                use, own = _choose_uses(term, found, answers.use, back)
            elif step == "everywhere":
                everywhere = _choose_everywhere(term, use, answers.every_session, back) if use else []
            else:
                serving = _choose_serving(term, own, answers.from_login, back)
        except Back:
            if not asked:
                raise
            at = order.index(asked.pop())
            if term is not None:
                term.rewind(marks[order[at]])
            continue
        if term is not None and given[step] is None and (step != "everywhere" or use):
            asked.append(step)
        at += 1
    return use, own, everywhere, serving


def licence_question(term: Terminal) -> Callable[[Any], bool]:
    """Asks a person at `term` to accept a model's licence: setup asks it with --yes too, since nothing else may."""

    def approve(licence: Any) -> bool:
        term.say(f"\n  It comes under the {licence.name}")
        term.say(f"  {term.style(licence.url, '2')}")
        return term.confirm("  Accept it?", False)  # a legal agreement: never accepted by Enter alone

    return approve


def _choose_model(term: Terminal | None, device: Device, model: str | None, before: Choice | None = None) -> Choice:
    """The model and precision chosen; going back to it highlights `before`, the choice made the time before."""
    if model is not None:  # named on the command line: a short name, kettei-core:q8 for Core at 8 bits
        if not names.is_name(model):
            raise InvalidRequest(f"ippon setup takes a model's short name (ippon pull --list); {model} isn't one")
        return Choice(names.find(model)[0], names.precision_of(model))
    every = False
    while True:
        listed = choices(device, every)
        listed = ordered(listed)
        if term is None:  # nobody to turn a precision: the first that fits as published (Kettei Core, else Dot)
            return listed[first_usable(listed, device)]
        start = first_usable(listed, device)
        at = next((i for i, c in enumerate(listed) if before and before in c.steps), start)
        if before is not None and before in listed[at].steps:
            listed[at] = before

        def details(at: int, shown: list[Choice] = listed) -> list[str]:
            return describe(shown[at], device)

        def turn(at: int, step: int, shown: list[Choice] = listed) -> list[Line]:
            shown[at] = shown[at].turned(step)
            return lines(shown, device)

        picked = term.pick(
            "  1/4  Choose a decision model",
            lines(listed, device),
            at,
            details,
            keys=() if every else (("a", "every build"),),
            turn=("precision", turn),
        )
        if picked == "a":
            every = True
            continue
        assert isinstance(picked, int)
        return listed[picked]


def _choose_uses(
    term: Terminal | None, found: dict[Agent, bool], given: tuple[str, ...] | None, back: bool
) -> tuple[list[Agent], bool]:
    """The agents to use the model with, and whether the user's own code is one of its uses."""
    if given is not None:
        return [a for a in AGENTS if a in given], "code" in given
    if term is None:
        return [a for a in AGENTS if found[a]], False
    rows = [Line(KINDS[a].name, "found" if found[a] else f"not installed · {KINDS[a].install}") for a in AGENTS]
    rows.append(Line("My own code", f"an API at {url_of(DEFAULT_PORT)}"))
    ticked = term.checklist("  3/4  Use it with", rows, [*(found[a] for a in AGENTS), False], back=back)
    return [a for a, on in zip(AGENTS, ticked, strict=False) if on], ticked[-1]


def _choose_everywhere(term: Terminal | None, use: Sequence[Agent], given: bool | None, back: bool) -> list[Agent]:
    if given is not None or term is None:
        return list(use) if given else []
    starts = " / ".join(f"ippon {a}" for a in use)
    picked = term.choose(
        f"  3/4  How should {' and '.join(KINDS[a].name for a in use)} get it?",
        [
            Line(f"When I start them with {starts}", "nothing in their settings changes"),
            Line("In every session", "adds ippon to their settings · ippon remove undoes it"),
        ],
        0,
        back=back,
    )
    return list(use) if picked == 1 else []


def _choose_serving(term: Terminal | None, own: bool, given: bool | None, back: bool) -> Serving:
    if given is not None or term is None:
        return "login" if given else "on demand"
    picked = term.choose(
        "  4/4  Keep ippon running?",
        [
            Line(
                "When something needs it",
                f"starts on demand · gives memory back after {daemon.STARTED_IDLE_MIN} idle minutes",
            ),
            Line("From login", "for your own code, which can't start it itself"),
        ],
        1 if own else 0,
        back=back,
    )
    return "login" if picked == 1 else "on demand"


def done(
    choice: Choice,
    use: Sequence[Agent],
    everywhere: Sequence[Agent],
    own: bool,
    serving: Serving,
    found: dict[Agent, bool],
    written: dict[Agent, list[str]],
    term: Terminal | None = None,
) -> str:
    """The last screen: what was set up and the commands that use it, only those that apply."""
    rows: list[tuple[str, str]] = []
    for agent in use:
        if agent in everywhere:
            rows.append((agent, f"{KINDS[agent].name}, now with {choice.label} beside it in every session"))
        else:
            rows.append((f"ippon {agent}", f"{KINDS[agent].name}, with {choice.label} beside it"))
    if not use:
        rows += [(f"ippon {a}", f"{KINDS[a].name}, with {choice.label} beside it") for a in AGENTS if found[a]]
    if own and serving == "on demand":
        rows.append(("ippon start", f"serve it to your own code at {url_of(DEFAULT_PORT)}"))
    elif own:
        rows.append((url_of(DEFAULT_PORT), "serving now and from every login: POST /v1/systemone"))
    rows.append(("ippon decide", "ask it a question from the terminal or a script"))
    width = max(len(command) for command, _ in rows)
    colour = term is not None and term.colour
    out = ["", f"  {styled('Ready', colour, '1')} · {choice.label} runs on this Mac", ""]
    out += [f"    {styled(command.ljust(width), colour, '1', 'amber')}   {what}" for command, what in rows]
    for agent, wrote in written.items():
        out += ["", f"  Added to {KINDS[agent].name}: " + "; ".join(wrote)]
    out += ["", "  Change any of this with ippon setup."]
    return "\n".join(out)
