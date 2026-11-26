"""Short model names: `ippon pull clef-flash`, `ippon decide d1-2b:q8_0 ...`, `Decider.load("d3-mini")`.

A build's name is FAMILY:TAG, its `name` in verified.py, so the names, the builds they load and what was checked of each
come from one table. The tag says what the build is, in one vocabulary across families (TAGS): the creator's checkpoint
in the precision it ships (bf16, or fp16 or fp32 where that is what the creator publishes), MLX's affine builds (8bit,
6bit, 5bit, 4bit), GGUF builds (q8_0) and Ollama's builds (mxfp8, nvfp4, mxfp4). A size is in the family's name only
where the creator publishes several sizes as one model.

A bare FAMILY loads its recommended build (`recommended`, the one rule): the smallest whose answers are verified,
or a verified safetensors build (the creator's checkpoint or an MLX build) at most 1.5 times its size; else the
smallest whose prompts are checked. A Hub id, a GGUF build (repo:QUANT), an ollama:// build and a path all hold a
"/", and a .gguf file or a folder that exists is a path, so none of them is taken for a name: they load as given.

A build below 8 bits (by TAGS) is never recommended: it loads only by its own name, as FAMILY:q4 or FAMILY:q6 (a
full-precision build quantized here), or by its Hub id, file or Ollama tag, and every such load warns how far it
drifts from the creator's answers (drift.py).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from . import terminal
from .errors import DecisionError, InvalidRequest, UnsupportedModel
from .quantization import FULL, QUANTIZED, low_bit, unsupported_format
from .quantization import listing as spoken
from .verified import MODELS, Build, Model

Kind = Literal["safetensors", "gguf", "ollama"]


class Tag(NamedTuple):
    kind: Kind
    bits: int | None  # its weights' bits, None at full precision


TAGS: dict[str, Tag] = {
    **{name: Tag("safetensors", None) for name in FULL},  # the creator's checkpoint, named as its precision is
    "8bit": Tag("safetensors", 8),
    "6bit": Tag("safetensors", 6),
    "5bit": Tag("safetensors", 5),
    "4bit": Tag("safetensors", 4),
    "q8_0": Tag("gguf", 8),
    "mxfp8": Tag("ollama", 8),
    "nvfp4": Tag("ollama", 4),
    "mxfp4": Tag("ollama", 4),
}
# tags of formats ippon does not run, refused with the family's builds that it does
REFUSED = ("2bit", "3bit", "q2", "q3", "q2_k", "q3_k_m", "w8a8")
STATUS = {"prompts": "prompts checked"}  # how a status reads in a listing, where verified.py's word is terse
# How much larger than a family's smallest verified build a verified safetensors build may be and still be the one a
# bare name loads: MLX reads it as published, where a GGUF or Ollama build is converted or packed on load.
SAFETENSORS_WITHIN = 1.5


Form = Literal["name", "hub", "gguf", "gguf-file", "ollama", "path"]
COMMIT = re.compile(r"[0-9a-f]{40}")  # a Hub commit, as a pinned revision and a snapshot folder's name give it


@dataclass(frozen=True)
class Reference:
    """What a model reference names (`parse`): its form, and a Hub id's or GGUF build's repository, revision and
    quantization."""

    form: Form
    text: str  # as given
    repo: str = ""  # hub and gguf: org/name
    revision: str | None = None  # hub and gguf: the branch, tag or commit after "@"
    quant: str = ""  # gguf: the file's quantization, Q8_0

    @property
    def unrevised(self) -> str:
        """The reference without its revision: what a build is known by whichever revision it is pinned to."""
        if self.form == "hub":
            return self.repo
        return f"{self.repo}:{self.quant}" if self.form == "gguf" else self.text


def is_commit(revision: str) -> bool:
    return COMMIT.fullmatch(revision) is not None


def split(model: str) -> tuple[str, str | None]:
    """A Hub id's repository and revision: 'org/name@rev' -> ('org/name', 'rev'); UnsupportedModel for anything
    else."""
    repo, _, revision = model.partition("@")
    if repo.count("/") != 1:
        raise UnsupportedModel(f"{model!r}: not a local directory or a Hub id like 'org/name[@revision]'")
    return repo, revision or None


def parse(model: str | Path) -> Reference:
    """What `model` names, in the order a load reads it: an ollama:// build, a .gguf file, a path that exists, a short
    name (no "/"), a GGUF build (org/name[@revision]:QUANT) or a Hub id (org/name[@revision]); anything else is a path,
    which a load then finds missing."""
    text = str(model)
    if text.startswith("ollama://"):
        return Reference("ollama", text)
    if text.endswith(".gguf"):
        return Reference("gguf-file", text)
    if text.startswith(("./", "../", "/", "~")) or _exists(text):
        return Reference("path", text)  # written as a path: never sent to the Hub as a repository's name
    if "/" not in text:
        return Reference("name", text)
    head, colon, quant = text.rpartition(":")
    if colon and quant and head.partition("@")[0].count("/") == 1:
        return Reference("gguf", text, *split(head), quant)
    if text.partition("@")[0].count("/") == 1:
        return Reference("hub", text, *split(text))
    return Reference("path", text)


def _exists(text: str) -> bool:
    """Whether `text` names a file or folder here; False for a name the file system can't hold (past 1,024 bytes, as a
    request's model may be), never an error."""
    try:
        return Path(text).expanduser().exists()
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class Named:
    """A build with a name, and its model (verified.py's `Model`)."""

    name: str
    entry: Model
    build: Build

    @property
    def model(self) -> str:
        return self.entry.name

    @property
    def licence(self) -> str:
        return self.entry.licence

    @property
    def family(self) -> str:
        return self.name.partition(":")[0]

    @property
    def tag(self) -> str:
        return self.name.partition(":")[2]

    @property
    def kind(self) -> Kind:
        """How it is published: safetensors (the creator's checkpoint or an MLX build), a GGUF file or Ollama's
        build."""
        form = parse(self.build.load).form
        return "ollama" if form == "ollama" else "gguf" if form == "gguf" else "safetensors"

    @property
    def size(self) -> float:
        """Its weights' GB, as verified.py gives them."""
        return self.build.gigabytes

    @property
    def resident(self) -> float:
        """The GB its weights hold in memory as published: its model's `resident_gb` at full precision where verified.py
        gives one, else its weights' size."""
        if self.entry.resident_gb is not None and self.tag in FULL:
            return self.entry.resident_gb
        return self.size

    @property
    def low_bit(self) -> bool:
        """Whether its weights are below 8 bits (its tag's): never recommended, loaded only when asked for."""
        return low_bit(TAGS[self.tag].bits)

    @property
    def status(self) -> str:
        return STATUS.get(self.build.status, self.build.status)

    @property
    def media(self) -> dict[str, str]:
        """What a request to it may carry besides text ("images", "audio"), and how the answers about each came out
        (verified.py's status)."""
        return dict(self.build.media)


def named() -> tuple[Named, ...]:
    """Every build with a name, in verified.py's order."""
    return tuple(Named(b.name, m, b) for m in MODELS for b in m.builds if b.name)


def catalogued(load: str) -> tuple[Model, Build] | None:
    """The catalogue's build that `load` (a Hub id, GGUF build or ollama:// build, any case, at any revision) loads,
    and its model, if one does: a build is known by its reference without the revision (`Reference.unrevised`)."""
    wanted = parse(load).unrevised.lower()
    return next(((m, b) for m in MODELS for b in m.builds if parse(b.load).unrevised.lower() == wanted), None)


def in_repository(repo: str | None) -> Model | None:
    """The catalogue's model one of whose builds (a Hub id or a GGUF build) is in Hub repository `repo` (any case):
    what a licence is known by, every build of a model being under its licence."""
    wanted = (repo or "").lower()
    return next((m for m in MODELS for b in m.builds if wanted and parse(b.load).repo.lower() == wanted), None)


def for_model(load: str) -> tuple[Build, ...]:
    """Every build of the model whose build `load` loads (`catalogued`); () for one the catalogue doesn't list."""
    found = catalogued(load)
    return found[0].builds if found else ()


def smallest(builds: Sequence[Named], fits: Callable[[Named], bool]) -> Named | None:
    """The smallest of `builds` that `fits`, never one below 8 bits (only a caller naming one loads it): what a
    refusal for memory offers instead."""
    return min((n for n in builds if not n.low_bit and fits(n)), key=lambda n: n.size, default=None)


def smaller_build(model: str | Path) -> str:
    """What a refusal for memory adds: the family's smallest published build smaller than `model`'s when `model` names
    one (an 8 GB M1 with an editor open has 1.6 to 2.0 GB free, where d1-2B's bf16 checkpoint, 3.6 GB, is refused and
    its creator's Q8_0 file, 1.8 GB, loads), else where to find the builds this Mac runs."""
    here = of_load(str(model))
    if here is not None:
        smaller = smallest(families()[here.family], lambda n: n.size < here.size)
        if smaller:
            return f" Its creator's {smaller.build.weights} build may fit: ippon pull {smaller.name}."
    return " `ippon recommend` lists the builds this Mac runs."


def families() -> dict[str, tuple[Named, ...]]:
    """Each family's builds, in verified.py's order."""
    out: dict[str, list[Named]] = {}
    for n in named():
        out.setdefault(n.family, []).append(n)
    return {family: tuple(builds) for family, builds in out.items()}


def recommended(builds: Sequence[Named]) -> tuple[Named, str] | None:
    """The build a bare family name loads, and why: among the verified builds whose answers are verified about the
    most kinds of media, the smallest, or a safetensors one (the creator's checkpoint or an MLX build, which MLX reads
    as published) when one is within SAFETENSORS_WITHIN times its size, the smallest of those; else the smallest whose
    prompts are checked (its answers maybe checked too, and past the bound). None when no build is either. A low-bit
    build is never one of them, and a smaller build checked about less never is either: d1-2B's name loads Liquid's
    checkpoint, whose answers about pictures and video are verified, not its Q8_0 file, whose answers about images
    are past the bound and about video unchecked."""
    builds = [n for n in builds if not n.low_bit]
    verified = [n for n in builds if n.build.status == "verified"]
    if verified:

        def sure_of(n: Named) -> list[str]:
            return sorted(kind for kind, status in n.media.items() if status == "verified")

        widest = [n for n in verified if len(sure_of(n)) == max(len(sure_of(m)) for m in verified)]
        reads = f" verified about {' and '.join(sure_of(widest[0]))}" if len(widest) < len(verified) else ""
        smallest = min(widest, key=lambda n: n.size)
        near = [n for n in widest if n.kind == "safetensors" and n.size <= SAFETENSORS_WITHIN * smallest.size]
        best = min(near, key=lambda n: n.size) if near else smallest
        if best is smallest:
            return best, f"the smallest verified build{reads}"
        return best, (
            f"a safetensors build within {SAFETENSORS_WITHIN:g}x of the smallest verified{reads} ({smallest.name}, "
            f"{smallest.build.weights})"
        )
    checked = [n for n in builds if n.build.status in ("prompts", "past bound")]
    if checked:
        return min(checked, key=lambda n: n.size), "the smallest build whose prompts are checked (none is verified)"
    return None


def is_name(model: str | Path) -> bool:
    """Whether `model` is to be read as a short name: a string with no "/", not a .gguf file nor a folder that
    exists."""
    return isinstance(model, str) and parse(model).form == "name"


# what may follow a family's name in place of a build's tag: the precision it computes in, as --precision takes it
PRECISIONS = (*QUANTIZED, *FULL)


# A creator's own 8-bit build, by its tag: FAMILY:q8 takes a verified one where the family has it (d1-2b:q8 is the
# creator's Q8_0 file, 2 GB), rather than quantizing the full-precision build here (3.8 GB downloaded, unverified)
OWN_EIGHT_BIT = tuple(name for name, tag in TAGS.items() if tag.bits == 8)


def own_eight_bit(family: str) -> tuple[Named, ...]:
    """The family's own 8-bit builds (OWN_EIGHT_BIT) whose answers are verified: an unverified one is never taken over
    the full-precision build quantized here."""
    return tuple(n for n in families().get(family, ()) if n.tag in OWN_EIGHT_BIT and n.build.status == "verified")


def title(model: str | Path) -> str:
    """How a person reads a model's name: its model's own for a short name (Kettei Core for kettei-core, Kettei Core
    q8 for kettei-core:q8), else the name as given (a Hub id, a folder)."""
    if not is_name(model):
        return str(model)
    try:
        found, _ = find(str(model))
    except DecisionError:
        return str(model)
    precision = precision_of(model)
    return f"{found.model} {precision}" if precision else found.model


def precision_of(model: str | Path) -> str | None:
    """The precision a short name asks for, as in kettei-core:q8 or kettei-core:q4, when its tag is a precision
    rather than one of the family's builds (nor q8 of a family that publishes its own 8-bit build); None otherwise."""
    if not is_name(model):
        return None
    wanted = str(model).strip().lower()
    family, _, tag = wanted.partition(":")
    exact = any(n.name == wanted for n in named())
    if tag == "q8" and own_eight_bit(family):
        return None
    return tag if tag in PRECISIONS and not exact and family in families() else None


def named_precision(model: str | Path, precision: str) -> str:
    """`precision`, or the one a short name carries (kettei-core:q8) when `precision` is "auto"; a name and a
    different precision given together are refused."""
    carried = precision_of(model)
    if carried is None or precision in ("auto", carried):
        return carried or precision
    raise InvalidRequest(f"{model} runs at {carried}, and precision asks for {precision}: give one of them")


def find(name: str) -> tuple[Named, str]:
    """The build `name` names (FAMILY:TAG, a bare FAMILY for its recommended build, or FAMILY:PRECISION for its
    recommended full-precision build, in any case) and why it was chosen ("" when the tag named it); UnsupportedModel
    naming the closest names when none does."""
    wanted = name.strip().lower()
    for n in named():
        if n.name == wanted:
            return n, ""
    family, tagged, tag = wanted.partition(":")
    groups = families()
    if family in groups:
        builds = groups[family]
        if tag in REFUSED:  # what it takes instead: the precisions ippon computes from a full build, then its builds
            computed = [f"{family}:{q}" for q in QUANTIZED] if any(n.kind == "safetensors" and n.tag in FULL
                                                                   for n in builds) else []  # fmt: skip
            raise unsupported_format(
                name, spoken(list(dict.fromkeys([*computed, *(n.name for n in builds if not n.low_bit)])))
            )
        if tag == "q8" and (own := own_eight_bit(family)) and (picked := recommended(own)):
            return picked[0], f"{family}'s own 8-bit build ({picked[1]})"
        if tag in PRECISIONS:  # ippon computes it at that precision, from a build stored at full precision
            builds = tuple(n for n in builds if n.kind == "safetensors" and n.tag in FULL)
        chosen = recommended(builds) if not tagged or tag in PRECISIONS else None
        if chosen is not None:
            return chosen
        listed = ", ".join(n.name for n in groups[family])
        raise UnsupportedModel(f"{name!r}: {family}'s builds are {listed}, and {family}:q8 runs one at 8 bits")
    close = difflib.get_close_matches(wanted, [*groups, *(n.name for n in named())], n=3)
    hint = f" (did you mean {' or '.join(close)}?)" if close else ""
    raise UnsupportedModel(
        f"{name!r}: not a model name ippon knows{hint}, a Hub id (org/name[@revision]) or a folder; "
        "`ippon pull --list` lists the names"
    )


def resolve(model: str | Path) -> str | Path:
    """What loads `model`: a name's build (a Hub id, a GGUF build or an ollama:// build), or `model` as given when it
    is not a name."""
    return find(model)[0].build.load if isinstance(model, str) and is_name(model) else model


def weights_at(model: str | Path, precision: str = "auto") -> str:
    """The weights `model` at `precision` runs on: "q8", "q6" or "q4" for a build quantized as it loads (a tag such as
    kettei-core:q8 says so too), "full" for a build as published, which bf16, fp16, fp32 and auto only compute in."""
    chosen = named_precision(model, precision)
    return chosen if chosen in QUANTIZED else "full"


REFERENCES = (
    Path(__file__).resolve().parent / "references",  # in the wheel (hatch_build.py ships each build's)
    Path(__file__).resolve().parents[2] / "references",  # in a checkout
)


def reference(model: str | Path) -> Path | None:
    """The reference `ippon verify` checks the named build `model` loads against (verified.py's `reference`), from
    the wheel or a checkout; None for a model ippon names no reference for."""
    if not isinstance(model, str):
        return None
    named = find(model)[0] if is_name(model) else of_load(model)
    return reference_file(named.build.reference) if named and named.build.reference else None


def reference_file(file: str) -> Path | None:
    """A reference file by its path under references/, from the wheel or a checkout; None where neither has it."""
    return next((root / file for root in REFERENCES if (root / file).is_file()), None)


def _resolved(model: str) -> str:
    """What `model` loads (`resolve`), as text; the text itself for a name nothing resolves (an alias a server was
    started with)."""
    try:
        return str(resolve(model))
    except DecisionError:
        return model


def identity(model: str) -> str:
    """What `model` loads, as one string two names of the same model share: a short name's build, a folder's full
    path, a Hub id in lowercase whatever revision it names (a named build is pinned to the one it was verified at),
    and the text itself for anything else (an alias a server was started with)."""
    try:
        resolved = str(resolve(model))
    except DecisionError:
        return model
    path = Path(resolved).expanduser()
    return str(path.resolve()) if _exists(resolved) or path.is_absolute() else parse(resolved).unrevised.lower()


def answers_for(loads: str, asked: str) -> bool:
    """Whether a model loaded from `loads` is the one `asked` names: the same model (`identity`), and at the revision
    `asked` pins when it pins one (org/name@REVISION, in any case, as the Hub reads commits), which `loads` must load
    too. A name that pins none, a short name as typed included, takes the model at any revision."""
    pinned = parse(asked).revision
    same = identity(loads) == identity(asked)
    return same and (pinned is None or pinned.lower() == (parse(_resolved(loads)).revision or "").lower())


def of_load(load: str) -> Named | None:
    """The named build that `load` (a Hub id, GGUF build or ollama:// build, any case, at any revision) loads, if one
    does."""
    found = catalogued(load)
    return next((n for n in named() if found and n.build is found[1]), None)


def table(colour: bool = False, width: int | None = None) -> str:
    """`ippon pull --list`: every family with its licence, and each of its builds: what its weights take, how it
    came out (verified.py) and what it loads, with the recommended build marked. At a terminal (`colour`, and its
    `width` in columns) a family's line is bold, the recommended build is in the accent of what is settled, what a
    build loads is dim, and a line longer than the terminal is cut instead of wrapping under the next."""
    groups = families()
    rows: list[tuple[str, ...]] = []
    for family, builds in groups.items():
        chosen = recommended(builds)
        rows.append((f"{family} ({builds[0].model}, {builds[0].licence})",))
        for n in builds:
            mark = "*" if chosen and chosen[0] is n else " "
            rows.append((f"  {mark} {n.name}", str(n.build.weights), n.status, n.build.load))
    widths = [max(len(row[i]) for row in rows if len(row) > 1) for i in range(3)]
    lines = []
    for row in rows:
        if len(row) == 1:
            lines.append(terminal.fitted([(row[0], ("1",))], colour, width))
            continue
        name, weights, status = (cell.ljust(w) + "  " for cell, w in zip(row, widths, strict=False))
        tone = ("mint",) if name.startswith("  *") else ()
        lines.append(terminal.fitted([(name, tone), (weights + status, ()), (row[-1], ("2",))], colour, width))
    lines.append(
        "* what `ippon pull FAMILY` takes: the smallest verified build, or a verified safetensors build within "
        f"{SAFETENSORS_WITHIN:g}x of its size; else the smallest whose prompts are checked"
    )
    return "\n".join(lines)


def listing(more: Callable[[Named], dict[str, object]] = lambda named: {}) -> list[dict[str, object]]:
    """`ippon pull --list --json`: the table's families and builds as JSON, each build with what `more` adds (the
    MCP `models` tool adds whether it is downloaded and what media it reads)."""
    out: list[dict[str, object]] = []
    for family, builds in families().items():
        chosen = recommended(builds)
        out.append(
            {
                "family": family,
                "model": builds[0].model,
                "licence": builds[0].licence,
                "recommended": chosen[0].name if chosen else None,
                "why": chosen[1] if chosen else None,
                "builds": [
                    {"name": n.name, "tag": n.tag, "load": n.build.load, "weights": str(n.build.weights)}
                    | {"gigabytes": n.build.gigabytes, "status": n.build.status, "note": n.build.note}
                    | {"recommended": bool(chosen and chosen[0] is n)}
                    | more(n)
                    for n in builds
                ],
            }
        )
    return out
