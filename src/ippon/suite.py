"""`ippon suite`: everything ippon measures on a Mac, in one command, into one folder to send back.

    ippon suite --models ~/models --out suite-m2 [--quick] [--arm mlx-vlm-model=PYTHON] [--only d1-2b,d3-mini]

For every model of SUITE (each named build of verified.py with a reference that downloads from the Hub, and a few
EXTRAS) whose directory is under --models, or a GGUF build, which downloads on its first load: `verify` against its
creator's fp32 reference, `verify --invariance`, `bench` (fresh states of 300 to 4,000 words with 1 and 5 questions,
20 to 100 words with one question, and a state asked again), an A/B of ippon's kernel choices against MLX's own
matmuls (`--switch gemm=mlx`), which tells whether the choices made from this chip's generation and core count help
here, and `bench --compare` against the arms given that fit the model. Each step runs as its own process, one after
another, so one model's memory is gone before the next loads; a step that fails is recorded and the suite goes on.
Before anything it records `doctor` and `info`. --quick runs fewer shapes and requests and skips the invariance
checks: a first look in about half an hour.

Out: one JSON per step (`<model>.<step>.json`), the logs (`<model>.<step>.log`), `suite.json` (the machine and every
step's outcome) and `summary.md`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import bench, choices, machine, names, processes
from ._portable import dumps, transcript
from .compare_arms import ARMS
from .compare_arms.lookup import fitting
from .errors import DecisionError, InvalidRequest
from .processes import spawn
from .profiles import resolve


@dataclass(frozen=True)
class Model:
    name: str  # the suite's name for it, used in file names: its build's short name, as FAMILY-TAG
    family: str  # its family's short name, which --only takes for every build of it
    directory: str  # its folder under --models, or a GGUF build's "repo:QUANT" (fetched on its first load)
    reference: str  # under references/: the creator's fp32 answers (or a tokens-only file)
    hub: str | None  # what to download it from, org/name@revision, for a build that is a Hub folder (scripts/suite.sh)
    # a build past its reference's bound as published (verified.py's "past bound": its own quantization), its
    # distance as recorded: verify matches the record within RECORDED_SLACK of it, and fails past that
    recorded: float | None = None

    @property
    def gguf(self) -> bool:
        return names.parse(self.directory).form == "gguf"


def _model(n: names.Named, reference: str = "", suffix: str = "") -> Model:
    """A named build as the suite runs it: a Hub folder in --models named as its repository, or a GGUF build by its
    spec; checked against its own reference, or `reference` for a `suffix`ed step of it (EXTRAS)."""
    load = names.parse(n.build.load)
    gguf = load.form == "gguf"
    return Model(
        n.name.replace(":", "-") + (f"-{suffix}" if suffix else ""),
        n.family,
        load.unrevised if gguf else load.repo.rpartition("/")[2],
        reference or n.build.reference,
        None if gguf or suffix else n.build.load,  # an extra reads its build's folder
        None if suffix else n.build.distance,
    )


# the suite's models besides one per build: a build checked against another reference of its creator's (videos)
EXTRAS = (
    ("d1-2b:bf16", "d1/d1-2B-video.json.gz", "video"),
    ("d1-omni-2b:bf16", "d1/d1-omni-2B-video.json.gz", "video"),
)
# every named build ippon downloads from the Hub (an Ollama build needs Ollama's own pull) with a reference
# (verified.py), and the EXTRAS
SUITE = tuple(_model(n) for n in names.named() if n.kind != "ollama" and n.build.reference) + tuple(
    _model(names.find(name)[0], reference, suffix) for name, reference, suffix in EXTRAS
)


RECORDED_SLACK = 1.1  # a recorded build's verify matches the record up to 10% past its recorded distance
REPEAT = ["--sizes", "1000", "--questions", "3", "--scenario", "repeat"]  # an agent asking about one state again


def benches(quick: bool) -> list[tuple[str, list[str]]]:
    """The bench steps: (name, bench options)."""
    if quick:
        return [("bench", ["--sizes", "300,2000", "--questions", "1,5", "--n", "10"])]
    return [
        ("bench", ["--sizes", "300,1000,2000,4000", "--questions", "1,5", "--n", "20"]),
        ("bench-short", ["--sizes", "20,50,100", "--questions", "1", "--n", "20"]),
        ("bench-repeat", [*REPEAT, "--n", "20"]),
    ]


def compares(quick: bool, arms: Sequence[str]) -> list[tuple[str, list[str]]]:
    """The steps timing `arms` (those that fit the model) against ippon, none without one: (name, bench
    options)."""
    if not arms:
        return []
    counts = ["--n", "10", "--rounds", "2"] if quick else ["--n", "20", "--rounds", "4"]
    sizes = "300,2000" if quick else "300,1000,4000"
    return [
        ("compare", ["--sizes", sizes, "--questions", "1,5", *counts, "--compare", *arms]),
        ("compare-repeat", [*REPEAT, *counts, "--compare", *arms]),
    ]


def run(
    models_dir: str,
    out: str,
    quick: bool = False,
    arms: Sequence[str] = (),
    only: Sequence[str] = (),
    log: Any = print,
    kernels: bool | None = None,
) -> dict[str, Any]:
    """`kernels`: whether ippon's matmul choices run on this Mac, so their A/B means something (default: an M5 with
    its neural accelerators, or an M1 or M2, which unpack MXFP8 matrices); elsewhere both arms are MLX's own matmuls
    and the A/B is recorded as skipped."""
    if kernels is None:
        profile = resolve()
        kernels = choices.active(profile) or choices.older(profile)
    root, folder = Path(models_dir).expanduser(), Path(out).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    unknown = [a.split("=")[0] for a in arms if a.split("=")[0] not in ARMS]
    if unknown:
        raise InvalidRequest(f"unknown arms {unknown}: arms are {', '.join(ARMS)}")
    me = processes.command()
    report: dict[str, Any] = {
        "machine": machine.describe(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": quick,
    }
    for name, command in (("doctor", ["doctor"]), ("info", ["info"])):
        report[name] = spawn([*me, *command]).stdout
    steps: list[dict[str, Any]] = []
    wanted = {name.lower() for name in only}
    for model in SUITE:
        if wanted and not wanted & {model.name, model.family}:
            continue
        path = Path(model.directory) if model.gguf else root / model.directory
        if not model.gguf and not path.is_dir():
            steps.append({"model": model.name, "step": "find", "ok": False, "note": f"no {path}"})
            log(f"{model.name}: not under {root}, skipped")
            continue
        # (step, command, its JSON: printed ("stdout"), written by the step ("file") or none); `timed` holds the bench
        # steps, each of which writes <model>.<step>.json
        todo: list[tuple[str, list[str], str | None]] = []
        timed: list[tuple[str, list[str]]] = []
        reference = [
            "--reference",
            str(names.reference_file(model.reference) or names.REFERENCES[-1] / model.reference),
        ]
        todo.append(("verify", ["verify", str(path), *reference, "--json"], "stdout"))
        if not quick:
            todo.append(("invariance", ["verify", str(path), *reference, "--invariance", "--json"], "stdout"))
        sizes = ["--sizes", "100,1000,4000", "--questions", "1,5", "--n", "10", "--rounds", "2" if quick else "4"]
        if kernels:
            timed.append(("ab-kernels", [*sizes, "--ab", "--switch gemm=mlx", "--switch gemm=kernel"]))
        else:  # recorded in its place, not run
            todo.append(("ab-kernels", [], None))
        fitting = [a for a in arms if a.split("=")[0] in _arms_of(path)]
        timed += benches(quick) + compares(quick, fitting)
        for step, options in timed:
            target = str(folder / f"{model.name}.{step}.json")
            todo.append((step, ["bench", str(path), *options, "--out", target], "file"))
        for step, command, result in todo:
            if not command:
                note = "skipped: no neural accelerators, so both arms would be MLX's own matmuls"
                steps.append({"model": model.name, "step": step, "ok": True, "note": note})
                continue
            steps.append(_step(me, model.name, step, command, result, folder, log))
            if step == "verify":
                as_recorded(steps[-1], model.recorded)
            (folder / "suite.json").write_text(dumps({**report, "steps": steps}, indent=1))
    report |= {"steps": steps, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (folder / "suite.json").write_text(dumps(report, indent=1))
    (folder / "summary.md").write_text(transcript(summary(report, folder)))
    return report


def as_recorded(entry: dict[str, Any], recorded: float | None) -> None:
    """A failed verify of a build past its bound as published, whose worst is within RECORDED_SLACK of its recorded
    distance, counted as matching the record (`recorded` in the entry); a worse one stays a failure."""
    worst = entry.get("worst")
    if recorded and entry.get("passed") is False and worst is not None and worst <= recorded * RECORDED_SLACK:
        entry |= {"ok": True, "passed": True, "recorded": recorded}


def _arms_of(path: Path) -> tuple[str, ...]:
    try:
        return fitting(path)
    except DecisionError:
        return ()


def _step(
    me: list[str], model: str, step: str, command: list[str], result: str | None, folder: Path, log: Any
) -> dict[str, Any]:
    log(f"{model}: {step}")
    started = time.perf_counter()
    done = spawn([*me, *command])
    seconds = round(time.perf_counter() - started, 1)
    (folder / f"{model}.{step}.log").write_text(transcript(done.stdout + "\n" + done.stderr))
    entry: dict[str, Any] = {"model": model, "step": step, "ok": done.returncode == 0, "seconds": seconds}
    if result == "stdout" and done.stdout.strip().startswith("{"):
        data = json.loads(done.stdout)
        (folder / f"{model}.{step}.json").write_text(dumps(data, indent=1))
        entry |= {"passed": data.get("passed"), "worst": data.get("worst")}
    elif done.returncode != 0:
        entry["note"] = done.stderr.strip()[-500:]
    return entry


def summary(report: dict[str, Any], folder: Path) -> str:
    m = report["machine"]
    lines = [
        f"# ippon suite: {m['chip']}, {m['memory_gib']} GiB, macOS {m['macos']}",
        "",
        f"MLX {m['mlx']}, ippon {m['ippon']} ({m['revision']}); {report['started']} to "
        f"{report.get('finished', '?')}{' (quick)' if report['quick'] else ''}",
        "",
        "| model | step | outcome | seconds |",
        "|---|---|---|---:|",
    ]
    for s in report["steps"]:
        outcome = "ok" if s["ok"] else "failed"
        if s.get("passed") is not None:
            outcome = f"{'passed' if s['passed'] else 'FAILED'}, worst {s['worst']:.4f}" if s.get("worst") else outcome
            if s.get("recorded"):
                outcome = (
                    f"as recorded, worst {s['worst']:.4f} (past its bound by its own quantization: {s['recorded']})"
                )
        lines.append(f"| {s['model']} | {s['step']} | {outcome} | {s.get('seconds', '')} |")
    for path in sorted(folder.glob("*.bench*.json")):
        data = json.loads(path.read_text())
        lines += ["", f"## {path.stem}", "", "| scenario | words | q | tokens | median ms | 95% CI | p95 ms |"]
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for c in data["cells"]:
            lo, hi = c["median_ci_ms"]
            cell = f"| {bench.label(c)} | {c['words']} | {c['questions']} | {c['input_tokens']} | {c['median_ms']}"
            lines.append(f"{cell} | {lo} to {hi} | {c['p95_ms']} |")
    for path in sorted(folder.glob("*.compare*.json")):
        data = json.loads(path.read_text())
        lines += ["", f"## {path.stem}", "", "Speed change: positive means ippon is faster.", ""]
        for c in data["cells"]:
            for arm, a in c["arms"].items():
                ours = c["ippon"]["median_ms"]
                lines.append(
                    f"- {c['words']} words, {c['questions']} q: ippon {ours} ms, {arm} {a['median_ms']} ms, "
                    f"speed change {a['speedup'] - 1:+.1%} ({a['verdict']})"
                )
    return "\n".join(lines) + "\n"
