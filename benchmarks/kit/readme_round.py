"""Run the README's workloads over HTTP, every runtime as shipped and ippon as installed, with a complete attempt
ledger.

Each round loads every runtime once and runs all its workloads against it, one result file each. A shape's timed
requests stop once its median is settled (its 95% interval within --precision of it, after at least --min-n and at
most --n requests, or after --cell-seconds); a later round runs only the pairs round 1 left within --close of each
other or with overlapping intervals, in the other order.

Use the local release gate through --gate before every process. All attempts stay on disk. A timed request that
overlapped another busy process is set aside in its result and the shape's next generated request is timed instead,
at most n times per shape; past that, the busy readings stay in the accepted run.
--resume continues a stopped campaign with its accepted runs. Invoke detached under the GPU lock.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import model_identity
import run as kit
import summarize

ENGINE_ATTEMPTS = 8  # runs of one runtime in one round before the campaign stops
CLIENTS = 8  # the concurrent workload's clients sending at once
# A workload: its result files' suffix, the requests it sends and the (state tokens, questions) shapes it times.
# `text` is the README's single-client pair of workloads; the others time what the README also claims: many clients
# at once through each runtime's own server, requests carrying an image or an audio clip, and one state re-sent as an
# agent re-sends its context (a new question each time, or a customer message longer), where prompt caches count.
WORKLOADS = {
    "text": ("", ["--sizes", "1000", "--questions", "1,50"], ((1000, 1), (1000, 50))),
    "typical": (".typical", ["--sizes", "500", "--questions", "3"], ((500, 3),)),
    "concurrent": (f".c{CLIENTS}", ["--sizes", "1000", "--questions", "1", "--clients", str(CLIENTS)], ((1000, 1),)),
    "image": (".image", ["--sizes", "100", "--questions", "3", "--media", "image"], ((100, 3),)),
    "video": (".video", ["--sizes", "100", "--questions", "3", "--media", "video"], ((100, 3),)),
    # an agent's screenshots: each step sends the earlier pictures again with one new one and one question, in runs of
    # 1 to 8 pictures (llama.cpp b11960 refuses a request with more than 8: "too many images, the maximum is 8")
    "pictures": (
        ".pictures",
        ["--sizes", "100", "--questions", "1", "--media", "image", "--scenario", "repeat", "--media-history", "7"],
        ((100, 1),),
    ),
    "audio": (".audio", ["--sizes", "100", "--questions", "1", "--media", "audio"], ((100, 1),)),
    # an agent asking new questions about an image or a video it already sent: the same file with every request
    "image-again": (
        ".image-again",
        ["--sizes", "100", "--questions", "1", "--media", "image", "--scenario", "repeat", "--same-media", "1"],
        ((100, 1),),
    ),
    "video-again": (
        ".video-again",
        ["--sizes", "100", "--questions", "1", "--media", "video", "--scenario", "repeat", "--same-media", "1"],
        ((100, 1),),
    ),
    "repeat": (".repeat", ["--sizes", "1000", "--questions", "1", "--scenario", "repeat"], ((1000, 1),)),
    "grow": (".grow", ["--sizes", "1000", "--questions", "1", "--scenario", "grow"], ((1000, 1),)),
    # agents deciding step by step: 16 clients at once, each a short (~300-token) one-question request
    "burst": (".c16s", ["--sizes", "300", "--questions", "1", "--clients", "16"], ((300, 1),)),
}
WRAPPED = ("tf_systemone.py", "clef.py", "pplx_decider.py")  # creators' libraries with no server
MEDIA_PAIRS = {  # the other runtime of each pair whose two arms read the medium
    "image": (
        "pplx-flash-maker-mps-bf16",
        "d1-2B-tf",
        "d1-omni-2B-tf",
        "d3-mini-tf",
        "d3-flash-tf",
        "torchcast-4b-maker-mps",
        "d1-2B-llama-Q8_0",  # llama.cpp with the build's mmproj, ippon with the same file beside the model
        "d1-omni-2B-llama-Q8_0",
        "h2o-4b-llama-Q8_0",
        "d3-mini-omlx",  # oMLX hands a request's files to these vision checkpoints
        "d3-flash-omlx",
        "d1-2B-bf16-ollama",
        "d1-omni-2B-bf16-ollama",
        "d1-2B-mxfp8-ollama",
        "d1-omni-2B-mxfp8-ollama",
    ),
    # a growing set of pictures, on every official vision build a rival serves: a one-way model (H2O's GGUF) lets a
    # runtime keep the earlier pictures' reading, and on a two-way model (d1, d3, pplx-decider v1.3) a runtime can
    # still keep each picture's encoding. pplx-decider's other runtime is Perplexity's own package. Clef-Flash stays
    # out: ippon reads it as text only, since its creators publish no reference for pictures
    "pictures": (
        "h2o-4b-llama-Q8_0",
        "pplx-flash-maker-mps-bf16",
        "d1-2B-llama-Q8_0",
        "d1-omni-2B-llama-Q8_0",
        "d1-2B-bf16-ollama",
        "d1-omni-2B-bf16-ollama",
        "d3-mini-mxfp8-ollama",
        "d3-flash-mxfp8-ollama",
        "d3-mini-omlx",  # oMLX hands a request's pictures to the d3 checkpoints, as in the one-picture workload
        "d3-flash-omlx",
    ),
    # a 4-second video, on every runtime that reads one: the creators' code, oMLX 0.7.2 on d3 (`videos`) and
    # llama.cpp b11960 on d1 (a data:video URL among `files`). Ollama 0.40.13 refuses video for every decision model
    "video": (
        "d1-2B-tf",
        "d1-omni-2B-tf",
        "d3-mini-tf",
        "d3-flash-tf",
        "d3-mini-omlx",
        "d3-flash-omlx",
        "d1-2B-llama-Q8_0",
        "d1-omni-2B-llama-Q8_0",
    ),
    # a 10-second clip on d1-Omni, the one model that hears: its creators' code, llama.cpp, and Ollama 0.40.13, which
    # takes a clip under `images` and tells it from a picture by its bytes
    "audio": ("d1-omni-2B-tf", "d1-omni-2B-llama-Q8_0", "d1-omni-2B-bf16-ollama", "d1-omni-2B-mxfp8-ollama"),
}
# the same file asked again: the pairs whose two arms read that medium
MEDIA_PAIRS["image-again"], MEDIA_PAIRS["video-again"] = MEDIA_PAIRS["image"], MEDIA_PAIRS["video"]


def workload_clients(options: list[str]) -> int:
    """How many clients a workload's options send at once (one unless --clients says otherwise)."""
    return int(options[options.index("--clients") + 1]) if "--clients" in options else 1


def workload_fields(options: list[str]) -> dict:
    """A workload's options as one entry of the client's --plan."""
    given = dict(zip(options[::2], options[1::2], strict=True))
    return {
        "sizes": given["--sizes"],
        "questions": given["--questions"],
        "clients": int(given.get("--clients", 1)),
        "media": given.get("--media"),
        "media_history": int(given.get("--media-history", 0)),
        "same_media": int(given.get("--same-media", 0)),
        "scenario": given.get("--scenario", "fresh"),
    }


def close_engines(folder: Path, plans: list, pairs: dict[str, list[list[str]]], margin: float) -> dict[str, set]:
    """Per workload, the engines of its pairs that round 1 left close: a measured cell within `margin` of the other
    runtime's, or the two medians' 95% intervals overlapping (for many clients at once, requests per second within
    `margin`). A pair that failed in round 1 is not run again."""
    out: dict[str, set] = {}
    for name, suffix, *_ in plans:
        out[name] = set()
        for other, ours in pairs[name]:
            paths = [folder / f"{engine}{suffix}.json" for engine in (other, ours)]
            if not all(path.exists() for path in paths):
                continue
            a, b = (json.loads(path.read_text()) for path in paths)
            if not (a.get("complete") and b.get("complete")) or a.get("errors") or b.get("errors"):
                continue
            if name == "concurrent":
                rates = [run["concurrent"]["requests_per_second"] for run in (a, b)]
                near = abs(rates[1] / rates[0] - 1) < margin
            else:
                near = False
                for shape in a["summary"].keys() & b["summary"].keys():
                    x, y = a["summary"][shape], b["summary"][shape]
                    overlap = (
                        x.get("median_ci_ms")
                        and y.get("median_ci_ms")
                        and (
                            max(x["median_ci_ms"][0], y["median_ci_ms"][0])
                            <= min(x["median_ci_ms"][1], y["median_ci_ms"][1])
                        )
                    )
                    near = near or abs(x["median_ms"] / y["median_ms"] - 1) < margin or bool(overlap)
            if near:
                out[name] |= {other, ours}
    return out


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def checked(
    result: dict, samples: int, warmup: int, shapes=WORKLOADS["text"][2], clients: int = 1, minimum: int = 0
) -> list[str]:
    """Why a workload's result can't be accepted: incomplete, unmonitored, or a shape short of its samples (exactly
    `samples` timed requests, or with `minimum` from `minimum` to `samples` of them: the client stops a shape once
    its median is settled)."""
    problems = []
    if not result.get("complete") or result.get("errors"):
        problems.append("incomplete or request errors")
    # Busy readings stay in the result without rejecting the run (a campaign may run while the Mac is in use);
    # each busy timed request is still replaced first, up to n times per shape.
    if result.get("activity_errors"):
        problems.append("activity monitor failure")
    for size, nq in shapes:
        timed_each = minimum if minimum and clients > 1 else samples  # many clients: --min-n each, one window
        for phase, count in (("warmup", warmup * clients), ("timed", timed_each * clients)):
            rows, aside = (
                [r for r in result.get(key, []) if r["phase"] == phase and r["size"] == size and r["nq"] == nq]
                for key in ("requests", "set_aside")
            )
            reps = sorted(r["rep"] for r in rows + aside)
            low = minimum if phase == "timed" and minimum and clients == 1 else count
            if not low <= len(rows) <= count or reps != list(range(len(reps))):
                problems.append(f"{phase} {size}x{nq}: missing samples")
            if any(r["problems"] or len(r["answers"]) != nq for r in rows):
                problems.append(f"{phase} {size}x{nq}: incomplete answers")
    return problems


def verdict(result: dict, code: int, problems: list[str], smoke: bool) -> str:
    """What a finished attempt at one workload means for the round: "failed", the runtime's own failure (request
    errors such as running out of memory, or its process ending during the workload), kept as the step's result while
    the round goes on; "accepted"; or "retry", busy readings or a guard stop (exit 143), kept beside the round while
    the workload runs again. On the M5 Pro, 27 Oct, Cloudflare's own Clef-Flash code ran out of MPS memory on every
    50-question request from the fifth on, in three attempts: a failure, not a retry."""
    if smoke:
        return "accepted" if code == 0 and not problems else "failed"
    if (
        result.get("errors")
        or (code not in (0, 143) and not result.get("complete"))
        or (code == 0 and any(p != "activity monitor failure" for p in problems))
    ):
        return "failed"
    return "accepted" if code == 0 and not problems else "retry"


def script(path: Path) -> list[str]:
    """The command running a gate or guard: an executable as it is (/usr/bin/true), else its text with sh."""
    return [str(path)] if os.access(path, os.X_OK) else ["sh", str(path)]


def execute(cmd: list[str], log: Path, env: dict, gate: Path, guard: Path | None) -> int:
    """Run one engine's client after the release gate, its output in `log`; stop it when `guard` prints stop (exit
    143). Its whole process group is drained before this returns, so no runner overlaps the next engine."""
    subprocess.run(script(gate), check=True)
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        while process.poll() is None:
            if guard and subprocess.check_output(script(guard), text=True).strip() == "stop":
                return 143
            time.sleep(10)
        return process.wait()
    finally:
        # The adapter owns its own session. Drain its servers too, even when the leader exits first.
        # Never let a surviving runner overlap the next engine or invalidate its memory/readiness gate.
        # macOS refuses a signal to a group left with only zombies (EPERM, not ESRCH): nothing runs there, so
        # it counts as drained. Unhandled, it ended the 25 Oct campaign after a guard stop of a 9B tune.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                process.poll()
                try:
                    os.killpg(process.pid, 0)
                except (ProcessLookupError, PermissionError):
                    break
                time.sleep(0.1)
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait(timeout=30)


def resumed(
    steps: list, folder: Path, rejected: Path, ledger: list[dict], number: int, accept
) -> tuple[list[dict], set[str]]:
    """A resumed round's steps it already settled: those the ledger recorded as failed, and results on disk that
    `accept` passes; a result it refuses was interrupted, and moves beside the round to run again."""
    entries, done = [], set()
    for step, _, shapes, clients in steps:
        failed = [e for e in ledger if e.get("round") == number and e.get("engine") == step and e.get("failed")]
        out = folder / f"{step}.json"
        if failed:
            entries.append({k: failed[-1][k] for k in ("engine", "exit", "problems", "failed")})
            done.add(step)
        elif out.exists():
            if accept(json.loads(out.read_text()), shapes, clients):
                entries.append({"engine": step, "exit": 0, "problems": []})
                done.add(step)
            else:
                rejected.mkdir(exist_ok=True)
                out.rename(rejected / f"{step}-interrupted.json")
    return entries, done


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--envs", type=Path, required=True)
    p.add_argument("--llama-server", type=Path, required=True)
    p.add_argument("--ollama", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True, help="local script: exit 0 only when release checks pass")
    p.add_argument("--guard", type=Path, help="local script printing stop when the active process must be restarted")
    p.add_argument("--cache", type=Path, required=True, help="a new scratch cache; never the live store")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=2, help="the most rounds; past the first, only close pairs run")
    p.add_argument("--n", type=int, default=20, help="the most timed requests per shape")
    p.add_argument("--min-n", type=int, default=6, help="timed requests per shape before its median can settle")
    p.add_argument("--precision", type=float, default=0.03, help="settled: the median's 95%% interval within this")
    p.add_argument("--cell-seconds", type=float, default=90.0, help="settled anyway after this long per shape")
    p.add_argument("--close", type=float, default=0.10, help="a pair runs another round when this close in round 1")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="check each engine on both workloads; no publishable timings")
    p.add_argument("--only", nargs="+", help="engine names, for an explicit diagnostic smoke")
    p.add_argument(
        "--pairs",
        nargs="+",
        help="a targeted round: only these pairs (their other runtime's engine) and models timed alone (their ippon "
        "engine); the newest campaign holding a pair's results is the one the summary takes",
    )
    p.add_argument("--resume", action="store_true", help="continue the stopped campaign in --out")
    p.add_argument(
        "--workloads",
        type=lambda value: value.split(","),
        default=["text"],
        help=f"comma-separated, of {', '.join(WORKLOADS)} (default: text)",
    )
    args = p.parse_args()

    def terminate(_signum, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    if unknown := [name for name in args.workloads if name not in WORKLOADS]:
        sys.exit(f"unknown workloads: {', '.join(unknown)}")
    args.models, args.envs = args.models.expanduser().resolve(), args.envs.expanduser().resolve()
    args.cache, args.out = args.cache.expanduser().resolve(), args.out.expanduser().resolve()
    if args.resume:
        if args.smoke or not (args.out / "CAMPAIGN.json").is_file():
            sys.exit("--resume continues a stopped measured campaign in --out")
    elif args.out.exists() and any(args.out.iterdir()):
        sys.exit("output directory is not empty; preserve it and choose a new campaign directory")
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.cache == Path.home() / "Library/Caches/ippon":
        sys.exit("choose a scratch cache")
    table = kit.engines(args.models, args.envs, args.llama_server.resolve(), args.ollama.resolve())
    pairs = [
        (group, model, other, ours + "-serve" if ours.endswith("-ippon") else ours)
        for group, model, other, ours in summarize.PAIRS
    ]
    alone = [engine + "-serve" for _, engine in summarize.ALONE]
    if args.pairs:
        known = {other for *_, other, _ in pairs} | {
            engine for _, engine in summarize.ALONE + summarize.SMALL_MAC_ALONE
        }
        if unknown := sorted(set(args.pairs) - known):
            sys.exit(f"unknown pairs: {', '.join(unknown)}")
        pairs = [pair for pair in pairs if pair[2] in args.pairs]
        alone = [engine + "-serve" for _, engine in summarize.ALONE + summarize.SMALL_MAC_ALONE if engine in args.pairs]

    def wrapped(engine: str) -> bool:
        spec = table[engine]
        return any(Path(part).name in WRAPPED for part in spec.get("http_cmd", spec["cmd"]))

    def chosen(workload: str) -> list[tuple[str, str, str, str]]:
        if workload in ("concurrent", "burst"):  # a wrapper's queue is the kit's: creators' libraries sit out
            return [pair for pair in pairs if not wrapped(pair[2]) and not wrapped(pair[3])]
        if workload in MEDIA_PAIRS:
            return [pair for pair in pairs if pair[2] in MEDIA_PAIRS[workload]]
        return pairs

    def engines_of(name: str) -> list[str]:  # a workload's engines: its pairs' (ippon's first), then those alone
        paired = (e for *_, other, ours in chosen(name) for e in (ours, other))
        return list(dict.fromkeys([*paired, *(alone if name not in MEDIA_PAIRS else ())]))

    plans = [(name, *WORKLOADS[name], engines_of(name)) for name in args.workloads]
    order = list(dict.fromkeys(engine for *_, engines in plans for engine in engines))
    if args.only:
        if not args.smoke:
            sys.exit("--only is diagnostic and requires --smoke")
        plans = [(*plan[:-1], [e for e in plan[-1] if e in args.only]) for plan in plans]
        order = list(dict.fromkeys(engine for *_, engines in plans for engine in engines))
    for engine in order:
        missing = [str(path) for path in table[engine]["needs"] if not path.exists()]
        if missing:
            sys.exit(f"{engine}: missing {missing}")

    def git(*command: str) -> str:  # from the repository's root, which the library's paths below are relative to
        return subprocess.check_output(["git", *command], cwd=kit.HERE.parents[1], text=True).strip()

    manifest_file = args.out / "CAMPAIGN.json"
    state_file = args.out / "ATTEMPTS.json"
    installed = json.loads(
        subprocess.check_output(
            [
                str(args.envs / "ippon/bin/python"),
                "-I",
                "-c",
                "import json; from ippon.machine import describe; from ippon.choices import RULE; "
                "print(json.dumps({**describe(), 'kernel_rule': RULE}))",
            ],
            text=True,
        )
    )
    # A commit that changes only the kit keeps the installed library, and with it its kernel choices.
    head, commit = git("rev-parse", "HEAD"), git("rev-parse", installed["revision"].rstrip("+"))
    library = ["src", "pyproject.toml", "hatch_build.py"]
    if installed["revision"].endswith("+") or git("diff", "--name-only", commit, head, "--", *library):
        sys.exit("install the current clean commit in the benchmark environment first")
    if not args.smoke and git("status", "--porcelain", "--", str(kit.HERE)):
        sys.exit("commit the benchmark kit first")
    manifest = {
        "kit_commit": head,
        "library_commit": commit,
        "generator_sha256": digest(kit.HERE / "gen_requests.py"),
        "date": dt.date.today().isoformat(),
        "machine": installed,
        "workloads": [
            {
                "name": name,
                "options": options,
                "pairs": [[other, ours] for *_, other, ours in chosen(name)],
                "alone": alone if name not in MEDIA_PAIRS else [],
            }
            for name, _, options, *_ in plans
        ],
        "samples": args.n,
        "min_samples": args.min_n,
        "precision": args.precision,
        "cell_seconds": args.cell_seconds,
        "close": args.close,
        "warmup": args.warmup,
        "rounds": args.rounds,
    }
    ledger: list[dict] = []
    env = {**os.environ, "IPPON_CACHE": str(args.cache), "OLLAMA_MODELS": str(args.models / "ollama")}

    def save():
        manifest_file.write_text(common.dumps(manifest, indent=2) + "\n")
        state_file.write_text(common.dumps(ledger, indent=2) + "\n")

    identity = args.out / "MODEL_BUILDS.json"
    if args.resume:
        saved = json.loads(manifest_file.read_text())
        same = {
            k: manifest[k]
            for k in ("library_commit", "generator_sha256", "samples", "min_samples", "precision", "cell_seconds",
                      "close", "warmup", "rounds")
        }  # fmt: skip
        rule = saved["machine"].get("kernel_rule")
        if any(saved.get(k) != v for k, v in same.items()) or rule != installed["kernel_rule"]:
            sys.exit("--resume needs the same library, request generator and sample counts")
        print("Checking model build fingerprints", flush=True)
        # A pair dropped from the kit after it started is not run again; every build still run is the one recorded.
        recorded = json.loads(identity.read_text())
        current = json.loads(common.dumps(model_identity.inventory(table, order, args.models)))
        if any(recorded["engines"].get(e) != current["engines"][e] for e in order) or any(
            recorded["files"].get(path) != entry for path, entry in current["files"].items()
        ):
            sys.exit("a model build changed since the campaign recorded it")
        manifest, ledger = saved, json.loads(state_file.read_text())
        ledger.append({"kind": "resume", "kit_commit": head, "time": dt.datetime.now().astimezone().isoformat()})
        env["IPPON_BENCH_CAMPAIGN"] = str(manifest_file)
    elif not args.smoke:
        print("Recording model build fingerprints", flush=True)
        identity.write_text(common.dumps(model_identity.inventory(table, order, args.models), indent=2) + "\n")
        manifest["model_builds"] = {"file": identity.name, "sha256": digest(identity)}
    save()
    if not args.smoke:
        env["IPPON_BENCH_CAMPAIGN"] = str(manifest_file)
    adaptive = [] if args.smoke else ["--min-n", str(args.min_n), "--precision", str(args.precision),
                                      "--cell-seconds", str(args.cell_seconds)]  # fmt: skip
    samples, warmup = (1, 0) if args.smoke else (args.n, args.warmup)
    minimum = 0 if args.smoke else args.min_n

    def accept(result: dict, shapes, clients: int) -> bool:
        return not checked(result, samples, warmup, shapes, clients, minimum)

    pair_engines = {name: [[other, ours] for *_, other, ours in chosen(name)] for name, *_ in plans}
    for round_index in range(1 if args.smoke else args.rounds):
        number = round_index + 1
        if args.resume and (args.out / f"r{number}").is_dir():
            continue
        round_plans = plans
        if number > 1:  # only the pairs round 1 left close run again, in the other order
            close = close_engines(args.out / "r1", plans, pair_engines, args.close)
            round_plans = [(*plan[:-1], [e for e in plan[-1] if e in close[plan[0]]]) for plan in plans]
            if not any(plan[-1] for plan in round_plans):
                ledger.append({"kind": "round-skipped", "round": number, "why": f"no pair within {args.close:.0%}"})
                save()
                break
        folder = args.out / f"attempt-r{number}"
        rejected = args.out / f"rejected-r{number}"
        folder.mkdir(exist_ok=args.resume)
        engines = list(dict.fromkeys(e for *_, chosen_engines in round_plans for e in chosen_engines))
        if round_index % 2:
            engines.reverse()
        entries, scheduled = [], []
        for engine in engines:
            spec = table[engine]
            mine = [
                (f"{engine}{suffix}", options, shapes, workload_clients(options))
                for name, suffix, options, shapes, chosen_engines in round_plans
                if engine in chosen_engines
            ]
            scheduled += [step for step, *_ in mine]
            done: set[str] = set()
            if args.resume:
                settled, done = resumed(mine, folder, rejected, ledger, number, accept)
                entries += settled
            first = 1 + max(
                (e["attempt"] for e in ledger if e.get("round") == number and e.get("engine") == engine), default=0
            )
            for attempt in range(first, first + ENGINE_ATTEMPTS):
                todo = [w for w in mine if w[0] not in done]
                if not todo:
                    break
                plan = [{"out": str(folder / f"{step}.json"), **workload_fields(options)} for step, options, *_ in todo]
                cmd = [*spec.get("http_cmd", spec["cmd"]), "--engine", engine, "--http", "--plan", json.dumps(plan),
                       "--n", str(samples), "--warmup", str(warmup), *adaptive]  # fmt: skip
                if "server" in spec:
                    cmd += ["--", *spec["server"]]
                print(f"ROUND {number}: {engine} attempt {attempt}: {', '.join(s for s, *_ in todo)}", flush=True)
                code = execute(cmd, folder / f"{engine}.log", {**env, **spec.get("env", {})}, args.gate, args.guard)
                retry = []
                for step, _, shapes, clients in todo:
                    out = folder / f"{step}.json"
                    if not out.exists() and code != 0:
                        if args.smoke:
                            sys.exit(f"SMOKE FAILED {step}: exit {code}; see {folder / f'{engine}.log'}")
                        if code == 143:  # the guard stopped it before this workload began: it runs again
                            retry.append(step)
                            continue
                        # the runtime refused or crashed before this workload began: running it again would too
                        said = [line for p in (folder / f"{step}.server.log", folder / f"{engine}.log") if p.exists()
                                for line in p.read_text().splitlines() if line.strip()]  # fmt: skip
                        outcome = {"engine": step, "exit": code, "problems": ["did not start"],
                                   "failed": (said[-1] if said else f"exit {code}")[:500]}  # fmt: skip
                        entries.append(outcome)
                        ledger.append({"kind": "engine-attempt", "round": number, "attempt": attempt, **outcome})
                        print(f"FAILED {step}: kept as its result: {outcome['failed'][:200]}", flush=True)
                        done.add(step)
                        continue
                    result = json.loads(out.read_text()) if out.exists() else {}
                    problems = checked(result, samples, warmup, shapes, clients, minimum)
                    print(f"FINISHED {step}: exit={code}; {problems or 'complete'}", flush=True)
                    outcome = {"engine": step, "exit": code, "problems": problems}
                    kind = verdict(result, code, problems, args.smoke)
                    if kind == "failed":
                        errors = result.get("errors") or [{"error": f"exit {code}; see {folder / f'{engine}.log'}"}]
                        outcome["failed"] = errors[0]["error"][:500]
                        entries.append(outcome)
                        ledger.append({"kind": "engine-attempt", "round": number, "attempt": attempt, **outcome})
                        print(f"FAILED {step}: kept as its result: {outcome['failed'][:200]}", flush=True)
                        done.add(step)
                    elif kind == "accepted":
                        entries.append(outcome)
                        done.add(step)
                    else:  # "retry"
                        rejected.mkdir(exist_ok=True)
                        out.rename(rejected / f"{step}-{attempt}.json")
                        ledger.append({"kind": "engine-attempt", "round": number, "attempt": attempt, **outcome,
                                       "busy": result.get("busy", [])})  # fmt: skip
                        retry.append(step)
                ledger.append({"kind": "engine-run", "round": number, "attempt": attempt, "engine": engine,
                               "exit": code, "retry": retry})  # fmt: skip
                save()
                if retry:  # this attempt's logs stay beside the round with its rejected results
                    rejected.mkdir(exist_ok=True)
                    for path in [folder / f"{engine}.log", *(folder / f"{s}.server.log" for s, *_ in mine)]:
                        if path.exists():
                            path.rename(rejected / f"{path.name[: -len('.log')]}-{attempt}.log")
            else:
                sys.exit(f"{engine}: {ENGINE_ATTEMPTS} runs in round {number} were stopped by the guard")
        passed = not args.smoke or all(not e.get("failed") for e in entries)
        ledger.append({"kind": "smoke" if args.smoke else "round", "round": number, "accepted": passed,
                       "scheduled": scheduled, "engines": entries})  # fmt: skip
        save()
        if args.smoke:
            if not passed:
                sys.exit("SMOKE FAILED: fix the failed arms before timing")
            print("SMOKE COMPLETE", flush=True)
            return
        folder.rename(args.out / f"r{number}")
    print("README ROUND COMPLETE", flush=True)


if __name__ == "__main__":
    main()
