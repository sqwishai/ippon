"""What every adapter shares (stdlib only): machine state, process memory, the run loop and the result file.

An adapter supplies `decide(request) -> (answers, tokens, extra)` where `request` is one entry of
gen_requests.plan with its `systemone` payload, `answers` is {question id: {outcome: probability}}
and `tokens` is the engine's own prompt-token count (or None). `run` warms up, times every request on the
client side (time.perf_counter around the whole call, rendering and tokenization included), checks every
answer is a distribution, and writes results/<engine>.json.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import queue
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import wave
import zlib
from pathlib import Path
from runpy import run_path

import gen_requests as g
from activity import Monitor
from http_adapter import run_http

# Load just the stdlib serializer without importing ippon or changing runtime module discovery.
dumps = run_path(str(Path(__file__).resolve().parents[2] / "src/ippon/_portable.py"))["dumps"]

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def _stop(signum: int, frame: object) -> None:
    sys.exit(128 + signum)


# A SIGTERM ends a kit process as an error would, so its `finally` stops the servers it started: by default it ended
# the process at once, and an Ollama server an adapter had started kept running, idle, for days.
signal.signal(signal.SIGTERM, _stop)
HTTP_TRANSPORT_RETRIES = 2
HTTP_RETRY_BACKOFF = 0.05


# ------------------------------------------------------------------------------------------ machine state


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except Exception as exc:
        return f"<{exc!r}>"


def machine_state() -> dict:
    """Capture environment readings at the start and end of a run."""
    pm = sh("pmset", "-g")
    mode = next((ln.split()[-1] for ln in pm.splitlines() if ln.strip().startswith("powermode")), None)
    return {
        "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "powermode": mode,  # 0 automatic, 1 low power, 2 high power
        "pmset_g": pm,
        "power_source": sh("pmset", "-g", "batt").splitlines()[:1],
        "therm": sh("pmset", "-g", "therm"),
        "swapusage": sh("sysctl", "-n", "vm.swapusage"),
        "memory_free_pct": next(
            (ln.split(":")[-1].strip() for ln in sh("memory_pressure").splitlines() if "free percentage" in ln), None
        ),
        "load_avg": os.getloadavg(),
    }


def power_now() -> str:
    """Return the execution-mode marker recorded for each request shape."""
    pm = sh("pmset", "-g")
    mode = next((ln.split()[-1] for ln in pm.splitlines() if ln.strip().startswith("powermode")), "?")
    return f"{mode}/{'AC' if 'AC Power' in sh('pmset', '-g', 'batt') else 'battery'}"


def device_info() -> dict:
    gpu = sh("system_profiler", "SPDisplaysDataType")
    cores = next((ln.split(":")[-1].strip() for ln in gpu.splitlines() if "Total Number of Cores" in ln), None)
    return {
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0),
        "gpu_cores": cores,
        "macos": sh("sw_vers", "-productVersion"),
        "macos_build": sh("sw_vers", "-buildVersion"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


# ----------------------------------------------------------------------------------------- process memory


class _RusageV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
            "cpu_time_qos_default",
            "cpu_time_qos_maintenance",
            "cpu_time_qos_background",
            "cpu_time_qos_utility",
            "cpu_time_qos_legacy",
            "cpu_time_qos_user_initiated",
            "cpu_time_qos_user_interactive",
            "billed_system_time",
            "serviced_system_time",
            "logical_writes",
            "lifetime_max_phys_footprint",
            "instructions",
            "cycles",
            "billed_energy",
            "serviced_energy",
            "interval_max_phys_footprint",
            "runnable_time",
        )
    ]


def footprint(pid: int | None = None) -> dict | None:
    """A process's physical footprint now and its lifetime peak (bytes), as Activity Monitor's "Memory":
    on Apple silicon it includes the GPU buffers (Metal/MPS/MLX) the process owns."""
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        info = _RusageV4()
        if lib.proc_pid_rusage(ctypes.c_int(pid or os.getpid()), ctypes.c_int(4), ctypes.byref(info)) != 0:
            return None
        return {"phys_footprint": info.phys_footprint, "lifetime_max_phys_footprint": info.lifetime_max_phys_footprint}
    except Exception:
        return None


# --------------------------------------------------------------------------------------------- the run


def add_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--engine", required=True, help="the name of this engine/config: results/<engine>.json")
    p.add_argument("--n", type=int, default=5, help="timed requests per (size, questions)")
    p.add_argument("--warmup", type=int, default=2, help="untimed requests per (size, questions) before timing")
    p.add_argument("--sizes", default=",".join(map(str, g.SIZES)))
    p.add_argument("--questions", default=",".join(map(str, g.QUESTION_COUNTS)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-input",
        type=int,
        required=True,
        help="the model's longest input in tokens: sizes whose prompt would pass it are skipped",
    )
    p.add_argument("--out", help="result file (default results/<engine>.json)")
    p.add_argument("--http", action="store_true", help="time HTTP, wrapping libraries that have no native server")
    p.add_argument("--gap", type=float, default=0.0, help="seconds of idle before each timed request")
    p.add_argument(
        "--clients",
        type=int,
        default=1,
        help="clients sending at once, each its requests back to back, to the runtime's own server",
    )
    p.add_argument("--media", choices=sorted(MEDIA), help="attach a fresh image, audio clip or video to every request")
    p.add_argument(
        "--media-history",
        type=int,
        default=0,
        help="with --media: also attach the shape's earlier media, oldest first, before the request's own, as an "
        "agent sends back its screenshots each step; after this many earlier ones a new run of media starts, so "
        "every request is the one before it plus one (default 0: its own alone)",
    )
    p.add_argument(
        "--same-media",
        type=int,
        default=0,
        help="1, with --media and --scenario repeat: one image, clip or video per shape, made once and sent with "
        "every request of its run, as an agent asks new questions about media it already sent (default 0)",
    )
    p.add_argument(
        "--scenario",
        choices=g.SCENARIOS,
        default="fresh",
        help="fresh: a new state every request; repeat, grow: one state re-sent with new questions or a message "
        "more; lines: a yes/no question about each numbered line of one state",
    )
    p.add_argument(
        "--plan",
        help="a JSON list of workloads ({out, sizes, questions, clients, media, scenario}) timed one after another "
        "against the runtime loaded once, each written to its own result file",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=0,
        help="stop a shape's timed requests once it has this many and its median is settled (--precision, "
        "--cell-seconds); never more than --n (0: always --n)",
    )
    p.add_argument(
        "--precision",
        type=float,
        default=0.03,
        help="with --min-n: settled when the 95%% interval of the median is within this fraction of it",
    )
    p.add_argument("--cell-seconds", type=float, default=0.0, help="with --min-n: also settled after this long")
    return p


# A photo-sized image and a ten-second clip at the 16 kHz d1-omni's speech encoder reads.
MEDIA = {"image": (1024, 768), "audio": 10, "video": (1280, 720, 4)}  # a video: width, height, seconds at 30 fps


def media_file(kind: str, key: str) -> str:
    """One request's image, clip or video as a data URL, its pixels or samples derived from `key`: every request
    carries media no runtime has seen, so none can reuse an earlier request's encoding, and every runtime gets the
    same bytes. Built before the request's timer starts."""
    mark = hashlib.sha256(key.encode()).digest()  # written into the media themselves: no two requests share them
    seed = int.from_bytes(mark[:4], "big")
    if kind == "video":  # FFmpeg's moving test pattern with seeded noise, H.264 in MP4 as a phone saves a clip
        width, height, seconds = MEDIA["video"]
        # FFmpeg takes a noise seed up to 2**31 - 1
        source = (
            f"testsrc2=size={width}x{height}:rate=30:duration={seconds},noise=alls=3:allf=t:all_seed={seed % 2**31}"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "clip.mp4")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", source, "-c:v", "libx264", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                check=True,
            )  # fmt: skip
            with open(path, "rb") as clip:
                return "data:video/mp4;base64," + base64.b64encode(clip.read()).decode()
    if kind == "image":
        width, height = MEDIA["image"]
        pattern = bytes((i * 7 + (i // 97) * 31 + seed) % 256 for i in range(3 * width + 3 * height))
        rows = b"".join(b"\0" + pattern[3 * y : 3 * (y + width)] for y in range(height))
        rows = b"\0" + mark + rows[1 + len(mark) :]

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

        header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png = (
            b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b"")
        )
        return "data:image/png;base64," + base64.b64encode(png).decode()
    rate, seconds = 16000, MEDIA["audio"]
    hz = 180 + seed % 400
    samples = [
        int(9000 * math.sin(2 * math.pi * hz * t / rate) * (0.6 + 0.4 * math.sin(t / 2600)))
        for t in range(rate * seconds)
    ]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(mark + struct.pack(f"<{len(samples) - len(mark) // 2}h", *samples[len(mark) // 2 :]))
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def most_clients(args) -> int:
    """The most clients that send at once to this server: --clients, or with --plan its busiest workload."""
    return max(
        [args.clients, *(int(w.get("clients", 1)) for w in json.loads(args.plan))] if args.plan else [args.clients]
    )


def server_log(args) -> Path:
    """Keep a native server's diagnostics beside its result, so later attempts cannot overwrite them."""
    if args.log:
        return Path(args.log)
    out = args.out or (json.loads(args.plan)[0]["out"] if args.plan else None)  # a plan's first workload names it
    if out:
        return Path(out).with_suffix(".server.log")
    return RESULTS / f"{args.engine}.server.log"


def check(answers: dict, qs: list[dict]) -> list[str]:
    """Problems with one response: a missing question, a missing outcome, or a distribution not summing to 1."""
    problems = []
    for q in qs:
        got = answers.get(q["id"])
        if got is None:
            problems.append(f"{q['id']}: missing")
            continue
        if set(got) != set(g.outcome_names(q)):
            problems.append(f"{q['id']}: outcomes {sorted(got)}")
        values = list(got.values())
        if any(not (0 <= v <= 1) or not math.isfinite(v) for v in values) or abs(math.fsum(values) - 1) > 1e-3:
            problems.append(f"{q['id']}: not a distribution {got}")
    return problems


def prompt_tokens_needed(size: int, nq: int) -> int:
    """A conservative bound on a request's prompt: the context (estimated tokens +10%) plus 400 tokens of
    template and question."""
    return int(size * 1.1) + 400


def median_interval(values: list[float]) -> tuple[float, float] | None:
    """The 95% interval of the median from the order statistics alone (no assumption about the distribution):
    [x_(k), x_(n+1-k)] for the largest k whose binomial tail is at most 2.5%; None below 6 values, where even the
    range covers less than 95%."""
    n = len(values)
    tail, k = 0.0, 0
    while k < n:
        tail += math.comb(n, k) / 2**n
        if tail > 0.025:
            break
        k += 1
    if k == 0:
        return None
    ordered = sorted(values)
    return ordered[k - 1], ordered[n - k]


def settled(values: list[float], precision: float) -> bool:
    """Whether the median of `values` is known to within `precision` of itself (its 95% interval's half-width)."""
    interval = median_interval(values)
    return interval is not None and (interval[1] - interval[0]) / 2 <= precision * statistics.median(values)


def summary(timed: list[dict], sizes: list[int], counts: list[int]) -> dict:
    """Per shape that has timed requests ("<size>x<questions>"): their median and its 95% interval, count, range
    and the engine's tokens."""
    out = {}
    for s in sizes:
        for c in counts:
            rows = [r for r in timed if r["size"] == s and r["nq"] == c]
            if rows:
                ms = [r["ms"] for r in rows]
                interval = median_interval(ms)
                out[f"{s}x{c}"] = {
                    "median_ms": round(statistics.median(ms), 2),
                    "median_ci_ms": [round(v, 2) for v in interval] if interval else None,
                    "n": len(ms),
                    "min_ms": min(ms),
                    "max_ms": max(ms),
                    "tokens": rows[0]["tokens"],
                }
    return out


def invoke(decide, req: dict, http: bool):
    """Retry HTTP socket failures at most twice, inside the caller's timer, recording each cause."""
    retried = []
    while True:
        try:
            return (*decide(req), retried)
        except (ConnectionError, urllib.error.URLError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if (
                not http
                or isinstance(exc, urllib.error.HTTPError)
                or not isinstance(reason, OSError)
                or isinstance(reason, (TimeoutError, ConnectionRefusedError))
                or len(retried) == HTTP_TRANSPORT_RETRIES
            ):
                raise
            retried.append(f"{type(reason).__name__}: {reason}")
            time.sleep(HTTP_RETRY_BACKOFF)


def run(args, decide, meta: dict, *, memory=None, server_pid: int | None = None, fits=None) -> dict:
    """Warm up, time every request, write the result file. `memory()` gives the engine's own memory
    numbers (bytes), sampled after each request; `server_pid` is a server process to measure as well.
    `fits(state, nq) -> (ok, reason)` says whether a size's largest request fits the model, from the
    engine's own tokenizer; without it, the estimate of prompt_tokens_needed against --max-input.
    With --plan, each workload in turn on the runtime as loaded, one result file each; the last result is returned."""
    if args.plan:
        results = [
            run(argparse.Namespace(**{**vars(args), **workload, "plan": None}), decide, meta, memory=memory,
                server_pid=server_pid, fits=fits)
            for workload in json.loads(args.plan)
        ]  # fmt: skip
        return results[-1]
    if args.clients > 1 and server_pid is None:
        sys.exit("--clients times a runtime's own server; this one has none, and a wrapper's queue is not its own")
    if args.clients > 1 and args.scenario != "fresh":
        sys.exit("--clients times fresh requests only")
    if args.http and server_pid is None and meta.get("transport") != "benchmark HTTP wrapper":
        return run_http(args, decide, meta, memory=memory, fits=fits, run=run)
    if server_pid is not None:
        meta = {**meta, "transport": "native HTTP"}
    campaign_path = os.environ.get("IPPON_BENCH_CAMPAIGN")
    campaign = json.loads(Path(campaign_path).read_text()) if campaign_path else None
    sizes = [int(s) for s in args.sizes.split(",")]
    counts = [int(c) for c in args.questions.split(",")]
    if args.scenario != "lines":
        for count in counts:
            g.questions(count)  # a request must contain the number of questions its result advertises
    reasons = {}
    for s in sizes:
        if fits is not None:
            ok, why = fits(g.state(s, args.seed, "fit-check"), max(counts))
        else:
            need = prompt_tokens_needed(s, max(counts))
            ok, why = need <= args.max_input, f"prompt of about {need} tokens exceeds max input {args.max_input}"
        if not ok:
            reasons[str(s)] = why
    skipped = [s for s in sizes if str(s) in reasons]
    kept = [s for s in sizes if s not in skipped]

    def footprints() -> dict:
        return {"self": footprint(), "server": footprint(server_pid) if server_pid else None}

    started = machine_state()
    result = {
        "engine": args.engine,
        **({"campaign": campaign} if campaign else {}),
        "meta": meta,
        "device": device_info(),
        "start": started,
        "config": {
            **(
                {
                    "http_transport_retries": HTTP_TRANSPORT_RETRIES,
                    "http_retry_backoff_s": HTTP_RETRY_BACKOFF,
                }
                if args.http
                else {}
            ),
            "n": args.n,
            "warmup": args.warmup,
            "sizes": sizes,
            "questions": counts,
            "seed": args.seed,
            "max_input": args.max_input,
            "gap_s": args.gap,
            "chars_per_token": g.CHARS_PER_TOKEN,
            "clients": args.clients,
            "media": args.media and {"kind": args.media, "size": MEDIA[args.media]},
            **({"same_media": args.same_media} if args.same_media else {}),
            "scenario": args.scenario,
            **(
                {"min_n": args.min_n, "precision": args.precision, "cell_seconds": args.cell_seconds}
                if args.min_n
                else {}
            ),
        },
        "skipped_sizes": reasons,
        "after_load": {**footprints(), "engine": memory() if memory else None},
        "requests": [],
        "set_aside": [],  # timed requests that overlapped another busy process, each replaced by its shape's next
        "errors": [],
    }
    out = Path(args.out) if args.out else RESULTS / f"{args.engine}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    peak_engine, shape, power = 0, None, None
    pending = g.plan(kept, counts, args.n, args.warmup, args.seed, args.scenario)
    extra = {(s, c): 0 for s in kept for c in counts}  # replacements timed per shape, at most n
    # --min-n: a shape's planned timed requests stop once its median is settled (or its time is up); the requests
    # sent are always the plan's first ones, so both arms of a pair answer the same requests as far as both went.
    stopped: dict[tuple[int, int], str] = {}
    began: dict[tuple[int, int], float] = {}

    def timed_ms(key: tuple[int, int]) -> list[float]:
        return [r["ms"] for r in result["requests"] if r["phase"] == "timed" and (r["size"], r["nq"]) == key]

    def stop_reason(key: tuple[int, int]) -> str | None:
        ms = timed_ms(key)
        if len(ms) < args.min_n:
            return None
        if settled(ms, args.precision):
            return "settled"
        if args.cell_seconds and time.monotonic() - began[key] >= args.cell_seconds:
            return "time"
        return None

    history: dict[tuple[int, int], list[str]] = {}  # per shape, the media of its requests so far, in sending order
    same: dict[tuple[int, int], str] = {}  # with --same-media, each shape's one media file

    def prepare(req: dict) -> dict:
        systemone = g.to_systemone(req["state"], g.asked(req))
        if args.media and args.same_media:
            shape = (req["size"], req["nq"])
            if shape not in same:
                same[shape] = media_file(args.media, f"{args.seed}/same/{req['size']}/{req['nq']}")
            systemone["files"] = [same[shape]]
        elif args.media:
            key = f"{args.seed}/{req['phase']}/{req['size']}/{req['nq']}/{req['rep']}"
            sent = history.setdefault((req["size"], req["nq"]), [])
            if len(sent) > args.media_history:  # the run is full: the next request starts a new one
                sent.clear()
            sent.append(media_file(args.media, key))
            systemone["files"] = list(sent)
        return {**req, "systemone": systemone}

    def row_of(req: dict, ms: float, answers: dict, tokens, extra_info, power, retries: list[str]) -> dict:
        return {
            **{k: req[k] for k in ("phase", "size", "nq", "rep")},
            "ms": round(ms, 2),
            "tokens": tokens,
            "state_chars": len(req["state"]),
            "request_sha256": hashlib.sha256(
                json.dumps(req["systemone"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "power": power,
            "answers": answers,
            **({"transport_retries": len(retries), "transport_errors": retries} if retries else {}),
            **({"first": req["first"]} if req.get("first") else {}),
            "problems": check(answers, g.asked(req)),
            **({"extra": extra_info} if extra_info else {}),
        }

    with Monitor(server_pid) as activity:
        if args.clients > 1:
            pending = []
            concurrent(args, decide, result, out, kept, counts, activity, prepare, row_of)
        while pending:
            sent = []  # timed rows with their monotonic windows, judged once the batch has run back to back
            for req in pending:
                key = (req["size"], req["nq"])
                if req["phase"] == "timed" and key in stopped:
                    continue  # settled: the plan's later requests of this shape are not sent
                if (req["phase"], req["size"], req["nq"]) != shape:
                    shape, power = (req["phase"], req["size"], req["nq"]), power_now()
                    result["complete"] = False
                    out.write_text(dumps(result, indent=1))  # what has run so far survives a crash
                req = prepare(req)
                which = {k: req[k] for k in ("phase", "size", "nq", "rep")}  # the request, in its row or its error
                if req["phase"] == "timed" and args.gap:
                    time.sleep(args.gap)
                if req["phase"] == "timed":
                    began.setdefault(key, time.monotonic())
                started, t0 = time.monotonic(), time.perf_counter()
                try:
                    answers, tokens, extra_info, retries = invoke(decide, req, args.http)
                except Exception as exc:
                    result["errors"].append({**which, "error": f"{type(exc).__name__}: {exc}"[:2000]})
                    print(
                        f"[{args.engine}] ERROR {req['phase']} size={req['size']} nq={req['nq']}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                ms = (time.perf_counter() - t0) * 1000
                ended = time.monotonic()
                mem = memory() if memory else None
                if mem:
                    peak_engine = max(peak_engine, max(v for v in mem.values() if isinstance(v, (int, float))))
                row = row_of(req, ms, answers, tokens, extra_info, power, retries)
                problems = row["problems"]
                result["requests"].append(row)
                if req["phase"] == "timed":
                    sent.append((row, started, ended))
                    if args.min_n and (why := stop_reason(key)):
                        stopped[key] = why
                print(
                    f"[{args.engine}] {req['phase']:6s} size={req['size']:5d} nq={req['nq']} rep={req['rep']} "
                    f"{ms:9.1f} ms tokens={tokens} {'PROBLEMS ' + str(problems) if problems else 'ok'}",
                    flush=True,
                )
            pending = []
            for row, started, ended in sent:
                busy = activity.during(started, ended)
                if not busy:
                    continue
                key = (row["size"], row["nq"])
                if extra[key] == args.n:
                    row["busy"] = busy  # out of replacements: the reading stays and the whole run is rejected
                    continue
                # Requests are generated in a fixed order, so the replacement is the same request for every engine
                # (a re-sent state's next step): the shape's next one after every timed request it has sent.
                extra[key] += 1
                result["requests"].remove(row)
                result["set_aside"].append({**row, "busy": busy})
                taken = sum(  # timed requests of the shape sent, and replacements already queued
                    r["phase"] == "timed" and (r["size"], r["nq"]) == key
                    for r in result["requests"] + result["set_aside"] + pending
                )
                stopped.pop(key, None)  # unsettled again until its replacement has been timed
                pending += g.plan([row["size"]], [row["nq"]], taken + 1, args.warmup, args.seed, args.scenario)[-1:]
                print(f"[{args.engine}] set aside timed size={row['size']} nq={row['nq']} rep={row['rep']}", flush=True)
                print("busy while timing: " + ", ".join(busy), flush=True)
    result["busy"] = sorted(
        {name for row in result["requests"] for name in row.get("busy", [])} | set(result.get("busy", []))
    )
    result["activity_errors"] = activity.errors
    if result["busy"]:
        print("busy while timing: " + ", ".join(result["busy"]), flush=True)
    result["end"] = machine_state()
    result["memory"] = {
        **footprints(),
        "engine_peak_sampled": peak_engine or None,
        "engine_now": memory() if memory else None,
    }
    timed = [r for r in result["requests"] if r["phase"] == "timed"]
    result["summary"] = summary(timed, kept, counts)
    if args.min_n:
        result["stopped"] = {f"{s}x{c}": stopped.get((s, c), "n") for s, c in began}
    result["complete"] = True
    out.write_text(dumps(result, indent=1))
    print(
        f"[{args.engine}] wrote {out}: {len(timed)} timed, {len(result['errors'])} errors, "
        f"skipped sizes {skipped or 'none'}",
        flush=True,
    )
    return result


def concurrent(args, decide, result, out, sizes, counts, activity, prepare, row_of) -> None:
    """--clients N: N clients send at once, each taking the next request as soon as its last is answered, through
    the warm-ups and then the timed requests (N per shape each, times --warmup and --n, or --min-n when given: a
    window's rate needs no settling). The timed requests form one window: its requests per second is its count over
    its span, and its busy readings stay on the run."""
    per_client = args.min_n or args.n
    plan = g.plan(sizes, counts, per_client * args.clients, args.warmup * args.clients, args.seed)
    power = power_now()
    for phase in ("warmup", "timed"):
        work: queue.SimpleQueue = queue.SimpleQueue()
        for req in plan:
            if req["phase"] == phase:
                work.put(prepare(req))
        rows, lock = [], threading.Lock()

        def client(index: int, work: queue.SimpleQueue, rows: list, lock: threading.Lock) -> None:
            while True:
                try:
                    req = work.get_nowait()
                except queue.Empty:
                    return
                started, t0 = time.monotonic(), time.perf_counter()
                try:
                    answers, tokens, extra_info, retries = invoke(decide, req, args.http)
                except Exception as exc:
                    with lock:
                        result["errors"].append(
                            {
                                **{k: req[k] for k in ("phase", "size", "nq", "rep")},
                                "error": f"{type(exc).__name__}: {exc}"[:2000],
                            }
                        )
                    continue
                ms = (time.perf_counter() - t0) * 1000
                row = {**row_of(req, ms, answers, tokens, extra_info, power, retries), "client": index}
                row["window_s"] = [round(started, 4), round(time.monotonic(), 4)]
                with lock:
                    rows.append(row)

        threads = [
            threading.Thread(target=client, args=(i, work, rows, lock), name=f"client-{i}") for i in range(args.clients)
        ]
        begin = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        end = time.monotonic()
        rows.sort(key=lambda r: r["window_s"][0])
        for row in rows:
            row["window_s"] = [round(row["window_s"][0] - begin, 4), round(row["window_s"][1] - begin, 4)]
        result["requests"] += rows
        print(
            f"[{args.engine}] {phase} {len(rows)} requests from {args.clients} clients in {end - begin:.2f} s",
            flush=True,
        )
        if phase == "timed":
            busy = activity.during(begin, end)
            result["busy"] = busy
            result["concurrent"] = {
                "clients": args.clients,
                "span_s": round(end - begin, 4),
                "requests_per_second": len(rows) / (end - begin),
            }
            if busy:
                print("busy while timing: " + ", ".join(busy), flush=True)
        out.write_text(dumps({**result, "complete": False}, indent=1))


def systemone_answers(response: dict) -> dict:
    """A System One response's answers as {question: {outcome: probability}} (noul: yes = P(true))."""
    out = {}
    for name, a in response["answers"].items():
        if a["type"] == "noul" or "noul" in a:
            p = float(a["noul"])
            out[name] = {"yes": p, "no": 1 - p}
        else:
            out[name] = {str(k): float(v) for k, v in a["probabilities"].items()}
    return out


def repo_revision(model_dir: str) -> str | None:
    """The Hub commit a `hf download --local-dir` checkout came from (its .cache metadata), if any."""
    meta = Path(model_dir).expanduser() / ".cache" / "huggingface" / "download"
    for f in sorted(meta.glob("*.metadata")):
        lines = f.read_text().splitlines()
        if lines:
            return lines[0].strip()
    return None
