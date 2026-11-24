"""The advertised 50-question HTTP workload must send and validate all 50 answers, timing only quiet requests; a
re-sent workload sends one state again and again, its busy requests replaced by the state's next step."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / "benchmarks/kit"
# what every program below starts with: the kit importable, no readings of this Mac, and a Monitor that sees no
# other process (a test that needs one busy sets its own after this)
PRELUDE = """
import sys
sys.path.insert(0, sys.argv[1])
import common
common.machine_state = lambda: {}
common.device_info = lambda: {}
common.power_now = lambda: "test"
common.footprint = lambda *a: None
class Quiet:
    busy, errors = [], []
    def __init__(self, *a): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def during(self, *a): return []
common.Monitor = Quiet
"""


def run_kit(program: str, *args: object, timeout: float = 30) -> None:
    """Run `program` after PRELUDE in a fresh isolated Python, with the kit's folder and `args` as its arguments."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", PRELUDE + program, str(KIT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_http_workload_counts_answers_and_replaces_busy_timed_requests(tmp_path: Path) -> None:
    program = """
import argparse, hashlib, json, sys, threading
from pathlib import Path
from activity import competing
from gen_requests import plan, questions, outcome_names, to_systemone
from summarize import _readme_accepted_rounds, _readme_run
assert competing("10 1 80 model\\n11 10 90 worker\\n12 1 51 unrelated", {10}) == ["unrelated (51%)"]
class BusyOnce:  # another process overlaps the first timed request only
    busy, errors, judged = set(), [], 0
    def __init__(self, *a): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def during(self, start, end):
        BusyOnce.judged += 1
        return ["indexer (90%)"] if BusyOnce.judged == 1 else []
common.Monitor = BusyOnce
parent = Path(sys.argv[2]).parent
campaign = {"rounds": 2}
(parent / "CAMPAIGN.json").write_text(json.dumps(campaign))
ledger = [{"kind": "round", "round": n, "accepted": True,
           "engines": [{"engine": "fixture", "exit": 0, "problems": []}]} for n in (1, 2)]
(parent / "ATTEMPTS.json").write_text(json.dumps(ledger))
folders = [parent / "r1", parent / "r2"]
_readme_accepted_rounds(folders, campaign, {"fixture"})
for selected, code in [(folders[:1], 0), ([parent / "attempt-r1-1", folders[1]], 0), (folders, 1)]:
    ledger[0]["engines"][0]["exit"] = code
    (parent / "ATTEMPTS.json").write_text(json.dumps(ledger))
    try:
        _readme_accepted_rounds(selected, campaign, {"fixture"})
    except ValueError:
        pass
    else:
        raise AssertionError("must refuse partial, rejected or failed rounds")
args = common.add_args(argparse.ArgumentParser()).parse_args([
    "--engine", "fixture", "--http", "--sizes", "1000", "--questions", "1,50",
    "--n", "1", "--warmup", "1", "--max-input", "32768", "--out", sys.argv[2]])
args.log = None
assert common.server_log(args) == Path(sys.argv[2]).with_suffix(".server.log")
loading_thread = threading.get_ident()
seen = []
def decide(req):
    assert threading.get_ident() == loading_thread
    qs = req["systemone"]["questions"]
    seen.append(len(qs))
    answers = {name: {k: 1 / len(outcome_names(q)) for k in outcome_names(q)} for name, q in qs.items()}
    return answers, 1000, None
result = common.run(args, decide, {})
assert seen == [1, 50, 1, 50, 1]
assert [(r["nq"], r["rep"], r["busy"]) for r in result["set_aside"]] == [(1, 0, ["indexer (90%)"])]
timed = [r for r in result["requests"] if r["phase"] == "timed"]
assert [(r["nq"], r["rep"]) for r in timed] == [(50, 0), (1, 1)] and not result["busy"]
replacement = to_systemone(plan([1000], [1], 2, 0)[-1]["state"], questions(1))
payload = json.dumps(replacement, sort_keys=True, separators=(",", ":")).encode()
assert timed[1]["request_sha256"] == hashlib.sha256(payload).hexdigest()
_readme_run(result, "fixture")
assert len({q["instructions"] for q in questions(50)}) == 50
assert common.check(result["requests"][1]["answers"], questions(50)) == []
del result["requests"][1]["answers"]["refund"]
assert common.check(result["requests"][1]["answers"], questions(50)) == ["refund: missing"]
for scenario in ("repeat", "grow"):  # one state per shape in a row: a new question, or a message more, each time
    sent = plan([1000], [1], 2, 1, 0, scenario)
    assert [r["first"] for r in sent] == ([0, 1, 2] if scenario == "repeat" else [0, 0, 0])
    assert len({r["state"] for r in sent}) == (1 if scenario == "repeat" else 3)
    assert all(b["state"].startswith(a["state"]) for a, b in zip(sent, sent[1:]))
BusyOnce.judged = 0
args.scenario, args.questions, args.out = "grow", "1", sys.argv[2] + ".grow"
grown = common.run(args, decide, {})
timed = [r for r in grown["requests"] if r["phase"] == "timed"]
assert [r["rep"] for r in grown["set_aside"]] == [0] and [r["rep"] for r in timed] == [1]
replacement = to_systemone(plan([1000], [1], 2, 1, 0, "grow")[-1]["state"], questions(1))
payload = json.dumps(replacement, sort_keys=True, separators=(",", ":")).encode()
assert timed[0]["request_sha256"] == hashlib.sha256(payload).hexdigest()
try:
    questions(51)
except ValueError:
    pass
else:
    raise AssertionError("must refuse a misleading question count")
"""
    output = tmp_path / "result.json"
    run_kit(program, str(output), timeout=20)
    result = json.loads(output.read_text())
    assert result["complete"]
    assert not result["errors"]
    assert result["meta"]["transport"] == "benchmark HTTP wrapper"
    assert [len(row["answers"]) for row in result["requests"]] == [1, 50, 50, 1]


def test_readme_charts_show_every_pair_once_and_never_call_a_loss_faster(tmp_path: Path) -> None:
    program = """
import sys
import xml.etree.ElementTree as ET
import chart, summarize as s
machine = {"chip": "Apple M5 Pro", "memory_bytes": 48 * 2**30, "macos": "27.0.1"}
pairs, ippon = [], {}
for i, (group, model, other, ours) in enumerate(s.readme_pairs()):
    ms = ippon.setdefault(ours, 10.0 + i)
    pct = -12.0 if i == 0 else 40.0  # one comparison ippon loses
    pairs.append({"group": group, "model": model, "runtime": "runtime 1.0", "other_engine": other,
                  "ippon_engine": ours, "rounds": 2, "campaign": "c", "date": "2026-10-27",
                  "machine": machine, "mlx": "0.32.4", "library_commit": "0" * 40,
                  "cells": [{"ippon_ms": n, "other_ms": n * (1 + pct / 100), "speed_percent": pct, "samples": [20, 20]}
                            for n in (ms, ms * 50)]})
report = {"library_commit": "0" * 40, "date": "2026-10-27", "mlx": "0.32.4", "pairs": pairs,
          "machine": machine}
models = s.chart_models(report)
drawn = [(m["name"], r["order"]) for m in models for p in m["panels"] if p["key"] == "one"
         for c in p["clusters"] for r in c["rivals"]]
assert len(drawn) == len(set(drawn)) == len([p for p in pairs if p["group"] not in s.CHARTS_LEAVE_OUT])
# The README charts the practical workloads only, and every model its own: d1-Omni-2B beside d1-2B, not folded in.
report["more"] = {"records": [{**p, "workload": "typical", "ippon": 10.0 + i, "other": 12.0 + i, "speed_percent": 20.0}
                              for i, p in enumerate(pairs)]}
readme = s.readme_chart_models(report)
assert readme and all(p["key"] in s.README_PANELS for m in readme for p in m["panels"])
assert {"d1-2B", "d1-Omni-2B"} <= {m["name"] for m in readme}
for key, _, charted in s.readme_section_models(report):  # each model once, in the section of the inputs it reads
    block = s.readme_charts(report, key)
    for m in charted:
        assert s.section(m) == key and f"{s.README_CHARTS}-{m['slug']}-light.svg" in block
        for theme in chart.THEMES.values():
            ET.fromstring(chart.model_svg(m, theme))
first = chart.model_svg(models[[m["name"] for m in models].index(s.family(pairs[0]["model"]))],
                        chart.THEMES["light"])
words = " ".join(ET.fromstring(first).itertext()).split()
words = " ".join(words)
assert "takes 12% less time" in words and "takes 12% longer" not in words  # a loss, said as the rival's time
"""
    run_kit(program)


def test_concurrent_clients_overlap_and_every_request_carries_new_media(tmp_path: Path) -> None:
    program = """
import argparse, sys, threading, time
import readme_round
from gen_requests import outcome_names, questions
lock, flight, files = threading.Lock(), [0, 0], []
def decide(req):
    with lock:
        flight[0] += 1
        flight[1] = max(flight)
        files.extend(req["systemone"]["files"])
    time.sleep(0.02)
    with lock:
        flight[0] -= 1
    answers = {q["id"]: {k: 1 / len(outcome_names(q)) for k in outcome_names(q)} for q in questions(1)}
    return answers, 1, None
options = ["--sizes", "100", "--questions", "1", "--clients", "4", "--media", "audio"]
args = common.add_args(argparse.ArgumentParser()).parse_args(
    ["--engine", "fixture", "--n", "3", "--warmup", "1", "--max-input", "32768", "--out", sys.argv[2], *options])
args.log = None
result = common.run(args, decide, {}, server_pid=1)
assert flight[1] == 4, flight
assert len(files) == len(set(files)) == 16 and all(f.startswith("data:audio/wav;base64,") for f in files)
assert readme_round.checked(result, 3, 1, ((100, 1),), 4) == []
assert result["concurrent"]["clients"] == 4 and result["concurrent"]["requests_per_second"] > 0
try:
    common.run(args, decide, {})
except SystemExit:
    pass
else:
    raise AssertionError("several clients must go to a runtime's own server, never to the kit's wrapper")
"""
    run_kit(program, str(tmp_path / "result.json"), timeout=60)


def test_one_load_runs_every_workload_and_settled_shapes_stop_early(tmp_path: Path) -> None:
    """A plan times several workloads on one loaded runtime; a shape whose median is settled stops at --min-n while a
    noisy one runs to --n, busy replacements continue the same request sequence, a runtime's own failure is kept as
    its result while busy or stopped attempts run again (also on resuming), and a second round takes only the pairs
    the first left close, which the summary accepts as the campaign's rounds."""
    program = """
import argparse, json, sys
from pathlib import Path
import readme_round
from gen_requests import outcome_names
from summarize import _readme_accepted_rounds, scheduled_steps
assert common.median_interval([1, 2, 3, 4, 5]) is None  # below 6 values even the range covers less than 95%
assert common.median_interval([6, 1, 5, 2, 4, 3]) == (1, 6)
assert common.median_interval(list(range(20))) == (5, 14)  # x_(6) and x_(15) of 20
class Clock:  # the request times the test chooses: decide moves the clock
    now = 0.0
    def perf_counter(self): return Clock.now
    def monotonic(self): return Clock.now
    def sleep(self, s): pass
common.time = Clock()
class BusyFirstTimed:
    errors, calls = [], 0
    def __init__(self, *a): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def during(self, start, end):
        BusyFirstTimed.calls += 1
        return ["indexer (90%)"] if BusyFirstTimed.calls == 1 else []
common.Monitor = BusyFirstTimed
def decide(req):  # 100 ms, but the 3-question shape alternates 100 and 200 ms, which never settles
    Clock.now += 0.1 if req["phase"] == "warmup" or req["nq"] == 1 else (0.1 if req["rep"] % 2 else 0.2)
    qs = req["systemone"]["questions"]
    return {k: {o: 1 / len(outcome_names(q)) for o in outcome_names(q)} for k, q in qs.items()}, 1000, None
out = Path(sys.argv[2])
plan = [{"out": str(out / "fixture.json"), "sizes": "1000", "questions": "1,3", "clients": 1, "media": None,
         "scenario": "fresh"},
        {"out": str(out / "fixture.repeat.json"), "sizes": "1000", "questions": "1", "clients": 1, "media": None,
         "scenario": "repeat"}]
args = common.add_args(argparse.ArgumentParser()).parse_args([
    "--engine", "fixture", "--http", "--plan", json.dumps(plan), "--n", "9", "--warmup", "1", "--max-input",
    "32768", "--min-n", "6", "--precision", "0.03"])
common.run(args, decide, {"transport": "benchmark HTTP wrapper"}, server_pid=None)
text, repeat = (json.loads((out / name).read_text()) for name in ("fixture.json", "fixture.repeat.json"))
one = [r for r in text["requests"] if r["phase"] == "timed" and r["nq"] == 1]
three = [r for r in text["requests"] if r["phase"] == "timed" and r["nq"] == 3]
# 1 question: steady, settled at 6 after its busy first request was replaced by the next of the same sequence
assert [r["rep"] for r in text["set_aside"]] == [0] and [r["rep"] for r in one] == [1, 2, 3, 4, 5, 6], one
assert text["stopped"] == {"1000x1": "settled", "1000x3": "n"}
assert [r["rep"] for r in three] == list(range(9))  # alternating 100 and 200 ms never settles: it runs to --n
assert text["summary"]["1000x1"]["median_ci_ms"] == [100.0, 100.0]
assert repeat["config"]["scenario"] == "repeat" and repeat["stopped"] == {"1000x1": "settled"}
shapes = ((1000, 1), (1000, 3))
assert readme_round.checked(text, 9, 1, shapes, 1, 6) == []
assert readme_round.checked(text, 9, 1, shapes, 1, 0) != []  # without --min-n every shape needs all 9
# round 2: only the pair round 1 left within 10%
r1 = out / "r1"
r1.mkdir()
def result(name, ms, rate=None):
    run = {"complete": True, "errors": [], "summary": {"1000x1": {"median_ms": ms, "median_ci_ms": [ms, ms]}}}
    if rate:
        run["concurrent"] = {"requests_per_second": rate}
    (r1 / f"{name}.json").write_text(json.dumps(run))
for name, ms, rate in [("far", 200, None), ("far-ippon", 100, None), ("near", 104, None), ("near-ippon", 100, None),
                       ("far.c8", 0, 4.0), ("far-ippon.c8", 0, 6.0)]:
    result(name, ms or 100, rate)
plans = [("text", "", None, None, None), ("concurrent", ".c8", None, None, None)]
pairs = {"text": [["far", "far-ippon"], ["near", "near-ippon"]], "concurrent": [["far", "far-ippon"]]}
assert readme_round.close_engines(r1, plans, pairs, 0.10) == {"text": {"near", "near-ippon"}, "concurrent": set()}
# an attempt's outcome: the runtime's own failure stays as its result; busy readings and a guard stop run again
oom = {"complete": True, "errors": [{"error": "MPS backend out of memory"}]}
assert readme_round.verdict(oom, 0, ["incomplete or request errors"], False) == "failed"
assert readme_round.verdict({}, 1, [], False) == "failed"  # its process ended during the workload
assert readme_round.verdict(text, 0, [], False) == "accepted"
assert readme_round.verdict({"complete": False}, 143, ["timed 1000x1: missing samples"], False) == "retry"
assert readme_round.verdict({"complete": True}, 0, ["activity monitor failure"], False) == "retry"
assert readme_round.verdict(oom, 0, ["incomplete or request errors"], True) == "failed"
assert readme_round.verdict({}, 1, [], True) == "failed"
assert readme_round.verdict(text, 0, [], True) == "accepted"
# a resumed round keeps what it recorded as failed or accepts on disk, and runs an interrupted result again
attempt = out / "attempt-r1"
attempt.mkdir()
(attempt / "kept.json").write_text('{"ok": true}')
(attempt / "cut.json").write_text("{}")
steps = [(name, None, None, 1) for name in ("oom", "kept", "cut", "todo")]
failed = [{"round": 1, "engine": "oom", "exit": 0, "problems": [], "failed": "out of memory"}]
entries, done = readme_round.resumed(steps, attempt, out / "rejected-r1", failed, 1, lambda r, *_: r.get("ok", False))
assert done == {"oom", "kept"} and [e["engine"] for e in entries] == ["oom", "kept"] and entries[0]["failed"]
assert (out / "rejected-r1/cut-interrupted.json").exists() and not (attempt / "cut.json").exists()
campaign = {"rounds": 2, "close": 0.10}
(out / "CAMPAIGN.json").write_text(json.dumps(campaign))
everyone = [{"engine": e, "exit": 0, "problems": []} for e in ("far", "far-ippon", "near", "near-ippon")]
ledger = [{"kind": "round", "round": 1, "accepted": True, "engines": everyone},
          {"kind": "round", "round": 2, "accepted": True, "scheduled": ["near", "near-ippon"], "engines": everyone[2:]}]
(out / "ATTEMPTS.json").write_text(json.dumps(ledger))
(out / "r2").mkdir()
engines = {"far", "far-ippon", "near", "near-ippon"}
_readme_accepted_rounds([r1, out / "r2"], campaign, engines)
assert scheduled_steps([r1, out / "r2"]) == {r1: None, out / "r2": {"near", "near-ippon"}}
(out / "ATTEMPTS.json").write_text(json.dumps(ledger[:1]))  # no pair was close: one round is the campaign
_readme_accepted_rounds([r1], campaign, engines)
"""
    run_kit(program, str(tmp_path), timeout=20)


def test_readme_campaign_selection_keeps_pairs_and_solo_evidence_together(tmp_path: Path) -> None:
    program = r"""
import hashlib, json, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import chart, summarize as s
out = Path(sys.argv[2])

def save(path, value):
    path.write_text(json.dumps(value))

def campaign(name, date, pairs, solo, base):
    parent = out / name
    folder = parent / "r1"
    folder.mkdir(parents=True)
    engines = {e for p in pairs for e in p[2:]} | set(solo)
    identity = parent / "MODEL_BUILDS.json"
    save(identity, {"engines": dict.fromkeys(engines, {})})
    commit = ("a" if name == "old" else "b") * 40
    manifest = {"library_commit": commit, "date": date, "rounds": 1, "close": 0.1, "samples": 3, "warmup": 1,
                "generator_sha256": hashlib.sha256(Path(sys.argv[1], "gen_requests.py").read_bytes()).hexdigest(),
                "machine": {"kernel_rule": "test"},
                "model_builds": {"file": identity.name, "sha256": hashlib.sha256(identity.read_bytes()).hexdigest()},
                "workloads": [{"name": "text", "pairs": [p[2:] for p in pairs], "alone": solo}]}
    save(parent / "CAMPAIGN.json", manifest)
    save(parent / "ATTEMPTS.json", [{"kind": "round", "round": 1, "accepted": True,
                                    "engines": [{"engine": e, "exit": 0, "problems": []} for e in engines]}])
    config = {"n": 3, "min_n": 2, "warmup": 1, "sizes": [1000], "questions": [1, 50],
              "seed": 0, "chars_per_token": s.requests.CHARS_PER_TOKEN, "precision": 0.03, "cell_seconds": 90}
    for engine in engines:
        rows = []
        ours = engine.endswith("-ippon-serve")
        for req in s.requests.plan([1000], [1, 50], 2, 1):
            qs = s.requests.questions(req["nq"])
            payload = s.requests.to_systemone(req["state"], qs)
            rows.append({k: req[k] for k in ("phase", "size", "nq", "rep")})
            rows[-1].update({"ms": base * req["nq"] * (1 if ours else 2), "problems": [],
                            "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True,
                                                                         separators=(",", ":")).encode()).hexdigest(),
                            "answers": {q["id"]: dict.fromkeys(s.requests.outcome_names(q),
                                                        1 / len(s.requests.outcome_names(q))) for q in qs}})
        run = {"engine": engine, "campaign": manifest, "complete": True, "errors": [], "busy": [],
               "config": config, "start": {"time": date + "T10:00:00"}, "end": {"time": date + "T10:01:00"},
               "device": {"chip": "Apple M5 Pro", "memory_bytes": 48 * 2**30, "macos": "27.0.1"},
               "meta": {"transport": "native HTTP", "revision": commit[:7], "mlx": "0.32.4",
                        "display_runtime": "runtime 1.0"}, "requests": rows}
        save(folder / f"{engine}.json", run)
    return folder

pairs_by_engine = {p[2]: p for p in s.readme_pairs()}
pairs = [pairs_by_engine[name] for name in ("clef-flash-llama-Q8_0", "clef-flash-mlx-8bit-omlx", "clef-flash-mlx-8bit")]
solo = [engine + "-serve" for _, engine in s.ALONE]
old = campaign("old", "2026-10-28", pairs, [], 100)
new = campaign("new", "2026-10-29", pairs[:1], solo, 30)
# Only oMLX's close pair ran a second round. The shared ippon arm must not change mlx-vlm's median.
old2 = old.parent / "r2"
old2.mkdir()
manifest = json.loads((old.parent / "CAMPAIGN.json").read_text())
manifest["rounds"] = 2
save(old.parent / "CAMPAIGN.json", manifest)
for path in old.glob("*.json"):
    run = json.loads(path.read_text()); run["campaign"] = manifest; save(path, run)
    if path.stem in pairs[1][2:]:
        for row in run["requests"]:
            row["ms"] *= 2
        save(old2 / path.name, run)
ledger = json.loads((old.parent / "ATTEMPTS.json").read_text())
ledger.append({"kind": "round", "round": 2, "accepted": True, "scheduled": list(pairs[1][2:]),
               "engines": [{"engine": e, "exit": 0, "problems": []} for e in pairs[1][2:]]})
save(old.parent / "ATTEMPTS.json", ledger)
old_rounds = [old, old2]
# A campaign that ran only the practical workloads (as an 8 GiB Mac's second round did) adds no text rows.
practical = campaign("practical", "2026-10-30", pairs[:1], [], 10)
manifest = json.loads((practical.parent / "CAMPAIGN.json").read_text())
manifest["workloads"] = [{"name": "typical", "pairs": manifest["workloads"][0]["pairs"]}]
save(practical.parent / "CAMPAIGN.json", manifest)
# Input ordering must not select the archive over the new targeted result, or pool across them.
report = s.readme_comparisons([practical, new, *old_rounds])
assert len(report["pairs"]) == 3 and len(report["alone"]) == len(s.ALONE)
for record in report["pairs"]:
    expected = 30 if record["other_engine"] == pairs[0][2] else 150 if record["other_engine"] == pairs[1][2] else 100
    assert [c["ippon_ms"] for c in record["cells"]] == [expected, expected * 50]
    assert all(c["samples"] == ([4, 4] if expected == 150 else [2, 2]) for c in record["cells"])
    assert record["library_commit"] == ("b" if expected == 30 else "a") * 40
for record in report["alone"]:
    assert record["date"] == "2026-10-29"
    assert [c["ippon_ms"] for c in record["cells"]] == [30, 1500]
    assert all("speed_percent" not in c for c in record["cells"])
assert "30 ms on ippon" in s.readme_headline(report)
assert "takes 2x as long" in s.readme_headline(report)
block = s.readme_charts(report)
assert all(f"{s.README_CHARTS}-{s.slug(r['model'])}" not in block for r in report["alone"])  # a table, not charts
assert s.alone_title(report) in s.readme_tables(report)
# Embedding a generated report must keep every section below its page heading.
doc = out / "benchmarks.md"
doc.write_text("# Benchmarks\\n" + s.BLOCKS["fresh"][0] + "\\n" + s.BLOCKS["fresh"][1])
sys.argv = ["summarize.py", str(new), *(str(p) for p in old_rounds), "--readme-round", "--docs", str(doc)]
s.main()
headings = [line for line in doc.read_text().splitlines() if line.startswith("#")]
assert [line for line in headings if line.startswith("## ")] == ["## Current README round"]
assert "### One-question latency" in headings and "### 50-question request time" in headings
assert "### " + s.alone_title(report) in headings
for model in (m for m in s.chart_models(report) if m["alone"]):
    for theme in chart.THEMES.values():
        svg = chart.model_svg(model, theme)
        words = " ".join(" ".join(ET.fromstring(svg).itertext()).split())
        assert "30 ms" in words and "1,500 ms" in words
        assert all(word not in words for word in ("faster", "slower", "%", "checkpoint", "rival"))
# Rivals measured against one ippon run share its cluster in the 8-client panel.
records = [{**p, "workload": "concurrent", "ippon": 4.0, "other": 2.0, "samples": [48, 48], "speed_percent": 100,
            "server_env": None, "clients": {"ippon": {"requests_per_second": 4.0, "median_ms": 900, "p90_ms": 950},
                                            "other": {"requests_per_second": 2.0, "median_ms": 1800, "p90_ms": 1900}}}
           for p in report["pairs"][1:]]
more = {**report, "pairs": [], "alone": [], "more": {"records": records}}
left_out, s.CHARTS_LEAVE_OUT = s.CHARTS_LEAVE_OUT, ()  # clustering alone, before the charts leave mlx-vlm out
clusters = [c for m in s.chart_models(more) for p in m["panels"] if p["key"] == "concurrent" for c in p["clusters"]]
assert len(clusters) == 1 and len(clusters[0]["rivals"]) == 2
s.CHARTS_LEAVE_OUT = left_out
clusters = [c for m in s.chart_models(more) for p in m["panels"] if p["key"] == "concurrent" for c in p["clusters"]]
assert [r["group"] for c in clusters for r in c["rivals"]] == ["omlx"]
# Required campaign folders and selected engines cannot disappear silently.
try:
    s.readme_comparisons([*old_rounds, out / "missing/r1"])
except ValueError as exc:
    assert "folder does not exist" in str(exc)
else:
    raise AssertionError("missing campaign accepted")
path = new / f"{pairs[0][2]}.json"
saved = path.read_text()
path.unlink()
try:
    s.readme_comparisons([*old_rounds, new])
except ValueError as exc:
    assert "missing required engines" in str(exc)
else:
    raise AssertionError("missing scheduled arm accepted")
# Partial campaigns keep an older complete pair while retaining finished solo measurements.
partial = s.readme_comparisons([*old_rounds, new], partial=True)
assert next(p for p in partial["pairs"] if p["other_engine"] == pairs[0][2])["cells"][0]["ippon_ms"] == 100
path.write_text(saved)
# The old source hash may differ only if its recorded request payloads are still identical.
manifest = json.loads((old.parent / "CAMPAIGN.json").read_text())
manifest["generator_sha256"] = "0" * 64
save(old.parent / "CAMPAIGN.json", manifest)
for path in old.parent.glob("r*/*.json"):
    run = json.loads(path.read_text()); run["campaign"] = manifest; save(path, run)
s.readme_comparisons(old_rounds)
path = old / f"{pairs[0][3]}.json"
run = json.loads(path.read_text())
run["requests"][0]["request_sha256"] = "wrong"
save(path, run)
try:
    s.readme_comparisons(old_rounds)
except ValueError as exc:
    assert "changed the recorded payloads" in str(exc)
else:
    raise AssertionError("changed generator payload accepted")
"""
    run_kit(program, str(tmp_path))


def test_practical_report_keeps_three_question_and_standalone_results(tmp_path: Path) -> None:
    program = r"""
import hashlib, json, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import chart, readme_round as rr, summarize as s
out = Path(sys.argv[2])
folder = out / "r1"
folder.mkdir()
_, _, other, ours = next(p for p in s.readme_pairs() if p[2] == "d1-2B-bf16-ollama")
solo = "kettei-core-ippon-serve"
names = ("typical", "repeat", "grow", "concurrent", "burst")
workloads = [{"name": name, "options": rr.WORKLOADS[name][1], "pairs": [[other, ours]], "alone": [solo]}
             for name in names]
campaign = {"library_commit": "a" * 40, "rounds": 1, "workloads": workloads}
(out / "CAMPAIGN.json").write_text(json.dumps(campaign))
steps = [engine + rr.WORKLOADS[name][0] for name in names for engine in (other, ours, solo)]
ledger = [{"kind": "round", "round": 1, "accepted": True,
           "engines": [{"engine": e, "exit": 0, "problems": []} for e in steps]}]
(out / "ATTEMPTS.json").write_text(json.dumps(ledger))
for workload in workloads:
    name = workload["name"]
    suffix, options, ((size, nq),) = rr.WORKLOADS[name]
    clients = rr.workload_clients(options)
    for engine in (other, ours, solo):
        rows = []
        scenario = "repeat" if name == "repeat" else "grow" if name == "grow" else "fresh"
        for req in s.requests.plan([size], [nq], clients, 1, 0, scenario):
            qs = s.requests.questions(nq, req.get("first", 0))
            payload = s.requests.to_systemone(req["state"], qs)
            rows.append({k: req[k] for k in ("phase", "size", "nq", "rep")})
            rows[-1].update({"ms": 600 if engine == other else 300, "problems": [],
                            "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True,
                                                                         separators=(",", ":")).encode()).hexdigest(),
                            "answers": {q["id"]: dict.fromkeys(s.requests.outcome_names(q),
                                            1 / len(s.requests.outcome_names(q))) for q in qs}})
        run = {"engine": engine, "complete": True, "errors": [], "busy": [], "requests": rows,
               "config": {"sizes": [size], "questions": [nq], "clients": clients, "scenario": scenario},
               "start": {"time": "2026-10-30T01:00:00"},
               "device": {"chip": "Apple M5 Pro", "memory_bytes": 48 * 2**30, "macos": "27.0.1"},
               "meta": {"transport": "native HTTP", "revision": "a" * 7, "mlx": "0.32.4",
                        "display_runtime": "Ollama 0.40.13"},
               "concurrent": {"span_s": clients / (2 if engine == other else 4)}}
        (folder / f"{engine}{suffix}.json").write_text(json.dumps(run))
report = s.readme_more([folder])
assert len(report["records"]) == 10 and len(report["sources"]) == len(steps)
for record in report["records"]:
    assert record["ippon"] == (4 if record["workload"] in s.CLIENTS else 300)  # requests a second, else ms
    assert record["speed_percent"] == (None if record.get("alone") else 100)
    if record.get("alone"):
        assert record["other"] is None and record["model"] == "Kettei Core"
    for source in record["sources"]:
        assert report["sources"][source] == hashlib.sha256((folder / Path(source).name).read_bytes()).hexdigest()
text = s.readme_more_tables(report)
assert "about 500 tokens, 3 questions" in text and text.count("ippon alone") == len(names)
typical = [p for m in s.chart_models({"more": report}) for p in m["panels"] if p["key"] == "typical"]
assert sorted(len(c["rivals"]) for p in typical for c in p["clusters"]) == [0, 1]
assert "| Kettei Core | 300 ms |" in s.readme_charts({"more": report})
summary, evidence = out / "SUMMARY.md", out / "EVIDENCE.json"
sys.argv = ["summarize.py", str(folder), "--more-only", "--out", str(summary), "--ledger", str(evidence)]
s.main()
assert summary.read_text() == text and json.loads(evidence.read_text())["records"] == report["records"]
# A later campaign that times a pair again replaces it, even where the earlier one cannot be compared: ippon's timed
# grow requests were set aside for a busy machine and replaced by later ones the other arm never sent.
import shutil
later = out.parent / "later"
shutil.copytree(out, later)
grow = folder / f"{ours}.grow.json"
run = json.loads(grow.read_text())
for row in run["requests"]:
    if row["phase"] == "timed":
        row["rep"] += 100
        row["request_sha256"] = "0" * 64
grow.write_text(json.dumps(run))
try:
    s.readme_more([folder])
except ValueError as exc:
    assert "no request in common" in str(exc)
else:
    raise AssertionError("unmatched grow arms accepted")
both = s.readme_more([folder, later / "r1"])
assert [r["sources"] for r in both["records"] if r["workload"] == "grow" and not r.get("alone")] == [
    [s.portable(later / "r1" / f"{other}.grow.json"), s.portable(later / "r1" / f"{ours}.grow.json")]
]
# A missing standalone arm cannot be published as a complete campaign.
(folder / f"{solo}.typical.json").unlink()
try:
    s.readme_more([folder])
except ValueError as exc:
    assert "no finished solo run" in str(exc)
else:
    raise AssertionError("missing standalone workload accepted")
"""
    run_kit(program, str(tmp_path))


def test_http_connection_retries_are_bounded_counted_and_timed(tmp_path: Path) -> None:
    program = r"""
import argparse, sys, urllib.error
from pathlib import Path
for clients, failure in [(1, ConnectionResetError), (4, ConnectionResetError),
                         (1, BrokenPipeError), (4, BrokenPipeError), (1, OSError), (4, OSError)]:
    attempts = {}
    def decide(req):
        key = req["state"]
        attempts[key] = attempts.get(key, 0) + 1
        if attempts[key] == 1:
            raise urllib.error.URLError(failure("peer closed the connection"))
        qs = req["systemone"]["questions"]
        return {name: dict.fromkeys(common.g.outcome_names(q), 1 / len(common.g.outcome_names(q)))
                for name, q in qs.items()}, 100, {"source": "fixture"}
    args = common.add_args(argparse.ArgumentParser()).parse_args([
        "--engine", "fixture", "--http", "--sizes", "100", "--questions", "1", "--n", "1", "--warmup", "0",
        "--clients", str(clients), "--max-input", "32768", "--out", str(Path(sys.argv[2]) / f"{clients}.json")])
    result = common.run(args, decide, {}, server_pid=123)
    assert len(result["requests"]) == clients and not result["errors"]
    assert all(row["transport_retries"] == 1 and row["ms"] >= 49 for row in result["requests"])
    assert all(row["extra"] == {"source": "fixture"} for row in result["requests"])
    assert set(attempts.values()) == {2}  # each logical call reused exactly the same request
    assert result["config"]["http_transport_retries"] == 2
    assert all(len(row["transport_errors"]) == 1 for row in result["requests"])
for error, http, expected in [
    (urllib.error.URLError(ConnectionResetError(54, "reset")), True, 3),
    (ConnectionResetError(54, "reset"), False, 1),
    (urllib.error.URLError(BrokenPipeError(32, "broken pipe")), True, 3),
    (urllib.error.URLError(OSError(22, "Invalid argument")), True, 3),
    (urllib.error.URLError(ConnectionRefusedError(61, "refused")), True, 1),
    (urllib.error.URLError(TimeoutError("timeout")), True, 1),
    (urllib.error.HTTPError("http://localhost", 500, "server error", {}, None), True, 1),
]:
    calls = []
    def fail(req):
        calls.append(req)
        raise error
    try:
        common.invoke(fail, {"same": "payload"}, http)
    except type(error):
        pass
    else:
        raise AssertionError("failure swallowed")
    assert len(calls) == expected
"""
    run_kit(program, str(tmp_path), timeout=20)


def test_every_gguf_build_gets_llama_flags_on_small_and_large_macs() -> None:
    """On an 8 GB Mac the kit shrinks llama.cpp's context, batches and cache; H2O-Lightning-4B keeps llama.cpp's own
    batch sizes, and shrinking flags it doesn't have crashed every round there before a pair was looked at."""
    run_kit(
        """
import run
for name in run.LLAMA_FLAGS:
    for small in (False, True):
        flags = run.llama_flags(name, small)
        assert flags[flags.index("-c") + 1] == (str(run.SMALL_CONTEXT) if small else run.LLAMA_FLAGS[name][3])
assert "-b" not in run.llama_flags("h2o-4b", True)
assert run.llama_flags("d1-2B", True)[run.llama_flags("d1-2B", True).index("-ub") + 1] == str(run.SMALL_WINDOW)
"""
    )


def test_a_recorded_round_keeps_the_version_it_ran_and_a_missing_folder_is_refused() -> None:
    program = """
import summarize as s
round_ = s.ROOT / "benchmarks/results/m5pro-2026-10-21/makers"
models = [record["model"] for record in s.comparisons([round_])]
assert "pplx-decider flash v1.2 · BF16" in models and not any("v1.3" in model for model in models)
sys.argv = ["summarize.py", str(round_.with_name("creators"))]
try:
    s.main()
except SystemExit as stopped:
    assert stopped.code == 2
else:
    raise AssertionError("a result folder that does not exist was summarized")
"""
    run_kit(program)
