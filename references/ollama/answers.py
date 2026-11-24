"""Ollama's own answers to a creator reference's requests, against the creator's fp32 answers.

Starts the given Ollama binary on its own port and model store, sends every request of an ippon reference file to its
POST /v1/systemone as the build named, and records each answer's largest probability difference from the
reference's fp32 answers, whether the most likely option changed, and what Ollama refused. `ippon verify` measures
ippon on the same build against the same file, so the two distances compare the runtimes on one set of weights.

    python3 -I references/ollama/answers.py --ollama ~/bin/ollama-v0.40.13/ollama --models-dir ~/models/ollama \
        --model d1:2b --reference references/d1/d1-2B.json.gz --out references/ollama/d1-2b.ollama.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def call(url: str, body: dict | None = None, timeout: float = 900) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def probabilities(answers: dict) -> dict[str, dict[str, float]]:
    """System One answers as {question: {outcome: probability}}, a noul as yes and no."""
    out = {}
    for name, a in answers.items():
        if a.get("type") == "noul" or "noul" in a:
            out[name] = {"yes": float(a["noul"]), "no": 1 - float(a["noul"])}
        else:
            out[name] = {str(k): float(v) for k, v in a["probabilities"].items()}
    return out


def reference_answers(case: dict) -> dict[str, dict[str, float]]:
    """A reference case's fp32 answers as {question: {outcome: probability}}: its System One answers, or its
    probabilities in each question's option order (a noul's yes then no, a score's levels from 0)."""
    if "answers" in case:
        return probabilities(case["answers"])
    out: dict[str, dict[str, float]] = {}
    if "probabilities" not in case:  # a request the creator's code refused: nothing to compare against
        return out
    for name, p in case["probabilities"].items():
        question = case["request"]["questions"][name]
        if question["type"] == "noul":
            keys = ["yes", "no"]
        elif question["type"] == "score":
            keys = [str(i) for i in range(len(p))]
        else:
            keys = [str(k) for k in question["criteria"]]
        out[name] = dict(zip(keys, map(float, p), strict=True))
    return out


def top(p: dict[str, float]) -> str:
    return max(p, key=lambda k: p[k])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ollama", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--model", required=True, help="the build's name:tag in the store")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    with gzip.open(args.reference) as f:
        reference = json.load(f)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = {k: v for k, v in os.environ.items() if not k.startswith("OLLAMA_")} | {
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_MODELS": str(args.models_dir.expanduser()),
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_NOPRUNE": "1",
        "OLLAMA_NO_CLOUD": "1",
    }
    server = subprocess.Popen([str(args.ollama.expanduser()), "serve"], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    try:
        for _ in range(240):
            try:
                tags = call(f"{base}/api/tags")["models"]
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise SystemExit("ollama serve did not answer within 120 s")
        listed = next(m for m in tags if args.model in (m["name"], m["model"]))
        show = call(f"{base}/api/show", {"model": args.model})
        cases, refused = [], []
        for case in reference["requests"]:
            try:
                answer = call(f"{base}/v1/systemone", {"model": args.model, **case["request"]})
            except urllib.error.HTTPError as error:
                refused.append({"name": case["name"], "status": error.code,
                                "error": error.read()[:300].decode(errors="replace")})  # fmt: skip
                continue
            ours, theirs = probabilities(answer["answers"]), reference_answers(case)
            worst, changed = 0.0, []
            for q, want in theirs.items():
                got = ours[q]
                worst = max(worst, *(abs(got[k] - want[k]) for k in want))
                ranked = sorted(want.values(), reverse=True)
                near_tie = len(ranked) > 1 and ranked[0] - ranked[1] <= reference["bound"]
                if top(got) != top(want) and not near_tie:
                    changed.append(q)
            cases.append({"name": case["name"], "worst": worst, "top_changed": changed,
                          "input_tokens": (answer.get("usage") or {}).get("input_tokens")})  # fmt: skip
        details = show.get("details") or {}
        result = {
            "reference": args.reference.as_posix(),
            "maker": reference["model"],
            "bound": reference["bound"],
            "ollama": call(f"{base}/api/version")["version"],
            "build": args.model,
            "manifest_digest": listed["digest"],
            "quantization": details.get("quantization_level"),
            "requests": len(reference["requests"]),
            "answered": len(cases),
            "worst": max((c["worst"] for c in cases), default=None),
            "past_bound": sum(c["worst"] > reference["bound"] for c in cases),
            "top_changed": sum(bool(c["top_changed"]) for c in cases),
            "refused": refused,
            "cases": cases,
        }
    finally:
        server.send_signal(signal.SIGINT)
        server.wait(timeout=60)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.model}: {result['answered']}/{result['requests']} answered, worst {result['worst']:.4f} "
          f"(bound {reference['bound']}), {result['past_bound']} past it, {result['top_changed']} top answers changed, "
          f"{len(refused)} refused")  # fmt: skip


if __name__ == "__main__":
    main()
