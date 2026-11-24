"""oMLX's own answers to a creator reference's requests, against the creator's fp32 answers.

Starts the given `omlx` on its own port with the models of one folder (`omlx serve --model-dir`), sends every request
of an ippon reference file to its POST /v1/systemone as the model named, and records each answer's largest probability
difference from the reference's fp32 answers, whether the most likely option changed, and what oMLX refused, as
references/ollama/answers.py does for Ollama. `ippon verify` measures ippon on the same checkpoint against the same
file.

    python3 -I references/omlx/answers.py --omlx ENVS/omlx/bin/omlx --model-dir ~/models/omlx \
        --model d3-mini --reference references/d3/d3-mini.json.gz --out references/omlx/d3-mini.omlx.json
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import mimetypes
import signal
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ollama"))
from answers import call, probabilities, reference_answers, top


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--omlx", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model", required=True, help="the model's folder name under --model-dir")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--media", type=Path, help="references/media: a media reference's images/ and videos/ by name")
    args = p.parse_args()

    def uri(kind: str, name: str) -> str:  # a media reference's file, as the data URI oMLX's System One reads
        path = args.media / name if "/" in name else args.media / kind / name
        mime = mimetypes.guess_type(path.name)[0] or ("video/mp4" if kind == "videos" else "image/png")
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

    def body(request: dict) -> dict:
        out = dict(request)
        for kind in ("images", "videos"):
            if out.get(kind) and args.media:
                out[kind] = [uri(kind, n) if isinstance(n, str) and not n.startswith("data:") else n for n in out[kind]]
        return out

    with gzip.open(args.reference) as f:
        reference = json.load(f)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    cmd = [str(args.omlx.expanduser()), "serve", "--model-dir", str(args.model_dir.expanduser()),
           "--host", "127.0.0.1", "--port", str(port)]  # fmt: skip
    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                if any(m.get("id") == args.model for m in call(f"{base}/v1/models").get("data", [])):
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            raise SystemExit("omlx serve did not list the model within 300 s")
        cases, refused = [], []
        for case in reference["requests"]:
            try:
                answer = call(f"{base}/v1/systemone", {"model": args.model, **body(case["request"])})
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
        omlx = [str(args.omlx.expanduser()), "--version"]
        version = subprocess.run(omlx, check=False, capture_output=True, text=True).stdout
        result = {
            "reference": args.reference.as_posix(),
            "maker": reference["model"],
            "bound": reference["bound"],
            "omlx": version.strip() or "version unrecorded",
            "model": args.model,
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
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    worst = "none" if result["worst"] is None else f"{result['worst']:.4f}"
    print(f"{args.model}: {result['answered']}/{result['requests']} answered, worst {worst} "
          f"(bound {reference['bound']}), {result['past_bound']} past it, {result['top_changed']} top answers changed, "
          f"{len(refused)} refused")  # fmt: skip


if __name__ == "__main__":
    main()
