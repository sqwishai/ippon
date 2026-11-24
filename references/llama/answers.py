"""Compare llama-server's decisions with a creator's fp32 reference, retaining every refusal.

Uses the same probability conversion as references/ollama/answers.py. The GGUF is read unchanged; no metadata
override, prompt translation or model patch supplies a protocol that the server does not implement.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ollama"))
from answers import call, probabilities, reference_answers, top

MEDIA = Path(__file__).resolve().parents[1] / "media"
TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".tif": "image/tiff",
         ".tiff": "image/tiff", ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
         ".mkv": "video/x-matroska"}  # fmt: skip


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def served(request: dict) -> dict:
    """A reference request as llama-server takes it: its images, then its videos, as data URLs in `files`."""
    paths = [*request.get("images", ()), *request.get("videos", ())]
    if not paths:
        return request
    body = {k: v for k, v in request.items() if k not in ("images", "videos")}
    body["files"] = [
        f"data:{TYPES[Path(p).suffix.lower()]};base64," + base64.b64encode((MEDIA / p).read_bytes()).decode()
        for p in paths
    ]
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, help="the build's media encoder, for a reference with images or video")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.reference) as source:
        reference = json.load(source)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    flags = ["-ngl", "all", "-c", "32768", "-b", "32768", "-ub", "32768", "-np", "16", "-kvu"]
    command = [str(args.server), "-m", str(args.gguf), "--host", "127.0.0.1", "--port", str(port), *flags]
    if args.mmproj:
        command += ["--mmproj", str(args.mmproj)]
        flags += ["--mmproj", args.mmproj.name]  # recorded without the local folder
    version = subprocess.run([str(args.server), "--version"], capture_output=True, text=True, check=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    cases, refused = [], []
    try:
        for _ in range(1200):
            if server.poll() is not None:
                raise SystemExit(f"llama-server exited with {server.returncode}; see the server log")
            try:
                if call(f"{base}/health", timeout=5).get("status") == "ok":
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            raise SystemExit("llama-server did not become healthy within 600 seconds")
        for case in reference["requests"]:
            try:
                answer = call(f"{base}/v1/systemone", served(case["request"]))
            except urllib.error.HTTPError as error:
                refused.append({"name": case["name"], "status": error.code, "error": error.read().decode()})
                continue
            got, want = probabilities(answer["answers"]), reference_answers(case)
            worst, changed = 0.0, []
            for question, expected in want.items():
                worst = max(worst, *(abs(got[question][key] - value) for key, value in expected.items()))
                ranked = sorted(expected.values(), reverse=True)
                near_tie = len(ranked) > 1 and ranked[0] - ranked[1] <= reference["bound"]
                if top(got[question]) != top(expected) and not near_tie:
                    changed.append(question)
            cases.append({"name": case["name"], "worst": worst, "top_changed": changed, "answers": answer})
    finally:
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=60)
    result = {
        "reference": args.reference.as_posix(),
        "reference_sha256": sha256(args.reference),
        "maker": reference["model"],
        "bound": reference["bound"],
        "llama_server_version": (version.stdout + version.stderr).strip(),
        "gguf": args.gguf.name,
        "gguf_sha256": sha256(args.gguf),
        "flags": flags,
        "requests": len(reference["requests"]),
        "answered": len(cases),
        "worst": max((case["worst"] for case in cases), default=None),
        "past_bound": sum(case["worst"] > reference["bound"] for case in cases),
        "top_changed": sum(bool(case["top_changed"]) for case in cases),
        "refused": refused,
        "cases": cases,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.gguf.name}: {len(cases)}/{result['requests']} answered; worst {result['worst']}; "
          f"{len(refused)} refused")  # fmt: skip


if __name__ == "__main__":
    main()
