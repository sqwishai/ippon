"""Answers checked against a reference: the accuracy contract every supported model and build is held to.

The reference is the model creator's own code run in fp32. A model's support in ippon, and each of its
precisions, is accepted only when on a fixed set of requests it renders the reference's tokens exactly,
every probability is within a stated bound of the reference's, and the most likely option is the same. A question
whose two most likely options are within the bound of each other in the reference is a near tie: a difference the
bound allows can swap them, so a changed top answer there is reported but is not a failure.
Speed measured with other answers would not be a comparison.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _emulate
from ._portable import record
from .arch.common import is_placeholder, media_digest, media_of
from .compare_arms import ARM_DTYPE, agreement, answers, difference, lookup
from .decider import Decider
from .errors import DecisionError, InvalidRequest, UnsupportedModel
from .processes import run_json
from .questions import with_media
from .schema import Request, parse_request


@dataclass(frozen=True)
class Comparison:
    """One request: the largest difference of any probability, and whether every question's most likely
    option is the reference's."""

    max_abs_difference: float
    top_agrees: bool  # every question's top answer is the reference's, near ties aside
    near_ties: int = 0  # questions whose top answer changed where the reference's top two are within `tie`
    threshold_crossings: tuple[tuple[float, int], ...] = ()  # diagnostic: option probabilities crossing each threshold


def _margin(probabilities: Sequence[float]) -> float:
    first, second = sorted(probabilities, reverse=True)[:2] if len(probabilities) > 1 else (1.0, 0.0)
    return first - second


def compare(
    got: Sequence[Sequence[float]],
    want: Sequence[Sequence[float]],
    tie: float = 0.0,
    thresholds: Sequence[float] = (),
) -> Comparison:
    """Compare a request's answers, one probability list per question, with the reference's. A changed top answer
    where the reference's top two are at most `tie` apart counts as a near tie, not a disagreement."""
    diff, changed = difference(got, want)  # ValueError when they differ in shape
    if not got or any(not g for g in got):
        raise ValueError("nothing to compare")
    pairs = list(zip(got, want, strict=True))
    if any(not math.isfinite(p) for g, w in pairs for values in (g, w) for p in values):
        raise ValueError("answers and reference must contain finite probabilities")
    ties = sum(_margin(want[i]) <= tie for i in changed)
    crossings = tuple(
        (t, sum((a >= t) != (b >= t) for g, w in pairs for a, b in zip(g, w, strict=True))) for t in thresholds
    )
    return Comparison(
        max_abs_difference=diff, top_agrees=ties == len(changed), near_ties=ties, threshold_crossings=crossings
    )


# -- a model against its creator's reference file ---------------------------------------------------------------------
#
# A reference file (made by references/<adapter>/make_reference.py, which runs the creator's own code in fp32 in its own
# environment) holds requests and, for each, the token ids of every prompt the creator's code reads, each question's
# probabilities in the request's option order (the model's, before the creator's server shapes its answer: a yes/no
# floor, which jumps at 0.5, or rounding; a format's `answer` returns them the same way), and the creator's input-token
# count:
#
#   {"schema": "ippon-reference/1", "model": {"repo": ..., "revision": ...}, "maker": {...}, "bound": 0.01,
#    "requests": [{"name": ..., "request": {...}, "prompts": [[ids...], ...], "probabilities": {"q": [...]},
#                  "input_tokens": 412}, ...]}
#
# and optionally "thresholds": [0.5, 0.7, 0.9] (probabilities whose crossings are counted, a diagnostic) and
# "reference_kind": "serving", "checkpoint_dequantized" or "original_fp32" (what the probabilities came from; the
# first two name an Ollama build's checkpoint identity as the model's "revision", which the loaded build must
# match: the SHA-256 of the sorted digests of its tensor blobs and the assets its adapter reads, ollama.py).

REFERENCE_SCHEMA = "ippon-reference/1"
# how far a probability may be from the reference's where the reference names no bound, and for --against, --invariance
BOUND = 0.01


def load_reference(path: str | Path) -> dict[str, Any]:
    """The reference at `path`; InvalidRequest when it can't be read or isn't a complete reference."""
    path = Path(path)
    try:
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        reference = json.loads(raw)
    except (OSError, EOFError, ValueError) as error:  # unreadable, a broken gzip, or not JSON
        raise InvalidRequest(f"{path}: can't read a {REFERENCE_SCHEMA} file ({error})") from None
    if not isinstance(reference, dict) or reference.get("schema") != REFERENCE_SCHEMA:
        raise InvalidRequest(f"{path}: not a {REFERENCE_SCHEMA} file")
    requests = reference.get("requests")
    if not isinstance(requests, list) or not requests:
        raise InvalidRequest(f"{path}: the reference contains no requests")
    if reference.get("complete") is False or any(
        reference.get(key, len(requests)) != len(requests) for key in ("planned_requests", "completed_requests")
    ):
        raise InvalidRequest(f"{path}: reference generation is incomplete")
    return dict(reference)


def media_files(path: str | Path, reference: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], dict[str, bytes]]:
    """A function giving a reference request's files by name: each of its `media` ([{name, sha256}]) read from the
    reference's `media_format.directory` (relative to a folder above the file: the repository's) and checked by its
    sha256. The request names them: its `images` (in order, an image sent twice named twice), its `audio` clip, its
    `videos`, and a spoken question's clip as {"type": "input_audio", "input_audio": {"file": NAME}}
    (`reference_request`)."""
    directory = reference.get("media_format", {}).get("directory")
    found = None
    if directory:
        found = next((d / directory for d in Path(path).resolve().parents if (d / directory).is_dir()), None)

    def read(case: Mapping[str, Any]) -> dict[str, bytes]:
        if not case.get("media"):
            return {}
        if found is None:
            raise FileNotFoundError(f"{path}: its media directory {directory!r} is not beside it")
        files = {}
        for item in case["media"]:
            data = (found / item["name"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"{found / item['name']}: not the file the reference was made with")
            files[item["name"]] = data
        return files

    return read


def reference_request(case: Mapping[str, Any], files: Mapping[str, bytes]) -> Request:
    """A reference request as ippon takes it: its images, clip and videos, named in the reference, from `files`, and
    each spoken question's clip, named as its input_audio part's `file`, as that part's base64 `data`."""
    body = dict(case["request"], images=None, audio=None, videos=None)
    questions = {}
    for name, q in body["questions"].items():
        said = q.get("instructions") if isinstance(q, Mapping) else None
        if isinstance(said, Mapping) and said.get("type") == "input_audio" and "file" in said.get("input_audio", {}):
            data = base64.b64encode(files[said["input_audio"]["file"]]).decode()
            q = dict(q, instructions={"type": "input_audio", "input_audio": {"data": data}})
        questions[name] = q
    clip = case["request"].get("audio")
    images = [files[n] for n in case["request"].get("images") or ()]
    videos = [files[n] for n in case["request"].get("videos") or ()]
    request = parse_request(dict(body, questions=questions))
    return with_media(request, images, None if clip is None else files[clip], videos)


@dataclass(frozen=True)
class RequestReport:
    name: str
    tokens_match: bool
    comparison: Comparison | None
    input_tokens_match: bool


@dataclass(frozen=True)
class Report:
    model: str
    bound: float
    requests: tuple[RequestReport, ...]
    against: str = "the reference"  # what the answers are compared with: a reference file, or another runtime
    notes: tuple[str, ...] = ()  # how the bound was found, when it was measured here

    @property
    def failures(self) -> list[str]:
        out = []
        for r in self.requests:
            if not r.tokens_match:
                out.append(f"{r.name}: prompt tokens differ from the creator's")
            if r.comparison is not None and (
                r.comparison.max_abs_difference > self.bound or not r.comparison.top_agrees
            ):
                out.append(
                    f"{r.name}: {r.comparison.max_abs_difference:.5f} from {self.against} (bound {self.bound})"
                    + ("" if r.comparison.top_agrees else ", a different top answer")
                )
            if not r.input_tokens_match:
                out.append(f"{r.name}: input token count differs from the creator's")
        return out

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def worst(self) -> float:
        return max((r.comparison.max_abs_difference for r in self.requests if r.comparison), default=0.0)

    def text(self) -> str:
        if all(r.comparison is None for r in self.requests):
            lines = [f"{self.model}: {len(self.requests)} requests, prompt tokens and counts only"]
        else:
            versus = "" if self.against == "the reference" else f" against {self.against}"
            lines = [
                f"{self.model}{versus}: {len(self.requests)} requests, worst {self.worst:.5f} (bound {self.bound})"
            ]
        lines += [f"  {n}" for n in self.notes] + [f"  {f}" for f in self.failures]
        ties = [r.name for r in self.requests if r.comparison is not None and r.comparison.near_ties]
        if ties:
            lines.append(f"  a different top answer in a near tie (not a failure): {', '.join(ties)}")
        lines.append("passed" if self.passed else "failed")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return record(
            {
                "model": self.model,
                "against": self.against,
                "notes": self.notes,
                "bound": self.bound,
                "worst": self.worst,
                "passed": self.passed,
                "failures": self.failures,
                "requests": [
                    {
                        "name": r.name,
                        "tokens_match": r.tokens_match,
                        "input_tokens_match": r.input_tokens_match,
                        "max_abs_difference": r.comparison.max_abs_difference if r.comparison else None,
                        "top_agrees": r.comparison.top_agrees if r.comparison else None,
                        "near_ties": r.comparison.near_ties if r.comparison else None,
                        "threshold_crossings": dict(r.comparison.threshold_crossings) if r.comparison else {},
                    }
                    for r in self.requests
                ],
            }
        )


def verify_model(model: str, reference: str | Path, **load: Any) -> Report:
    """Run every request of a reference file through `model` and compare tokens, probabilities and token counts;
    a tokens-only reference (no probabilities) checks the prompts and their count without running the model.
    `load` are `Decider.load`'s options (precision, profile, switches, chunk, ...)."""
    ref = load_reference(reference)
    media = media_files(reference, ref)
    image_token = ref.get("media_format", {}).get("image_token")
    audio_token = ref.get("media_format", {}).get("audio_token")
    video_token = ref.get("media_format", {}).get("video_token")
    tokens_only = all("probabilities" not in case for case in ref["requests"])
    bound = float(ref.get("bound", BOUND))
    reports = []
    with Decider.load(model, **({"pin_memory": False} | load | {"warm_up": not tokens_only})) as decider:
        checkpoint = decider.details.get("checkpoint")
        same_checkpoint = ref.get("reference_kind") in ("serving", "checkpoint_dequantized")
        if same_checkpoint and checkpoint and ref.get("model", {}).get("revision") != checkpoint:
            raise UnsupportedModel("this same-checkpoint reference names a different Ollama checkpoint")
        with decider.uncached():  # every request read, even with kept answers on
            for case in ref["requests"]:
                request = reference_request(case, media(case))
                if "error" in case:  # a request the creator's code refuses: it matches when ippon refuses it too
                    try:
                        decider.render(request)
                    except DecisionError:
                        reports.append(RequestReport(case["name"], True, None, True))
                    else:
                        reports.append(RequestReport(case["name"], False, None, False))
                    continue
                rendering = decider.format.render(request)
                # a media reference names the creator's image token (and audio and video tokens): its prompts are the
                # whole prompt with that token where the placeholders (arch/common) are, the clip's rows' the audio
                # token, the pictures' the image token, the rest (a video's frame pairs') the video token
                if image_token is None:
                    text = [list(p.tokens) for p in rendering.prompts]
                else:
                    clips = [*request.audio, *(q.spoken for q in request.questions)]
                    heard = {media_digest(c.sha256) for c in clips if c is not None}
                    pictures = {media_digest(i.sha256) for i in request.images}
                    found = dict.fromkeys(heard, audio_token) | dict.fromkeys(pictures, image_token)
                    other = image_token if video_token is None else video_token  # a video's frame pairs'
                    text = [
                        [found.get(media_of(t)[0], other) if is_placeholder(t) else t for t in p.tokens]
                        for p in rendering.prompts
                    ]
                tokens_match = text == case["prompts"]
                if "probabilities" in case:
                    result = decider.answer(request)
                    got = [list(result[q.id].probabilities) for q in request.questions]
                    want = [case["probabilities"][q.id] for q in request.questions]
                    comparison: Comparison | None = compare(got, want, tie=bound, thresholds=ref.get("thresholds", ()))
                    counted = result.response["usage"]["input_tokens"]
                else:  # a tokens-only reference: the prompts and their count, without running the model
                    comparison, counted = None, decider.format.usage(rendering)["input_tokens"]
                reports.append(
                    RequestReport(case["name"], tokens_match, comparison, case.get("input_tokens") in (None, counted))
                )
    kinds = {
        "serving": "the published runtime on this checkpoint",
        "checkpoint_dequantized": "this checkpoint dequantized in FP32",
        "original_fp32": "the creator's original FP32 checkpoint",
    }
    return Report(model, bound, tuple(reports), against=kinds.get(str(ref.get("reference_kind", "")), "the reference"))


# -- a build against another runtime of the same weights ------------------------------------------------------------
#
# A quantized build has no fp32 reference of its own: its creator's fp32 model is another model. What can be checked is
# that ippon computes, on the build's own weights, what the runtime the build was published for computes (the
# creator's MLX path for its MLX builds, a community port for a community build), the way a reference file holds a
# model to its creator's code: the arm (`compare_arms/`) runs in its own environment, as `bench --compare` runs it,
# once in fp32 on the build's weights (ARM_DTYPE, on the M4's exact kernels on an M5: the reference) and once as
# published (its own precision, as people run it), and the bound is the larger of BOUND and that published
# run's distance from its fp32 one. An arm that can't compute in fp32 is compared as published, with BOUND.


def _exact_environment() -> dict[str, str]:
    """An arm's environment for its fp32 run: ARM_DTYPE set and, on an M5, the M4's kernels (whose fp32 matmuls are
    exact where the M5's round to about TF32) unless a family is chosen already."""
    env = {**os.environ, ARM_DTYPE: "float32"}
    family = _emulate.architecture_for("m4", _emulate.chip_name())
    if family is not None and _emulate.VARIABLE not in env:
        env[_emulate.VARIABLE] = family
    return env


def verify_against(
    model: str,
    arm: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    bound: float | None = None,
    **load: Any,
) -> Report:
    """Answer `cases` ([{name, request}]) with ippon and with `arm` ("name=python", an arm the model's adapter
    lists) on the same model directory, and compare each question's probabilities with the arm's in fp32; `bound`
    replaces the measured one. `load` are `Decider.load`'s options for ippon's side."""
    ((name, (python, script)),) = lookup.arms_for(model, [arm]).items()
    items = [{"warm": False, "gap": 0.0, "body": c["request"]} for c in cases]
    with tempfile.TemporaryDirectory(prefix="ippon-verify-") as tmp:
        requests = Path(tmp) / "requests.json"
        requests.write_text(json.dumps(items))

        def run(tag: str, env: dict[str, str] | None) -> dict[str, Any]:
            out = Path(tmp) / f"{tag}.json"
            return run_json([python, "-I", script, model, str(requests), str(out)], out, f"arm {name}", env)

        reference = run("fp32", _exact_environment())
        notes = []
        if reference.get("dtype") == "float32":
            published = run("published", None)
            distance = agreement(
                [r["answers"] for r in published["results"]], [r["answers"] for r in reference["results"]]
            )["max_probability_difference"]
            measured = round(distance + 5e-4, 4)
            notes.append(f"{name} as published ({published['runtime']}) is {measured - 5e-4:.5f} from its fp32")
            limit = max(BOUND, measured)
            against = f"{name} in fp32 ({reference['runtime']})"
        else:
            limit, against = BOUND, f"{name} ({reference['runtime']}; it has no fp32 run)"
    limit = limit if bound is None else bound
    reports = []
    with Decider.load(model, **({"pin_memory": False} | load)) as decider, decider.uncached():
        for case, theirs in zip(cases, reference["results"], strict=True):
            response = decider.systemone(dict(case["request"]))
            mine, given = answers(response), theirs["answers"]
            if set(mine) != set(given) or any(set(mine[q]) != set(given[q]) for q in mine):
                raise UnsupportedModel(f"{case['name']}: {name} names other questions or options than ippon")
            got = [list(mine[q].values()) for q in mine]
            want = [[given[q][o] for o in mine[q]] for q in mine]
            counted = response["usage"]["input_tokens"]
            matched = theirs["input_tokens"] in (None, counted)
            reports.append(RequestReport(case["name"], True, compare(got, want, tie=limit), matched))
    return Report(model, limit, tuple(reports), against=against, notes=tuple(notes))
