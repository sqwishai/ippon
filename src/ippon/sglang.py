"""SGLang's HTTP decisions format translated to and from System One requests and answers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import InvalidRequest
from .schema import (
    MIN_SCORE_LEVELS,
    Answer,
    QuestionType,
    carried,
    expected_level,
    has_image_parts,
    tempered,
    top_option,
)

# SGLang's decisions API (POST /v1/decisions, docs.sglang.io/docs/supported-models/decision_models), which `serve`
# answers beside System One at that path: the same questions as a list, each with an `id`, a `type` of choice, score
# or yes_no and the `question`; the state as `input`.
SGLANG_PROMPT_FORMAT = 1  # the `prompt_format_version` its responses carry; a request pinning another is refused
SGLANG_TYPES: dict[str, QuestionType] = {"choice": "choice", "score": "score", "yes_no": "noul"}
MIN_SGLANG_OPTIONS = 2  # as many past that as the model takes, which the decider checks
_SGLANG_FIELDS = frozenset(
    {
        "model",
        "input",
        "images",
        "questions",
        "temperature",
        "chat_template_kwargs",
        "prompt_format_version",
        "return_prompt_token_ids",
    }
)
_SGLANG_QUESTION_FIELDS = {
    "choice": frozenset({"id", "type", "question", "options"}),
    "score": frozenset({"id", "type", "question", "levels"}),
    "yes_no": frozenset({"id", "type", "question", "yes", "no"}),
}
_THINKING = frozenset({"enable_thinking", "thinking"})  # chat template switches a request may only turn off


@dataclass(frozen=True)
class SGLang:
    """How an SGLang decisions request wants its answers: its `model` echoed back, its `temperature` on the labels'
    scores."""

    model: str = "default"
    temperature: float = 1.0


def is_sglang(body: object) -> bool:
    """Whether a request body is SGLang's decisions request rather than System One's: its questions are a list
    (System One names them in an object), or it has SGLang's `input` (System One's is `state`) and no object of
    questions, so SGLang's own refusal names what it lacks."""
    if not isinstance(body, Mapping):
        return False
    questions = body.get("questions")
    return isinstance(questions, list) or ("input" in body and not isinstance(questions, Mapping))


def from_sglang(body: Mapping[str, Any]) -> tuple[dict[str, Any], SGLang]:
    """An SGLang decisions request as the System One request that asks the same, and how to answer it
    (`sglang_response`); InvalidRequest for one SGLang refuses, or one that asks what ippon doesn't do.

    A choice's options become its criteria, each name with its description; a score's levels its levels, lowest
    first; a yes_no a noul whose instructions are the question and whose criteria are its `yes` and `no` descriptions.
    The input is the state as the request gives it (text, or a JSON object or array, which the model's format renders
    as its creator's code renders a state), and `images` are the request's images: data URLs or base64, as System One's
    `images` (a URL is never fetched; `detail` is not read). A decision model reads its creator's own prompt, so a
    request's chat template options (other than thinking switched off, which it always is) and
    `return_prompt_token_ids` are refused, as is a `prompt_format_version` other than SGLANG_PROMPT_FORMAT."""
    for key in body:
        if key not in _SGLANG_FIELDS:
            raise InvalidRequest(f"unknown field {key!r}")
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise InvalidRequest("'model' must be a string")
    state = body.get("input")
    if not isinstance(state, (str, Mapping, list)) or (isinstance(state, str) and not state.strip()):
        raise InvalidRequest("'input' is required: a string, not blank, or a JSON object or array")
    if has_image_parts(state):
        raise InvalidRequest("input: images go in 'images'")
    temperature = body.get("temperature")
    temperature = 1.0 if temperature is None else temperature
    try:  # an integer too large for a float (400 digits) is no temperature either
        valid = not isinstance(temperature, bool) and 0 < float(temperature) < math.inf
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid or not isinstance(temperature, (int, float)):
        raise InvalidRequest("'temperature' must be a number above 0")
    version = body.get("prompt_format_version")
    if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
        raise InvalidRequest("'prompt_format_version' must be an integer")
    if version is not None and version != SGLANG_PROMPT_FORMAT:
        raise InvalidRequest(f"prompt_format_version {version}: this server's is {SGLANG_PROMPT_FORMAT}")
    ids = body.get("return_prompt_token_ids")
    if ids is not None and not isinstance(ids, bool):
        raise InvalidRequest("'return_prompt_token_ids' must be true or false")
    if ids:
        raise InvalidRequest(
            "return_prompt_token_ids: a decision model reads its creator's own prompt and answers with its own head, "
            "so there are no chat prompt and label tokens to replay "
            "(`ippon lint --model MODEL --prompt` shows what it reads)"
        )
    kwargs = body.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, Mapping):
        raise InvalidRequest("'chat_template_kwargs' must be an object")
    others = sorted(key for key, value in (kwargs or {}).items() if not (key in _THINKING and value is False))
    if others:
        raise InvalidRequest(
            f"chat_template_kwargs {', '.join(map(repr, others))}: a decision model reads its creator's own prompt, "
            "which takes no chat template options (and never thinks)"
        )
    asked = body.get("questions")
    if not isinstance(asked, list) or not asked:
        raise InvalidRequest("'questions' must list one question or more")
    questions: dict[str, Any] = {}
    for i, question in enumerate(asked):
        qid, converted = _sglang_question(i, question)
        if qid in questions:
            raise InvalidRequest(f"questions[{i}]: the id {qid!r} is used twice")
        questions[qid] = converted
    request: dict[str, Any] = {"model": None if model in (None, "default") else model, "state": state}
    request["questions"] = questions
    if body.get("images") is not None:
        request["images"] = _sglang_images(body["images"])
    return request, SGLang(model or "default", float(temperature))


def _sglang_question(i: int, body: object) -> tuple[str, dict[str, Any]]:
    """A question of an SGLang request: its id, and the System One question it asks."""
    if not isinstance(body, Mapping):
        raise InvalidRequest(f"questions[{i}]: a question is a JSON object")
    qid = body.get("id")
    if not isinstance(qid, str) or not qid.strip():
        raise InvalidRequest(f"questions[{i}]: 'id' must be a string, not blank")
    kind = body.get("type")
    if kind not in SGLANG_TYPES:
        raise InvalidRequest(f"{qid}: 'type' must be one of {', '.join(SGLANG_TYPES)}")
    for key in body:
        if key not in _SGLANG_QUESTION_FIELDS[kind]:
            raise InvalidRequest(f"{qid}: unknown field {key!r} for a {kind} question")
    text = body.get("question")
    if not _given(text):
        raise InvalidRequest(f"{qid}: 'question' is required: a string, not blank, or a JSON object or array")
    if kind == "choice":
        criteria: Any = _sglang_options(qid, body.get("options"))
    elif kind == "score":
        criteria = _sglang_levels(qid, body.get("levels"))
    else:
        criteria = {}
        for key, side in (("true", "yes"), ("false", "no")):
            if body.get(side) is not None and not _given(body[side]):
                raise InvalidRequest(
                    f"{qid}: {side!r} is a description: a string, not blank, or a JSON object or array"
                )
            if body.get(side) is not None:
                criteria[key] = body[side]
    return qid, {"type": SGLANG_TYPES[kind], "instructions": text, "criteria": criteria}


def _given(value: object) -> bool:
    """Whether `value` is text that isn't blank, or a JSON object or array: what SGLang takes for a question, a
    level or a description."""
    return isinstance(value, (Mapping, list)) or (isinstance(value, str) and bool(value.strip()))


def _sglang_options(qid: str, options: object) -> dict[str, Any]:
    """A choice's options, {"name", "description"?} each, as System One criteria: names refused as SGLang refuses
    them (blank, holding control or line break characters, or the same as another once trimmed and case-folded)."""
    if not isinstance(options, list) or len(options) < MIN_SGLANG_OPTIONS:
        raise InvalidRequest(f"{qid}: 'options' must list {MIN_SGLANG_OPTIONS} options or more")
    criteria: dict[str, Any] = {}
    seen: set[str] = set()
    for j, option in enumerate(options):
        if not isinstance(option, Mapping) or not set(option) <= {"name", "description"}:
            raise InvalidRequest(f'{qid}: options[{j}] is {{"name": ..., "description": ...}}')
        name, description = option.get("name"), option.get("description")
        if not isinstance(name, str) or not name.strip() or not name.isprintable():
            raise InvalidRequest(f"{qid}: options[{j}]: 'name' must be a string, not blank, on one line")
        if name.strip().casefold() in seen:
            raise InvalidRequest(f"{qid}: options[{j}]: the name {name!r} is another option's")
        if description is not None and not _given(description):
            raise InvalidRequest(f"{qid}: options[{j}]: 'description' must be a string, not blank, or a JSON object")
        seen.add(name.strip().casefold())
        criteria[name] = description
    return criteria


def _sglang_levels(qid: str, levels: object) -> list[Any]:
    if not isinstance(levels, list) or len(levels) < MIN_SCORE_LEVELS:
        raise InvalidRequest(f"{qid}: 'levels' must list {MIN_SCORE_LEVELS} levels or more, lowest first")
    for j, level in enumerate(levels):
        if not _given(level):
            raise InvalidRequest(f"{qid}: levels[{j}] must be a string, not blank, or a JSON object or array")
    return levels


def _sglang_images(images: object) -> list[Any]:
    """An SGLang request's images as System One's `images`: each {"url": ..., "detail"?} or the URL alone."""
    if not isinstance(images, list):
        raise InvalidRequest("'images' must be a list of images: {\"url\": ...} or the URL alone")
    urls = []
    for j, image in enumerate(images):
        if isinstance(image, Mapping) and not set(image) <= {"url", "detail"}:
            raise InvalidRequest(f'images[{j}]: an image is {{"url": ..., "detail": ...}} or the URL alone')
        urls.append(image.get("url") if isinstance(image, Mapping) else image)
    return urls


def sglang_response(answers: Mapping[str, Answer], response: Mapping[str, Any], how: SGLang) -> dict[str, Any]:
    """The SGLang decisions response for a request's answers (its System One `response` gives the usage and what was
    cut to fit the model's window): each question's probabilities at the request's temperature, which divides the
    labels' log-probabilities (a softmax's logits, up to a constant) before they are normalized again, and its choice
    (the most likely option) or score (the probability-weighted mean level). `label_mass` is 1: a decision model
    answers with its own head over the question's options, so all of its probability is on them."""
    shaped: dict[str, dict[str, Any]] = {}
    for qid, answer in answers.items():
        q = answer.question
        at = tempered(answer.probabilities, how.temperature)
        probabilities = dict(zip(q.options, at, strict=True))
        if q.type == "noul":
            shaped[qid] = {
                "type": "yes_no",
                "probabilities": {"yes": probabilities["true"], "no": probabilities["false"]},
            }
        elif q.type == "choice":
            shaped[qid] = {"type": "choice", "probabilities": probabilities, "choice": top_option(q.options, at)}
        else:
            shaped[qid] = {"type": "score", "probabilities": probabilities, "score": expected_level(at)}
        shaped[qid]["label_mass"] = 1.0
    out = {
        "object": "decisions",
        "model": how.model,
        "prompt_format_version": SGLANG_PROMPT_FORMAT,
        "answers": shaped,
        "usage": {"prompt_tokens": int(response.get("usage", {}).get("input_tokens", 0)), "completion_tokens": 0},
    }
    return out | carried(response)
