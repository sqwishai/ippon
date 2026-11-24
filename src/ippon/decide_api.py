"""Sqwish's decide API (`POST /v1/decide`, the request the sqwishai SDK's `decide` sends), answered on this Mac by any
model ippon runs (`Decider.sqwish`), as the SDK's own local engine answers it (`sqwishai.kettei`, 0.5.0: _request.py
and _model.py).

A request is `{"context", "decisions": [{"id", "question", "kind", "outcomes", "rubric", "policy"}], "model",
"temperature"}`. It is checked as the API checks it, with the API's refusals, and each decision becomes one System One
question on the context: a binary decision a yes/no question whose criteria describe yes and no, a single decision a
choice among its outcomes, an ordinal decision a choice among its levels in order. Each model's adapter writes the
question as its creators' code does, so a model the SDK runs reads the prompts the SDK writes. The answer follows the
API's arithmetic: each decision's probabilities by outcome name, its `top` and `p_top`, an ordinal's `expected_level`, a
temperature applied in log space, and a policy's cheapest action or abstention.

Pictures come as the API takes them: `images` [{"id", "media_type", "data"}], each shown where a string context says
`<image:N>` (N its id, 1 to 64), every image with its anchor and every anchor with its image, up to 8, JPEG, PNG or
WebP in base64. Videos and audio clips, which the hosted API doesn't take, come the same way on this Mac, `videos` with
`<video:N>` and `audio` with `<audio:N>`, for a model that sees or hears them (d1-2B, d3, d1-Omni-2B), their format
read from their bytes. A model gets each kind in the order its anchors first appear, renumbered 1, 2, ... in that
order, and the anchors stay in the text, so it still says which is which: a model whose creators place pictures in
the text there (elephant-1 reads `<image:N>`) reads each at its anchor, and one whose creators put them ahead of the
text (d1, d3) reads them first, in that order. A later mention of an anchor becomes words ("image N"). A model that
reads text only refuses them.
"""

from __future__ import annotations

import json
import math
import re
import secrets
from dataclasses import dataclass, replace
from typing import Any

from .errors import InvalidRequest
from .schema import MEDIA, MediaField, Result, carried, tempered

MAX_DECISIONS = 128
MAX_OUTCOMES = 255
MAX_JSON_DEPTH = 32
MAX_ID = 128
LONG_OUTCOME = 500  # single decisions with an outcome name longer than this must all list the same outcomes
TEMPERATURES = (0.2, 5.0)
ABSTAIN = "abstain"  # the action a policy takes when p_top is below its abstain_below
FIELDS = {"id", "question", "kind", "outcomes", "rubric", "policy"}
POLICY_FIELDS = {"action_costs", "abstain_below"}
SURROGATE = re.compile("[\ud800-\udfff]")  # a lone UTF-16 surrogate: no tokenizer reads it
MAX_MEDIA = 8  # of each kind, as the API takes images
MEDIA_IDS = (1, 64)
# the media types a request's field takes, as the API does (None: any of the kind, read from the bytes)
TYPES: dict[MediaField, tuple[str, ...] | None] = {
    "images": ("image/jpeg", "image/png", "image/webp"),
    "audio": None,
    "videos": None,
}
REQUEST_FIELDS = {
    "context",
    "decisions",
    "model",
    "temperature",
    "images",
    "videos",
    "audio",
    "fallback",
    "wait_for_warm",
}


def refused(path: str, message: str) -> InvalidRequest:
    return InvalidRequest(f"{path}: {message}")


def text(value: Any) -> str:
    """A value as the model reads it: a string as it is, anything else as its JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def check_json(value: Any, path: str, depth: int = 0) -> None:
    """Refuse what JSON can't carry, nesting past MAX_JSON_DEPTH, and text holding a lone surrogate."""
    if depth > MAX_JSON_DEPTH:
        raise refused(path, f"JSON nesting exceeds {MAX_JSON_DEPTH} levels")
    if isinstance(value, str):
        if SURROGATE.search(value):
            raise refused(path, "text must not hold a lone UTF-16 surrogate")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise refused(path, "JSON numbers must be finite")
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise refused(path, "JSON object keys must be strings")
            check_json(key, path, depth + 1)
            check_json(child, path, depth + 1)
    elif isinstance(value, list):
        for child in value:
            check_json(child, path, depth + 1)
    elif value is not None and not isinstance(value, (bool, int)):
        raise refused(path, "only JSON values are accepted")


def check_text(value: Any, path: str, *, empty: bool = True) -> None:
    """A context or a question: a string, an object or an array, as JSON carries them."""
    if not isinstance(value, (str, dict, list)):
        raise refused(path, "must be a string, an object or an array")
    check_json(value, path)
    if not empty and isinstance(value, str) and not value.strip():
        raise refused(path, "must not be empty")


@dataclass(frozen=True)
class Decision:
    """One decision as the API takes it: its outcomes' names, in order, and their descriptions."""

    id: str
    kind: str  # single, binary or ordinal
    question: Any
    names: tuple[str, ...]
    notes: tuple[Any, ...]  # each outcome's description, or None
    policy: dict[str, Any] | None

    def as_systemone(self) -> dict[str, Any]:
        """The question ippon asks for it: a binary decision a noul whose criteria describe yes and no, the others a
        choice among their outcomes in order."""
        if self.kind == "binary":
            criteria = {k: n for k, n in zip(("true", "false"), self.notes, strict=True) if n is not None}
            return {"type": "noul", "instructions": self.question, **({"criteria": criteria} if criteria else {})}
        return {
            "type": "choice",
            "instructions": self.question,
            "criteria": dict(zip(self.names, self.notes, strict=True)),
        }


def binary(outcomes: Any, rubric: Any, path: str) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    if outcomes is not None and (not isinstance(outcomes, dict) or not set(outcomes) <= {"yes", "no"}):
        raise refused(f"{path}.outcomes", 'binary outcomes, when given, are {"yes": description, "no": description}')
    if rubric is not None:
        raise refused(f"{path}.rubric", "rubric is for ordinal decisions; describe binary outcomes in outcomes")
    found = outcomes or {}
    return ("yes", "no"), (found.get("yes"), found.get("no"))


def single(outcomes: Any, rubric: Any, path: str) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    if not isinstance(outcomes, (list, dict)) or not 2 <= len(outcomes) <= MAX_OUTCOMES:
        raise refused(f"{path}.outcomes", f"single outcomes must list 2 to {MAX_OUTCOMES} outcomes")
    if isinstance(outcomes, list) and len(set(map(str, outcomes))) != len(outcomes):
        raise refused(f"{path}.outcomes", "outcomes must be distinct")
    if rubric is not None:
        raise refused(f"{path}.rubric", "rubric is for ordinal decisions; describe single outcomes in outcomes")
    if isinstance(outcomes, dict):
        return tuple(map(str, outcomes)), tuple(outcomes.values())
    return tuple(map(text, outcomes)), (None,) * len(outcomes)


def ordinal(outcomes: Any, rubric: Any, path: str) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    if not isinstance(outcomes, list) or not 2 <= len(outcomes) <= MAX_OUTCOMES:
        raise refused(f"{path}.outcomes", "ordinal outcomes must be an ordered list of at least 2 levels")
    if not isinstance(rubric, list) or len(rubric) != len(outcomes):
        raise refused(f"{path}.rubric", "an ordinal decision needs a rubric: one description per level, in order")
    check_json(rubric, f"{path}.rubric")
    return tuple(map(text, outcomes)), tuple(rubric)


SHAPES = {"single": single, "binary": binary, "ordinal": ordinal}


def cost(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_policy(policy: Any, names: tuple[str, ...], path: str) -> dict[str, Any] | None:
    """action_costs giving every action a cost for every outcome, abstain_below from 0 to 1, or both."""
    if policy is None:
        return None
    check_json(policy, path)
    if not isinstance(policy, dict) or set(policy) - POLICY_FIELDS or all(v is None for v in policy.values()):
        raise refused(path, "a policy takes action_costs, abstain_below or both")
    below, costs = policy.get("abstain_below"), policy.get("action_costs")
    if below is not None and (isinstance(below, bool) or not isinstance(below, (int, float)) or not 0 <= below <= 1):
        raise refused(f"{path}.abstain_below", "must be a number from 0 to 1")
    if costs is None:
        return policy
    if not isinstance(costs, dict) or not 1 <= len(costs) <= MAX_OUTCOMES or any(not a.strip() for a in costs):
        raise refused(f"{path}.action_costs", f"action_costs names 1 to {MAX_OUTCOMES} nonempty actions")
    if ABSTAIN in costs:
        raise refused(f"{path}.action_costs", f'"{ABSTAIN}" is reserved for the abstain policy')
    for action, row in costs.items():
        if not isinstance(row, dict) or set(row) != set(names) or not all(cost(x) for x in row.values()):
            raise refused(f"{path}.action_costs", f"{action!r} must give a finite cost for every outcome")
    return policy


def decision(raw: Any, index: int) -> Decision:
    path = f"decisions[{index}]"
    if not isinstance(raw, dict):
        raise refused(path, "a decision is an object")
    if extra := sorted(set(raw) - FIELDS):
        raise refused(f"{path}.{extra[0]}", "extra inputs are not permitted")
    did, kind = raw.get("id"), raw.get("kind", "single")
    if not isinstance(did, str) or not 1 <= len(did) <= MAX_ID or not did.strip():
        raise refused(f"{path}.id", f"an id is 1 to {MAX_ID} characters, not all spaces")
    check_text(raw.get("question"), f"{path}.question", empty=False)
    if not isinstance(kind, str) or kind not in SHAPES:
        raise refused(f"{path}.kind", "must be single, binary or ordinal")
    if raw.get("outcomes") is not None:
        check_json(raw["outcomes"], f"{path}.outcomes")
    names, notes = SHAPES[kind](raw.get("outcomes"), raw.get("rubric"), path)
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise refused(f"{path}.outcomes", "outcome names must be distinct and nonempty")
    if ABSTAIN in names:
        raise refused(f"{path}.outcomes", f'"{ABSTAIN}" is reserved for the abstain policy')
    return Decision(did, kind, raw["question"], names, notes, check_policy(raw.get("policy"), names, f"{path}.policy"))


def check(body: Any) -> tuple[Any, list[Decision], float | None]:
    """A request's context, decisions and temperature, refused as the API refuses the request."""
    if not isinstance(body, dict):
        raise refused("body", "a decide request is a JSON object")
    if extra := sorted(set(body) - REQUEST_FIELDS):
        raise refused(extra[0], "extra inputs are not permitted")
    context, decisions, temperature = body.get("context"), body.get("decisions"), body.get("temperature")
    check_text(context, "context")
    # taken as the API takes them, and nothing to do here: no hosted model to fall back to, none to wait for
    fallback, warm = body.get("fallback"), body.get("wait_for_warm")
    if not (
        fallback is None
        or isinstance(fallback, str)
        or (isinstance(fallback, list) and all(isinstance(f, str) for f in fallback))
    ):
        raise refused("fallback", "a model name, a list of them, or null")
    if warm is not None and (isinstance(warm, bool) or not isinstance(warm, (int, float)) or not 0 < warm < math.inf):
        raise refused("wait_for_warm", "seconds, a number above 0, or null")
    if not isinstance(decisions, list) or not 1 <= len(decisions) <= MAX_DECISIONS:
        raise refused("decisions", f"a request asks 1 to {MAX_DECISIONS} decisions")
    found = [decision(raw, index) for index, raw in enumerate(decisions)]
    if len({d.id for d in found}) != len(found):
        raise refused("decisions", "decision ids must be unique")
    if len({d.names for d in found if d.kind == "single" and any(len(n) > LONG_OUTCOME for n in d.names)}) > 1:
        raise refused(
            "decisions",
            f"decisions with outcome names over {LONG_OUTCOME} characters must all list the same outcomes in the "
            "same order; send the others in requests of their own",
        )
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not TEMPERATURES[0] <= temperature <= TEMPERATURES[1]
    ):
        raise refused("temperature", f"must be a number from {TEMPERATURES[0]} to {TEMPERATURES[1]}")
    placed(body, context)
    return context, found, temperature


def placed(body: dict[str, Any], context: Any) -> tuple[Any, dict[str, list[str]], dict[str, dict[int, int]]]:
    """The context with each kind's anchors numbered in the order its media first appear (a later mention in words),
    the request's images, videos and audio clips as data: URIs in that order, by field (none: the context as sent),
    and each kind's ids by their new numbers, by anchor word; refused as the API refuses images."""
    given = {m: body.get(m.field) for m in MEDIA if body.get(m.field) not in (None, [])}
    if not given:
        return context, {}, {}
    if not isinstance(context, str):
        raise refused("context", "a context with media is a string that places each with <image:N>, <video:N>, ...")
    out: dict[str, list[str]] = {}
    places: dict[str, dict[int, int]] = {}
    for medium, items in given.items():
        context, out[medium.field], places[medium.anchor] = _place(
            medium.field, medium.anchor, TYPES[medium.field], items, context
        )
    return context, out, places


def renamed(value: Any, places: dict[str, dict[int, int]]) -> Any:
    """A question's text with each <image:id> it names written as the context's later mentions are ("image 2", by
    the number the context's anchor now has): a question about <image:9> asks about that image."""
    if not isinstance(value, str):
        return value
    for word, order in places.items():

        def name(match: re.Match[str], word: str = word, order: dict[int, int] = order) -> str:
            return f"{word} {order.get(int(match[1]), match[1])}"

        value = re.sub(rf"<{word}:(\d+)>", name, value)
    return value


def _place(
    field: str, word: str, types: tuple[str, ...] | None, items: Any, context: str
) -> tuple[str, list[str], dict[int, int]]:
    """One kind's media checked and placed: the context with its anchors renumbered, its media in that order."""
    if not isinstance(items, list) or len(items) > MAX_MEDIA:
        raise refused(field, f"a list of 1 to {MAX_MEDIA} items")
    by_id: dict[int, str] = {}
    for i, item in enumerate(items):
        where = f"{field}[{i}]"
        if not isinstance(item, dict) or set(item) != {"id", "media_type", "data"}:
            raise refused(where, f"an item is {{id, media_type, data}}, placed by <{word}:id>")
        n, kind, data = item["id"], item["media_type"], item["data"]
        if isinstance(n, bool) or not isinstance(n, int) or not MEDIA_IDS[0] <= n <= MEDIA_IDS[1]:
            raise refused(f"{where}.id", f"a whole number from {MEDIA_IDS[0]} to {MEDIA_IDS[1]}")
        if n in by_id:
            raise refused(f"{where}.id", f"{field} ids must be unique; {n} is used twice")
        if not isinstance(kind, str) or (kind not in types if types else not kind.startswith(word + "/")):
            raise refused(f"{where}.media_type", f"one of {', '.join(types)}" if types else f"a {word}/... type")
        if not isinstance(data, str) or not data:
            raise refused(f"{where}.data", "the file in base64")
        by_id[n] = f"data:{kind};base64,{data}"
    anchor = re.compile(rf"<{word}:(\d+)>")
    order: dict[int, int] = {}  # an anchor's id -> its medium's place, 1-based, in the order anchors first appear
    for match in anchor.finditer(context):
        order.setdefault(int(match[1]), len(order) + 1)
    if set(order) != set(by_id):
        missing, unplaced = sorted(set(by_id) - set(order)), sorted(set(order) - set(by_id))
        why = f"no <{word}:{missing[0]}> places it" if missing else f"<{word}:{unplaced[0]}> has no {word}"
        raise refused(field, f"every item needs its anchor and every anchor its item: {why}")
    seen: set[int] = set()

    def numbered(match: re.Match[str]) -> str:
        place = order[int(match[1])]
        first = place not in seen
        seen.add(place)
        return f"<{word}:{place}>" if first else f"{word} {place}"

    return anchor.sub(numbered, context), [by_id[n] for n in order], order


def act(policy: dict[str, Any], names: tuple[str, ...], p: list[float], top: int) -> dict[str, Any]:
    """The cheapest action with every action's expected cost, or the top outcome when the policy prices none;
    abstain when p_top is below abstain_below."""
    result: dict[str, Any] = {"action": names[top]}
    if costs := policy.get("action_costs"):
        expected = {
            action: sum(x * row[name] for x, name in zip(p, names, strict=True)) for action, row in costs.items()
        }
        result = {"action": min(expected, key=expected.__getitem__), "expected_costs": expected}
    if policy.get("abstain_below") is not None and p[top] < policy["abstain_below"]:
        result["action"] = ABSTAIN
    return result


def answer(d: Decision, p: list[float]) -> dict[str, Any]:
    top = max(range(len(p)), key=p.__getitem__)
    out: dict[str, Any] = {
        "kind": d.kind,
        "probabilities": dict(zip(d.names, p, strict=True)),
        "top": d.names[top],
        "p_top": p[top],
    }
    if d.kind == "ordinal":
        out["expected_level"] = sum(i * x for i, x in enumerate(p))
    if d.policy:
        out["policy"] = act(d.policy, d.names, p, top)
    return out


def to_systemone(body: Any) -> tuple[dict[str, Any], list[Decision], float | None]:
    """A decide request checked, as the System One request ippon answers it with (its `model` kept, to pick a served
    model), with its decisions and temperature."""
    context, decisions, temperature = check(body)
    state, media, places = placed(body, context)
    if places:
        decisions = [replace(d, question=renamed(d.question, places)) for d in decisions]
    systemone: dict[str, Any] = {"state": state, "questions": {d.id: d.as_systemone() for d in decisions}, **media}
    if isinstance(body.get("model"), str):
        systemone["model"] = body["model"]
    return systemone, decisions, temperature


def response(result: Result, decisions: list[Decision], temperature: float | None, model: str) -> dict[str, Any]:
    """The decide API's response from ippon's answer to the request `to_systemone` made, with the milliseconds its
    rendering and scoring took (`Result.timing_ms`), as the API and the SDK's own local engine give them."""
    at = 1.0 if temperature is None else float(temperature)
    answers = {}
    for d in decisions:
        # in outcome order (a binary's yes, then no)
        answers[d.id] = answer(d, tempered(result.answers[d.id].probabilities, at))
    out: dict[str, Any] = {
        "id": "local_" + secrets.token_hex(12),
        "model": model,
        "decisions": answers,
        "usage": {
            "prompt_tokens": result.response.get("usage", {}).get("input_tokens", 0),
            "decisions": len(decisions),
        },
        "timing_ms": dict(result.timing_ms),
    }
    if at != 1:
        out["temperature"] = {d.id: temperature for d in decisions}
    return out | carried(result.response)
