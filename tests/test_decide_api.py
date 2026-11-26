"""Sqwish's decide API on ippon: its answers follow the API's arithmetic (temperature, expected level, a policy's
cheapest action and abstention) and its requests are refused as the API refuses them."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from ippon import decide_api
from ippon.errors import InvalidRequest
from ippon.schema import Answer, parse_request

BODY: dict[str, Any] = {
    "context": "Charged twice.",
    "decisions": [
        {"id": "refund", "question": "Refund?", "kind": "binary"},
        {
            "id": "team",
            "question": "Which team?",
            "outcomes": ["billing", "engineering"],
            "policy": {
                "action_costs": {"route": {"billing": 0, "engineering": 5}, "ask": {"billing": 2, "engineering": 2}}
            },
        },
        {
            "id": "urgency",
            "question": "How urgent?",
            "kind": "ordinal",
            "outcomes": ["low", "mid", "high"],
            "rubric": ["later", "soon", "now"],
            "policy": {"abstain_below": 0.9},
        },
    ],
    "temperature": 2.0,
}


def test_answers_follow_the_apis_arithmetic() -> None:
    systemone, decisions, temperature = decide_api.to_systemone(BODY)
    request = parse_request(systemone)
    p = {"refund": (0.8, 0.2), "team": (0.6, 0.4), "urgency": (0.1, 0.3, 0.6)}
    result = SimpleNamespace(
        answers={q.id: Answer(q, p[q.id]) for q in request.questions},
        response={"usage": {"input_tokens": 42}},
        timing_ms={},
    )
    out = decide_api.response(result, decisions, temperature, "kettei-dot")  # type: ignore[arg-type]

    def tempered(xs: tuple[float, ...]) -> list[float]:  # log space, divided by 2, renormalised
        w = [math.exp(math.log(x) / 2) for x in xs]
        return [v / sum(w) for v in w]

    refund, team, urgency = (out["decisions"][k] for k in ("refund", "team", "urgency"))
    assert refund["probabilities"] == pytest.approx(dict(zip(("yes", "no"), tempered(p["refund"]), strict=True)))
    assert refund["top"] == "yes"
    t = tempered(p["team"])
    assert team["policy"]["expected_costs"] == pytest.approx({"route": 5 * t[1], "ask": 2.0})
    assert team["policy"]["action"] == min(team["policy"]["expected_costs"], key=team["policy"]["expected_costs"].get)
    u = tempered(p["urgency"])
    assert urgency["expected_level"] == pytest.approx(u[1] + 2 * u[2])
    assert urgency["top"] == "high"
    assert urgency["policy"]["action"] == "abstain"  # p_top under 0.9
    assert out["temperature"] == {"refund": 2.0, "team": 2.0, "urgency": 2.0}
    assert out["usage"] == {"prompt_tokens": 42, "decisions": 3}
    # costs each a float can hold, whose expected cost isn't: refused, as the SDK refuses it, never an inf answered
    huge = {"id": "h", "question": "?", "outcomes": ["a", "b"],
            "policy": {"action_costs": {"go": {"a": sys.float_info.max, "b": sys.float_info.max}}}}  # fmt: skip
    systemone, decisions, _ = decide_api.to_systemone({"context": "x", "decisions": [huge]})
    (question,) = parse_request(systemone).questions
    result = SimpleNamespace(answers={"h": Answer(question, (0.6, 0.4000001))}, response={}, timing_ms={})
    with pytest.raises(InvalidRequest, match="expected costs must be finite"):
        decide_api.response(result, decisions, None, "kettei-dot")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"decisions": []}, "1 to 128 decisions"),
        ({"temperature": 9}, "from 0.2 to 5.0"),
        ({"decisions": [{"id": "x", "question": "?", "outcomes": ["a"]}]}, "2 to 255 outcomes"),
        ({"decisions": [{"id": "x", "question": "?", "kind": "ordinal", "outcomes": ["a", "b"]}]}, "needs a rubric"),
        ({"decisions": [{"id": "x", "question": "?", "outcomes": ["abstain", "go"]}]}, "reserved"),
        ({"decisions": [{"id": "x", "question": " ", "kind": "binary"}]}, "must not be empty"),
        ({"decisions": [{"id": "x\ud800", "question": "?", "kind": "binary"}]}, r"decisions\[0\]\.id: .*surrogate"),
        ({"surprise": 1}, "extra inputs"),
        ({"images": [{"id": 1, "media_type": "image/png", "data": "AAAA"}]}, "no <image:1> places it"),
        (
            {"context": "<image:1>, <image:2>", "images": [{"id": 1, "media_type": "image/png", "data": "AAAA"}]},
            "<image:2> has no image",
        ),
        ({"context": ["<image:1>"], "images": [{"id": 1, "media_type": "image/png", "data": "AAAA"}]}, "is a string"),
        ({"context": "<image:1>", "images": [{"id": 1, "media_type": "image/gif", "data": "AAAA"}]}, "image/jpeg"),
        (
            {"context": "x", "images": [{"id": i, "media_type": "image/png", "data": "A"} for i in range(1, 10)]},
            "1 to 8 items",
        ),
    ],
)
def test_requests_are_refused_as_the_api_refuses_them(change: dict[str, Any], match: str) -> None:
    with pytest.raises(InvalidRequest, match=match):
        decide_api.check({**BODY, **change})


def test_media_keep_their_anchors_in_the_order_they_first_appear() -> None:
    images = [{"id": n, "media_type": "image/png", "data": f"P{n}"} for n in (1, 2)]
    pictures = {
        **BODY,
        "context": "Damage report <image:2>. Front photo <image:1> and again <image:2>.",
        "images": images,
    }
    systemone, _, _ = decide_api.to_systemone(pictures)
    assert systemone["state"] == "Damage report <image:1>. Front photo <image:2> and again image 1."
    asked = {**pictures, "decisions": [{"id": "d", "question": "Is <image:1> blurry?", "kind": "binary"}]}
    assert decide_api.to_systemone(asked)[0]["questions"]["d"]["instructions"] == "Is image 2 blurry?"  # its photo
    assert systemone["images"] == ["data:image/png;base64,P2", "data:image/png;base64,P1"]
    body = {
        **BODY,
        "context": "Call <audio:4> about the clip <video:9>, then <audio:3>.",
        "videos": [{"id": 9, "media_type": "video/mp4", "data": "V"}],
        "audio": [
            {"id": 3, "media_type": "audio/wav", "data": "A3"},
            {"id": 4, "media_type": "audio/mpeg", "data": "A4"},
        ],
    }
    systemone, _, _ = decide_api.to_systemone(body)
    assert systemone["state"] == "Call <audio:1> about the clip <video:1>, then <audio:2>."
    assert systemone["videos"] == ["data:video/mp4;base64,V"]
    assert systemone["audio"] == ["data:audio/mpeg;base64,A4", "data:audio/wav;base64,A3"]
    with pytest.raises(InvalidRequest, match=r"a video/\.\.\. type"):
        decide_api.check({**body, "videos": [{"id": 9, "media_type": "audio/wav", "data": "V"}]})
