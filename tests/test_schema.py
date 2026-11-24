from __future__ import annotations

import base64
import hashlib
from typing import Any

import pytest

from helpers import b64
from ippon import InvalidRequest, parse_request
from ippon.errors import NonFiniteAnswer
from ippon.lint import lint
from ippon.questions import with_media
from ippon.schema import Answer, check_anchors

Q = {"q": {"type": "noul"}}
WAV = b"RIFF" + b"\0" * 4 + b"WAVEfmt " + b"\0" * 24  # a WAV clip's first bytes: its format is named from them


def body(**questions: Any) -> dict[str, Any]:
    return {"state": "A short ticket.", "questions": questions}


def png(n: int) -> bytes:
    """A PNG's signature, then bytes that make image n differ from the others."""
    return bytes.fromhex("89504e470d0a1a0a") + bytes([n]) * 24


def uri(n: int) -> str:
    return "data:image/png;base64," + b64(png(n))


def part(n: int) -> dict[str, Any]:
    """Image n as Perplexity's Decisions API takes it: an OpenAI-style part."""
    return {"type": "image_url", "image_url": {"url": uri(n)}}


def test_each_question_type_is_parsed_in_the_callers_order() -> None:
    """Questions keep the caller's order and criteria; a choice's criteria may be a list of labels, and instructions,
    descriptions and a score's levels any JSON value; the state may be missing, an object or a list."""
    request = parse_request(
        body(
            urgent={
                "type": "noul",
                "instructions": "Is it urgent?",
                "criteria": {"false": "It can wait", "true": None},
            },
            team={"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "sales": None}},
            mood={"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
            listed={"type": "choice", "criteria": ["billing", "sales"]},
            valued={
                "type": "choice",
                "instructions": {"ask": "which?"},
                "criteria": {"a": {"means": "first"}, "b": None},
            },
            levels={"type": "score", "criteria": ["Can wait", {"level": "today", "note": "hours"}, "Now"]},
        )
    )
    urgent, team, mood, listed, valued, levels = request.questions
    assert [q.id for q in request.questions] == ["urgent", "team", "mood", "listed", "valued", "levels"]
    assert urgent.options == ("true", "false")
    assert urgent.criteria == (("false", "It can wait"), ("true", None))
    assert team.options == ("billing", "sales")
    assert team.description("billing") == "Payments"
    assert team.description("sales") is None
    assert mood.options == ("0", "1", "2")
    assert mood.description("2") == "furious"
    assert listed.criteria == (("billing", None), ("sales", None))
    assert listed.instructions is None
    assert valued.instructions == {"ask": "which?"}
    assert valued.description("a") == {"means": "first"}
    assert levels.options == ("0", "1", "2")
    assert levels.description("1") == {"level": "today", "note": "hours"}
    q = {"q": {"type": "noul"}}
    assert parse_request({"questions": q}).state is None
    assert parse_request({"state": {"a": 1}, "questions": q}).state == {"a": 1}
    assert parse_request({"state": ["x"], "questions": q}).state == ["x"]


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ([], "JSON object"),
        ({"questions": {}}, "non-empty"),
        ({"questions": {"q": {"type": "noul"}}, "videos": "aGk="}, "list of base64"),
        ({"questions": {"q": {"type": "noul"}}, "videos": ["iVBORw0KGgo="]}, r"videos\[0\]: a PNG file, not a video"),
        ({"questions": {"q": {"type": "noul"}}, "images": "aGk="}, "list of base64"),
        ({"questions": {"q": {"type": "noul"}}, "images": ["https://example.com/a.png"]}, r"images\[0\]: URLs"),
        ({"questions": {"q": {"type": "noul"}}, "temperature": 1}, "unknown field"),
        ({"state": 3, "questions": {"q": {"type": "noul"}}}, "'state'"),
        ({"questions": {"q": {"type": "maybe"}}}, "'type'"),
        ({"questions": {"q": {"type": "noul", "weight": 2}}}, "unknown field"),
        ({"questions": {"q": {"type": "noul", "question": "Is it?"}}}, "put that in 'instructions'"),
        ({"messages": [], "questions": {"q": {"type": "noul"}}}, "put that in 'state'"),
        (
            {"questions": {"q": {"type": "noul"}}, "images": ["AAAAIGZ0eXBoZWljAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"]},
            "a HEIC photo",
        ),
        ({"questions": {"q": {"type": "noul", "criteria": {"yes": "y"}}}}, "noul's criteria"),
        ({"questions": {"q": {"type": "choice"}}}, "needs criteria"),
        ({"questions": {"q": {"type": "choice", "criteria": {}}}}, "at least 1 option"),
        ({"questions": {"q": {"type": "choice", "criteria": ["a", "a"]}}}, "distinct"),
        ({"questions": {"q": {"type": "choice", "criteria": {"": "empty"}}}}, "non-empty"),
        ({"questions": {"q": {"type": "score", "criteria": ["only one"]}}}, "2 levels or more"),
        ({"questions": {"q": {"type": "score", "criteria": {"low": "l"}}}}, "needs criteria"),
        ({"questions": {"q": {"type": "score", "criteria": ["low", None]}}}, "needs criteria"),
        ({"state": [part(1)], "images": [uri(2)], "questions": Q}, "'images': the state holds image parts too"),
        ({"state": {"photo": part(1)}, "files": [uri(2)], "questions": Q}, "'files': the state holds image parts"),
        ({"state": {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}, "questions": Q}, "URLs"),
        ({"questions": {"q": {"type": "noul", "instructions": ["Is it this?", part(1)]}}}, "q: images go in the state"),
        ({"questions": {"q": {"type": "choice", "criteria": {"this": part(1), "no": None}}}}, "images go in the state"),
        ({"questions": Q, "files": "aGk="}, "'files' must be a list"),
        ({"questions": Q, "unsure_below": 0}, "'unsure_below' must be a number above 0 and at most 1"),
        ({"questions": Q, "unsure_below": True}, "'unsure_below' must be a number"),
        ({"questions": Q, "files": [b64(WAV)], "audio": b64(WAV)}, "send its clips one way"),
        ({"questions": Q, "files": ["data:audio/wav;base64," + b64(png(1))]}, r"files\[0\]: a PNG image, not a clip"),
        ({"questions": {"q": {"type": "noul", "instructions": {"type": "input_audio"}}}}, "a spoken question is"),
        (
            {
                "questions": {
                    "q": {
                        "type": "noul",
                        "instructions": {"type": "input_audio", "input_audio": {"data": b64(WAV), "format": "flac"}},
                    }
                }
            },
            "format wav or mp3",
        ),
        (
            {
                "questions": {
                    "q": {"type": "noul", "instructions": {"type": "input_audio", "input_audio": {"data": b64(png(1))}}}
                }
            },
            "q: instructions: a PNG image, not a clip",
        ),
    ],
)
def test_invalid_requests_say_what_is_wrong(bad: Any, message: str) -> None:
    with pytest.raises(InvalidRequest, match=message):
        parse_request(bad)


def test_answers_take_the_systemone_shape() -> None:
    request = parse_request(
        body(
            urgent={"type": "noul"},
            team={"type": "choice", "criteria": ["billing", "sales"]},
            mood={"type": "score", "criteria": ["calm", "annoyed", "furious"]},
        )
    )
    urgent, team, mood = request.questions
    answers = [Answer(urgent, (0.9, 0.1)), Answer(team, (0.25, 0.75)), Answer(mood, (0.2, 0.3, 0.5))]
    out = {"answers": {a.question.id: a.to_systemone() for a in answers}}
    assert out["answers"]["urgent"] == {"type": "noul", "noul": 0.9}
    assert out["answers"]["team"] == {
        "type": "choice",
        "choice": "sales",
        "confidence": 0.75,
        "probabilities": {"billing": 0.25, "sales": 0.75},
    }
    score = out["answers"]["mood"]
    assert score["score"] == pytest.approx(1.3)
    assert score["confidence"] == 0.5
    assert score["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}


@pytest.mark.parametrize(
    ("probabilities", "error", "message"),
    [
        ((0.5,), ValueError, "probabilities for"),
        ((0.7, 0.7), ValueError, "probabilities sum"),
        ((1.1, -0.1), ValueError, "non-negative"),
        ((float("nan"), 1.0), NonFiniteAnswer, "finite"),
        ((float("inf"), -float("inf")), NonFiniteAnswer, "finite"),
    ],
)
def test_wrong_probabilities_are_raised_not_returned(
    probabilities: tuple[float, ...], error: type[ValueError], message: str
) -> None:
    (q,) = parse_request(body(q={"type": "noul"})).questions
    with pytest.raises(error, match=message):
        Answer(q, probabilities)


def test_images_and_audio_arrive_as_checked_media_in_order() -> None:
    """`images` and `audio` as Ollama sends them (base64) or llama.cpp's server takes them (data URIs), and `files`,
    whose clip is told apart from its images by its data URI's type or, in plain base64, by its bytes, and whose
    video by its data URI's type."""
    request = parse_request({"questions": Q, "images": [b64(png(1)), uri(1)], "audio": b64(WAV)})
    assert [m.format for m in request.images] == ["png", "png"]
    assert request.images[0].sha256 == request.images[1].sha256  # the same bytes, as Ollama or as a data URI
    assert [m.format for m in request.audio] == ["wav"]
    assert parse_request({"questions": Q}).images == ()
    for clip in (b64(WAV), "data:audio/wav;base64," + b64(WAV)):
        files = parse_request({"questions": Q, "files": [uri(1), clip, b64(png(1)), clip]})
        assert files.audio == request.audio * 2  # every clip, in order
        assert parse_request({"questions": Q, "audio": [clip, clip]}).audio == files.audio
        assert files.images == request.images
    mp4 = b"\0\0\0\x18ftypisom" + b"\0" * 12  # an MP4 file's first bytes
    films = parse_request(
        {"questions": Q, "files": ["data:video/mp4;base64," + b64(mp4), b64(mp4)], "videos": [b64(mp4)]}
    )
    assert [m.kind for m in films.videos] == ["video"] * 3  # a video in `files`, as a data URI or plain, then `videos`
    assert films.images == films.audio == ()


@pytest.mark.parametrize(
    ("fields", "state", "images"),
    [
        # Perplexity's Decisions API: image parts anywhere in the state, read in the order they are written, each
        # taken out whole (an item from its list, a value with its key from its object)
        ({"state": ["Which color is the square?", part(1)]}, ["Which color is the square?"], [1]),
        ({"state": {"caption": "From the listing", "photo": part(1)}}, {"caption": "From the listing"}, [1]),
        ({"state": part(1)}, "", [1]),
        (
            {"state": {"a": [part(1), {"b": part(2), "c": 3}], "d": part(3), "e": "x"}},
            {"a": [{"c": 3}], "e": "x"},
            [1, 2, 3],
        ),
        ({"state": [part(n) for n in range(8)]}, [], list(range(8))),
        # llama.cpp's server: the image parts of chat messages, their text parts kept; `files` first, then `images`
        (
            {"state": [{"role": "user", "content": [part(1), {"type": "text", "text": "Hi"}]}]},
            [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            [1],
        ),
        ({"state": "s", "files": [uri(1), b64(png(2))], "images": [uri(3), b64(png(4))]}, "s", [1, 2, 3, 4]),
    ],
)
def test_images_come_in_the_order_each_server_reads_them_and_leave_the_state(
    fields: dict[str, Any], state: Any, images: list[int]
) -> None:
    request = parse_request({**fields, "questions": Q})
    assert request.state == state
    assert [m.sha256 for m in request.images] == [hashlib.sha256(png(n)).hexdigest() for n in images]


def test_a_state_may_place_only_media_the_request_holds() -> None:
    """<image:2> with one image would be answered about nothing: lint and the decider refuse it. With no image sent at
    all, the tag is the state's own text (a ticket or a page may hold one): answered, and lint asks whether the image
    was forgotten. A Python caller's image, added after the body is read (questions.with_media), counts."""
    body = {"state": "The photo <image:1>.", "questions": {"q": {"type": "noul", "instructions": "Is it blue?"}}}
    assert "sends no images" in lint(body)[0].message
    check_anchors(parse_request(body))  # text, never refused
    twice = {**body, "state": "<image:1> and <image:2>."}
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    check_anchors(with_media(parse_request(body), [png], None))
    # Non-anchor formats (anchors=frozenset()) treat literal '<image:2>' in a ticket as ordinary text.
    check_anchors(with_media(parse_request(twice), [png], None), frozenset())
    with pytest.raises(InvalidRequest, match="'images' holds 1"):
        check_anchors(with_media(parse_request(twice), [png], None))
