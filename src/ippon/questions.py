"""Questions written in Python instead of System One JSON.

    decider.decide(state=ticket, questions={
        "refund": YesNo("Is a refund asked for?"),
        "team": Choice("Which team should handle it?", ["billing", "sales", "support"]),
        "mood": Score("How upset is the customer?", ["calm", "annoyed", "furious"]),
    })

Each builds the same System One question a JSON request would carry, so both forms give the same answers, and a
Python caller's images, audio clips and videos (bytes, paths, arrays) join a parsed request with `with_media`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import InvalidRequest
from .media.decode import as_media
from .schema import Request


@dataclass(frozen=True)
class YesNo:
    """A yes/no question (System One "noul"), with optional descriptions of what yes and no mean."""

    instructions: str | None = None
    yes: Any = None
    no: Any = None

    def to_systemone(self) -> dict[str, Any]:
        body: dict[str, Any] = {"type": "noul"}
        if self.instructions is not None:
            body["instructions"] = self.instructions
        criteria = {k: v for k, v in (("true", self.yes), ("false", self.no)) if v is not None}
        if criteria:
            body["criteria"] = criteria
        return body


@dataclass(frozen=True)
class Choice:
    """One of several options: a list of labels, or labels with descriptions."""

    instructions: str
    options: Sequence[str] | Mapping[str, Any]

    def to_systemone(self) -> dict[str, Any]:
        if not isinstance(self.options, (Mapping, Sequence)) or isinstance(self.options, (str, bytes, bytearray)):
            raise InvalidRequest('Choice options must be a list of labels, like ["billing", "sales"], or a mapping')
        criteria = dict(self.options) if isinstance(self.options, Mapping) else list(self.options)
        return {"type": "choice", "instructions": self.instructions, "criteria": criteria}


@dataclass(frozen=True)
class Score:
    """A level on an ordered scale, lowest first."""

    instructions: str
    levels: Sequence[Any]

    def to_systemone(self) -> dict[str, Any]:
        if not isinstance(self.levels, Sequence) or isinstance(self.levels, (str, bytes, bytearray)):
            raise InvalidRequest('Score levels must be an ordered list, like ["low", "high"], lowest first')
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}


Builder = YesNo | Choice | Score


def request_body(
    state: Any, questions: Mapping[str, Builder | Mapping[str, Any]] | None, model: str | None = None
) -> dict[str, Any]:
    """A System One request body from a state and questions (builders or System One dicts)."""
    if not isinstance(questions, Mapping) or not questions:
        raise InvalidRequest(
            'questions must be a non-empty mapping of named questions, like {"refund": YesNo("Refund?")}'
        )
    converted: dict[str, Any] = {}
    for question_id, question in questions.items():
        if not isinstance(question, (YesNo, Choice, Score, Mapping)):
            raise InvalidRequest(
                f"{question_id}: a question must be YesNo(...), Choice(...), Score(...) or a System One mapping"
            )
        try:
            converted[question_id] = (
                question.to_systemone() if isinstance(question, (YesNo, Choice, Score)) else question
            )
        except InvalidRequest as error:
            raise InvalidRequest(f"{question_id}: {error}") from None
    body: dict[str, Any] = {"questions": converted}
    if state is not None:
        body["state"] = state
    if model is not None:
        body["model"] = model
    return body


def with_media(request: Request, images: Sequence[Any], audio: Any, videos: Sequence[Any] = ()) -> Request:
    """`request` with a Python caller's images, audio clips and videos (media/decode.as_media) after any its body
    carried."""
    if isinstance(images, (str, bytes, Path)) or hasattr(images, "convert") or hasattr(images, "shape"):
        images = [images]
    if isinstance(videos, (str, bytes, Path)):
        videos = [videos]
    if audio is None:
        audio = []
    elif isinstance(audio, (str, bytes, Path)) or hasattr(audio, "shape"):
        audio = [audio]
    if not images and not audio and not videos:
        return request
    return replace(
        request,
        images=request.images + tuple(as_media(image, "image") for image in images),
        audio=request.audio + tuple(as_media(clip, "audio") for clip in audio),
        videos=request.videos + tuple(as_media(video, "video") for video in videos),
    )
