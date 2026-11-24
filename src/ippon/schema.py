"""System One requests and answers, the schema the open decision models' own servers speak.

A request has a state (text, a JSON object or a JSON list; some models accept none), named questions,
and, for a model that reads them, images, an audio clip and videos (base64 or data: URIs, media/decode.py), sent in
the form of any of the servers ippon stands in for:

    images  a list of images (Ollama's and llama.cpp's servers)
    files   a list of images and audio clips, each told apart by its bytes (llama.cpp's server)
    audio   a clip or a list of them, heard in order (d1-omni's own name)
    videos  a list of videos, read after the images (vllm-sr d3's server)
    state   image parts anywhere in it, {"type": "image_url", "image_url": {"url": "data:image/..."}} (Perplexity's
            Decisions API), taken out of it (_state_images)

Each question is of one of three types:

    noul    a yes/no question; criteria optional: {"true": "...", "false": "..."}
    choice  one of its options, as many as the model takes: {"label": "description" or null, ...}, or a list of labels
    score   a level on a scale of 2 or more levels: ["lowest level", ..., "highest level"]

Instructions and descriptions are usually strings; any JSON value is accepted and left to the model's format,
since some creators' servers accept objects there and render them as JSON. For a model that hears them, a question
may be spoken: its instructions an audio clip, {"type": "input_audio", "input_audio": {"data": <a WAV clip, base64 or
a data: URI>}} (OpenAI's audio part, as llama.cpp's server takes it).

An answer gives a probability to each of a question's options and is returned in the same shape the
creators' servers return: noul -> P(true); choice -> the most likely label, its confidence and every
probability; score -> the creator's score (the expected level, or the median level where the creator's code uses
it, as d1's does), the confidence of the most likely level, the legend and every probability.

parse_request() validates a request body once, at the boundary, and returns typed, immutable values;
everything after it trusts them.

"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from .errors import InvalidRequest, NonFiniteAnswer
from .formulas import Confidence, top_probability
from .media.decode import AUDIO_FORMATS, MAX_IMAGES, VIDEO_FORMATS, Media, MediaKind, parse_media, sniff

QuestionType = Literal["noul", "choice", "score"]
QUESTION_TYPES: tuple[QuestionType, ...] = ("noul", "choice", "score")
MIN_SCORE_LEVELS = 2
NOUL_OPTIONS = ("true", "false")
IMAGE_PART = "image_url"  # the `type` of an OpenAI-style image part
MediaField = Literal["images", "audio", "videos"]


@dataclass(frozen=True)
class Medium:
    """A kind of media a request carries: its field, the word its anchors place it by in a state (<image:1>), which
    is also the kind a `files` entry is told apart as, and how a list of what a model reads names it."""

    field: MediaField
    anchor: MediaKind
    shown: str


# every kind of media, in the order a list of what a model reads shows them; what reads a kind's field, anchor or name
# reads it here
MEDIA = (Medium("images", "image", "images"), Medium("audio", "audio", "audio"), Medium("videos", "video", "video"))
MEDIA_FIELDS: tuple[MediaField, ...] = tuple(m.field for m in MEDIA)
BY_ANCHOR = {m.anchor: m for m in MEDIA}

State = str | Mapping[str, Any] | Sequence[Any] | None

# A question and a request as JSON Schema: every field parse_request takes, with what it holds (the MCP tool's input
# schema; a field added here is one a request may give)
QUESTION: dict[str, Any] = {
    "type": "object",
    "description": "One question. noul: yes or no, answered with P(yes). choice: one of its options, answered with "
    "each option's probability. score: a level on an ordered scale, answered with the creator's score (the expected "
    "level, or the median level where the creator's code uses it) and every level's probability.",
    "properties": {
        "type": {"type": "string", "enum": list(QUESTION_TYPES)},
        "instructions": {
            "type": "string",
            "description": "The question as a full sentence naming what to look at, one thing per question: "
            "'Is the customer asking for money back?'",
        },
        "criteria": {
            "description": 'noul: optional, {"true": what yes means, "false": what no means}. choice: required, '
            '{"label": "what belongs to it" or null, ...} with distinct labels, or a list of labels; add a '
            'catch-all such as "other" when a state may fit none. score: required, a list of 2 or more levels, lowest '
            'first: ["calm", "annoyed", "angry"].',
            "anyOf": [
                {"type": "object", "additionalProperties": {"type": ["string", "null"]}},
                {"type": "array", "items": {"type": "string"}, "minItems": 1},
            ],
        },
    },
    "required": ["type", "instructions"],
    "additionalProperties": False,
}
REQUEST: dict[str, dict[str, Any]] = {
    "model": {
        "type": "string",
        "description": "The model to ask, by the name a server serves it as, a short name, a Hub id or a folder. "
        "Leave it out for the server's first model.",
    },
    "state": {
        "type": ["string", "object", "array", "null"],
        "description": "What to decide about: text, or JSON (a ticket with its fields, a diff with its description, "
        "an agent's trace). Send what the questions need: input past the model's window follows the creator's "
        "limit policy, which may split questions, cut input or reject it. `truncated` lists any omitted input.",
    },
    "questions": {
        "type": "object",
        "description": "The questions, each by an id of your choosing ('refund', 'team'); the response answers each "
        "under its id. Causal models can share the state across questions; bidirectional models read each full prompt.",
        "additionalProperties": QUESTION,
        "minProperties": 1,
    },
    "unsure_below": {
        "type": "number",
        "exclusiveMinimum": 0,
        "maximum": 1,
        "description": "Optional: list, as `unsure`, the questions whose most likely option's probability is under "
        "this. Act on the rest, and send those on to the user or a larger model.",
    },
    "images": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Images as base64 or data:image/...;base64 URIs, for a model that reads them; URLs are never "
        "fetched.",
    },
    "files": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Images, audio clips and videos in one list, as base64 or data: URIs, each told apart by its "
        "bytes, for a model that reads them; URLs are never fetched.",
    },
    "audio": {
        "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
        "description": "An audio clip, or a list of clips in the order they are heard, as base64 or data: URIs, for a "
        "model that hears them.",
    },
    "videos": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Videos as base64 or data:video/...;base64 URIs, read after the images, for a model that reads "
        "them; URLs are never fetched.",
    },
}
STATE_FIELDS = ("state", "files", *MEDIA_FIELDS)  # what the state a request asks about is given in
_REQUEST_FIELDS = frozenset(REQUEST)
_QUESTION_FIELDS = frozenset(QUESTION["properties"])
# the names people reach for first (from chat APIs and classifiers), and the System One field that holds each
MEANT = {
    "messages": "state", "input": "state", "text": "state", "context": "state", "prompt": "instructions",
    "question": "instructions", "description": "instructions", "options": "criteria", "choices": "criteria",
    "labels": "criteria", "classes": "criteria",
}  # fmt: skip


def _unknown(key: str, fields: frozenset[str], where: str) -> str:
    """The message for a field `where` doesn't take: the fields it takes, and the one `key` likely meant."""
    meant = MEANT.get(key)
    hint = f"; put that in {meant!r}" if meant in fields else ""
    return f"unknown field {key!r}{hint} ({where} takes {', '.join(sorted(fields))})"


@dataclass(frozen=True)
class Question:
    """One question of a request.

    `criteria` keeps the caller's (key, description) pairs in the caller's order, since some models
    show them in that order: the labels of a choice, "true"/"false" of a noul (possibly none, or only
    one), and "0".."n-1" for the levels of a score. `options` are the answer's options in the order its
    probabilities come in."""

    id: str
    type: QuestionType
    instructions: Any  # usually a string; None when absent; a Media for a spoken question
    criteria: tuple[tuple[str, Any], ...]

    @property
    def spoken(self) -> Media | None:
        """The clip a spoken question's instructions are, or None for written ones."""
        return self.instructions if isinstance(self.instructions, Media) else None

    @property
    def options(self) -> tuple[str, ...]:
        if self.type == "noul":
            return NOUL_OPTIONS
        return tuple(key for key, _ in self.criteria)

    def description(self, option: str) -> Any:
        """The caller's description of an option (usually a string), or None when it has none."""
        return dict(self.criteria).get(option)


@dataclass(frozen=True)
class Request:
    model: str | None
    state: State
    questions: tuple[Question, ...]
    images: tuple[Media, ...] = ()  # in the order the model reads them, for a model that takes images
    audio: tuple[Media, ...] = ()  # clips, in the order the model hears them, for a model that takes audio
    videos: tuple[Media, ...] = ()  # in the order the model reads them, for a model that takes videos
    unsure_below: float | None = None  # ippon's: list the questions whose confidence is under it (`Answer.confidence`)

    def media(self, field: MediaField) -> tuple[Media, ...]:
        """The request's media of the kind `field` names."""
        match field:
            case "images":
                return self.images
            case "audio":
                return self.audio
            case "videos":
                return self.videos


SURROGATE = re.compile("[\ud800-\udfff]")  # half of a UTF-16 pair, alone: JSON can escape it, UTF-8 can't hold it
# the escape a surrogate in JSON bytes must come as: bytes without one can't hold one, so their text isn't searched (a
# 12 MB body of four photos: 1.8 ms instead of 26)
SURROGATE_ESCAPE = re.compile(rb"\\u[dD][89a-fA-F]")
MAX_DEPTH = 100  # nesting a request may have: far past any real state, well short of what recursive code can take


def _once(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object whose every key is given once: a key given twice would keep only its last value."""
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise InvalidRequest(f"the key {key!r} is given twice in one object")
        found[key] = value
    return found


def read_json(data: bytes | str) -> Any:
    """`data` decoded as JSON from a client (an HTTP body, an MCP message): InvalidRequest, never another error, for
    anything else, including what Python's decoder itself can't take (nesting deep enough to exhaust its recursion,
    integers past its 4,300 digits), and for nesting past MAX_DEPTH, which the renderers after it walk recursively."""
    try:
        value = json.loads(data, object_pairs_hook=_once)
    except RecursionError:
        raise InvalidRequest(f"the JSON is nested deeper than {MAX_DEPTH} levels") from None
    except (ValueError, UnicodeDecodeError) as error:  # JSONDecodeError, and an integer too long to read
        raise InvalidRequest(f"not JSON: {error}") from None
    surrogates = not isinstance(data, bytes) or SURROGATE_ESCAPE.search(data) is not None
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, str):
            if surrogates and SURROGATE.search(item):
                raise InvalidRequest(
                    "text holds a lone UTF-16 surrogate (a \\ud800 to \\udfff escape), which no model reads"
                )
        elif isinstance(item, (dict, list)):
            if depth > MAX_DEPTH:
                raise InvalidRequest(f"the JSON is nested deeper than {MAX_DEPTH} levels")
            if isinstance(item, dict):
                pending.extend((key, depth) for key in item)  # a question's id is a key
            pending.extend((child, depth + 1) for child in (item.values() if isinstance(item, dict) else item))
    return value


def parse_request(body: object) -> Request:
    """A System One request body (decoded JSON) as a Request, or InvalidRequest saying what is wrong."""
    if not isinstance(body, Mapping):
        raise InvalidRequest("a request is a JSON object")
    for key in body:
        if key not in _REQUEST_FIELDS:
            raise InvalidRequest(_unknown(key, _REQUEST_FIELDS, "a request"))
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise InvalidRequest("'model' must be a string")
    state = body.get("state")
    if state is not None and not isinstance(state, (str, Mapping, list)):
        raise InvalidRequest("'state' must be a string, a JSON object or a JSON list")
    questions = body.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise InvalidRequest("'questions' must be a non-empty object of named questions")
    parsed = tuple(_parse_question(qid, q) for qid, q in questions.items())
    state, in_state = _state_images(cast(State, state))
    files, clips, films = _parse_files(body.get("files"))
    images = _parse_media_list(body.get("images"), "image")
    if in_state and (files or images):
        fields = " and ".join(repr(name) for name, got in (("files", files), ("images", images)) if got)
        raise InvalidRequest(
            f"{fields}: the state holds image parts too; send a request's images one way (Perplexity's Decisions "
            "API takes them in the state only)"
        )
    given = body.get("audio")
    audio = (parse_media(given, "audio", "audio"),) if isinstance(given, str) else _parse_media_list(given, "audio")
    if clips and audio:
        raise InvalidRequest("audio: the request's 'files' hold clips too; send its clips one way")
    videos = films + _parse_media_list(body.get("videos"), "video")
    unsure = body.get("unsure_below")
    if unsure is not None and (isinstance(unsure, bool) or not isinstance(unsure, (int, float)) or not 0 < unsure <= 1):
        raise InvalidRequest("'unsure_below' must be a number above 0 and at most 1")
    all_images = in_state or files + images  # llama.cpp's order: `files`, then `images`
    if len(all_images) > MAX_IMAGES:
        raise InvalidRequest(f"images: {len(all_images)} images in one request; at most {MAX_IMAGES}")
    return Request(
        model=model,
        state=state,
        questions=parsed,
        images=all_images,
        audio=clips or audio,
        videos=videos,  # `files` first, as for images
        unsure_below=None if unsure is None else float(unsure),
    )


ANCHOR = re.compile(rf"<({'|'.join(BY_ANCHOR)}):(\d+)>")  # where a model that reads them there places a request's media


def check_anchors(request: Request, taken: Collection[str] | None = None) -> None:
    """InvalidRequest when the state places a medium the request doesn't have (<image:2> with one image), which a
    model would otherwise answer about without it. Media without an anchor are fine: many models read them first.
    It reads the whole request, a Python caller's media added (questions.with_media), so the decider runs it. `taken`
    is the media fields the model reads ("images", "audio", "videos"; every one when None, as lint has no model):
    for any other, and for a kind the request sends none of, an anchor is the state's own text, read as written: a
    ticket or a web page that contains "<image:1>" is still asked about, never refused."""
    if request.state is None:
        return
    text = request.state if isinstance(request.state, str) else json.dumps(request.state, ensure_ascii=False)
    for kind, n in ANCHOR.findall(text):
        field = BY_ANCHOR[kind].field
        if taken is not None and field not in taken:
            continue
        count = len(request.media(field))
        if count and not 1 <= int(n) <= count:
            raise InvalidRequest(
                f"state: <{kind}:{n}> places the request's {kind} number {n}, and its '{field}' holds {count}"
                " (numbered from 1)"
            )


def _state_images(state: State) -> tuple[State, tuple[Media, ...]]:
    """The state with its image parts taken out, and their images in the order they appear in it (InvalidRequest for
    a part that isn't an image). As many as the model's window holds: the hosted API's cap of 8 is not the model's.

    This is Perplexity's Decisions API form: an image is an OpenAI-style part, {"type": "image_url", "image_url":
    {"url": "data:image/...;base64,..."}}, anywhere in the state (the state itself, an item of a list or a value of
    an object, at any depth), read in the order it is written. pplx-decider's own code takes images apart from the
    state: its `pplx_decider` package reads a request's `images`, in order, before "State:" and the state as text
    (protocol.py, `Encoder.messages`), and has no image parts. So each part comes out of the state whole, leaving the
    rest as written: an item leaves its list, a value leaves its object with its key, and a state that is one part
    leaves "", no text. llama.cpp's server takes the image parts out of a state of chat messages the same way."""
    if not has_image_parts(state):
        return state, ()
    images: list[Media] = []
    rest = _without_images(state, "state", images)
    return ("" if rest is _TAKEN else cast(State, rest)), tuple(images)


_TAKEN = object()  # what an image part leaves in its place: nothing


def has_image_parts(value: object) -> bool:
    """Whether `value` is an image part or holds one, at any depth. It reads every state sent as an object or a list,
    so it walks a stack and steps past text and numbers (a tuple of types: `str | int | float` takes 8.5 ms) before
    the slower Mapping check: 5.9 ms for a 1.1 MB JSON state on an M5 Pro, against 13.4 ms recursing (json.dumps of
    the state, which rendering it costs: 4.3 ms)."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict) or (not isinstance(item, (str, int, float)) and isinstance(item, Mapping)):
            if item.get("type") == IMAGE_PART:
                return True
            stack.extend(item.values())
    return False


def _without_images(value: object, where: str, images: list[Media]) -> object:
    """`value` without its image parts (_TAKEN when it is one), their images appended to `images` in order."""
    if isinstance(value, Mapping):
        if value.get("type") == IMAGE_PART:
            image_url = value.get("image_url")
            url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
            images.append(parse_media(url, "image", where))
            return _TAKEN
        kept = {key: _without_images(v, f"{where}.{key}", images) for key, v in value.items()}
        return {key: v for key, v in kept.items() if v is not _TAKEN}
    if isinstance(value, list):
        items = [_without_images(item, f"{where}[{i}]", images) for i, item in enumerate(value)]
        return [item for item in items if item is not _TAKEN]
    return value


def _parse_media_list(value: object, kind: MediaKind) -> tuple[Media, ...]:
    """Validate an image, clip or video list once, preserving the caller's order and error locations."""
    if value is None:
        return ()
    field = BY_ANCHOR[kind].field
    if not isinstance(value, list):
        raise InvalidRequest(f"'{field}' must be a list of base64 strings or data: URIs")
    return tuple(parse_media(v, kind, f"{field}[{i}]") for i, v in enumerate(value))


def _parse_files(value: object) -> tuple[tuple[Media, ...], tuple[Media, ...], tuple[Media, ...]]:
    """A request's `files`, llama.cpp's server's list of media: its images, its audio clips and its videos, each in
    order, each entry one of them as `_file_kind` tells."""
    if value is None:
        return (), (), ()
    if not isinstance(value, list):
        raise InvalidRequest("'files' must be a list of base64 strings or data: URIs")
    media = [parse_media(v, _file_kind(v), f"files[{i}]") for i, v in enumerate(value)]

    def of(kind: MediaKind) -> tuple[Media, ...]:
        return tuple(m for m in media if m.kind == kind)

    return of("image"), of("audio"), of("video")


def _file_kind(value: object) -> MediaKind:
    """Whether a `files` entry is an audio clip, a video or an image: as its data URI's type says (llama.cpp's server
    reads a video/mp4 one with a vision projector), or for plain base64 as its first bytes say (an audio file or not),
    since llama.cpp's server tells them apart by their bytes. parse_media then checks the whole entry against it, so
    bytes of another kind are refused."""
    if isinstance(value, str) and value[:5].lower() == "data:":
        kind = value[5:11].lower()
        return "audio" if kind == "audio/" else "video" if kind == "video/" else "image"
    if isinstance(value, str):
        with suppress(ValueError):  # not base64: parse_media says so
            head = base64.b64decode(value[:88], validate=True)  # 64 bytes: enough for WebM's DocType
            found = sniff(head)
            if found in VIDEO_FORMATS and head[8:11] != b"M4A":  # a container: a video, unless an M4A recording
                return "video"
            if found in AUDIO_FORMATS:
                return "audio"
    return "image"


def _parse_question(qid: object, body: object) -> Question:
    if not isinstance(qid, str) or not qid:
        raise InvalidRequest("question names must be non-empty strings")
    if not isinstance(body, Mapping):
        raise InvalidRequest(f"{qid}: a question is a JSON object")
    for key in body:
        if key not in _QUESTION_FIELDS:
            raise InvalidRequest(f"{qid}: " + _unknown(key, _QUESTION_FIELDS, "a question"))
    kind = body.get("type")
    if kind not in QUESTION_TYPES:
        raise InvalidRequest(
            f"{qid}: 'type' must be noul (yes or no), choice (one of named options) or score (a level on a scale)"
        )
    instructions = body.get("instructions")
    criteria = body.get("criteria")
    if kind == "noul":
        pairs = _noul_criteria(qid, criteria)
    elif kind == "choice":
        pairs = _choice_criteria(qid, criteria)
    else:
        pairs = _score_criteria(qid, criteria)
    if has_image_parts(instructions) or any(has_image_parts(value) for _, value in pairs):
        # Perplexity's Decisions API documents images in the state only; the creator's code would read a part here as
        # JSON text, its base64 included.
        raise InvalidRequest(f"{qid}: images go in the state, not in a question")
    if isinstance(instructions, Mapping) and instructions.get("type") == "input_audio":
        instructions = _spoken(qid, instructions)
    return Question(id=qid, type=cast(QuestionType, kind), instructions=instructions, criteria=pairs)


def _spoken(qid: str, instructions: Mapping[str, Any]) -> Media:
    """A spoken question's clip, from its input_audio part."""
    part = instructions.get("input_audio")
    if set(instructions) != {"type", "input_audio"} or not isinstance(part, Mapping) or "data" not in part:
        raise InvalidRequest(f'{qid}: a spoken question is {{"type": "input_audio", "input_audio": {{"data": ...}}}}')
    if part.get("format", "wav") not in ("wav", "mp3") or set(part) - {"data", "format"}:
        raise InvalidRequest(f"{qid}: a spoken question's input_audio is its clip's data (format wav or mp3)")
    return parse_media(part["data"], "audio", f"{qid}: instructions")


def _noul_criteria(qid: str, criteria: object) -> tuple[tuple[str, Any], ...]:
    if criteria is None:
        return ()
    if not isinstance(criteria, Mapping) or not set(criteria) <= set(NOUL_OPTIONS):
        raise InvalidRequest(f'{qid}: a noul\'s criteria are {{"true": ..., "false": ...}}')
    return tuple(criteria.items())


def _choice_criteria(qid: str, criteria: object) -> tuple[tuple[str, Any], ...]:
    if isinstance(criteria, Mapping):
        pairs = tuple(criteria.items())
    elif isinstance(criteria, list):
        if not all(isinstance(c, str) for c in criteria) or len(set(criteria)) != len(criteria):
            raise InvalidRequest(f"{qid}: a choice's labels must be distinct strings")
        pairs = tuple((label, None) for label in criteria)
    else:
        raise InvalidRequest(f'{qid}: a choice needs criteria: {{"label": "description", ...}} or [labels]')
    if not pairs:
        raise InvalidRequest(f"{qid}: a choice has at least 1 option")
    if any(not isinstance(label, str) or not label for label, _ in pairs):
        raise InvalidRequest(f"{qid}: choice labels must be non-empty strings")
    return pairs


def _score_criteria(qid: str, criteria: object) -> tuple[tuple[str, Any], ...]:
    """A score's levels, lowest first: each any JSON value but null (text usually), as a choice's descriptions."""
    if not isinstance(criteria, list) or any(level is None for level in criteria):
        raise InvalidRequest(f'{qid}: a score needs criteria: ["lowest level", ..., "highest level"]')
    if len(criteria) < MIN_SCORE_LEVELS:
        raise InvalidRequest(f"{qid}: a score has {MIN_SCORE_LEVELS} levels or more, not {len(criteria)}")
    return tuple((str(i), level) for i, level in enumerate(criteria))


@dataclass(frozen=True)
class Answer:
    """A question's probabilities, one per option in question.options. They are checked on creation:
    a wrong number of them, a negative or non-finite one, or a sum away from 1 is a bug upstream, and
    is raised instead of returned."""

    question: Question
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.question.options)
        if len(self.probabilities) != n:
            raise ValueError(f"{self.question.id}: {len(self.probabilities)} probabilities for {n} options")
        if not all(math.isfinite(p) for p in self.probabilities):
            raise NonFiniteAnswer(f"{self.question.id}: probabilities must be finite")
        if not all(p >= 0.0 for p in self.probabilities):
            raise ValueError(f"{self.question.id}: probabilities must be non-negative")
        if abs(math.fsum(self.probabilities) - 1.0) > 1e-4:
            raise ValueError(f"{self.question.id}: probabilities sum to {math.fsum(self.probabilities)}")

    @property
    def by_option(self) -> dict[str, float]:
        """Each option's probability, by option."""
        return dict(zip(self.question.options, self.probabilities, strict=True))

    @property
    def confidence(self) -> float:
        """The most likely option's probability: how sure the model is of its answer, for any question type."""
        return max(self.probabilities)

    @property
    def top(self) -> str:
        """The most likely option: "true"/"false" for a noul, a label for a choice, "0".."n-1" for a score."""
        return top_option(self.question.options, self.probabilities)

    @property
    def p_yes(self) -> float:
        """A noul's probability of "true"."""
        if self.question.type != "noul":
            raise TypeError(f"{self.question.id} is a {self.question.type}, not a noul")
        return self.by_option["true"]

    @property
    def expected_level(self) -> float:
        """A score's expected level, sum of level x probability."""
        if self.question.type != "score":
            raise TypeError(f"{self.question.id} is a {self.question.type}, not a score")
        return expected_level(self.probabilities)

    def to_systemone(self) -> dict[str, Any]:
        """The answer in the System One response shape."""
        return systemone_answer(self.question, self.probabilities)


@dataclass(frozen=True)
class Result:
    """A request's answers: typed (`result["id"]`) and as the model's own server returns them (`response`)."""

    answers: Mapping[str, Answer]
    response: dict[str, Any]
    precision: str = ""  # what it computed in: "fp16", "bf16", ... (set when the decider answers)
    fallback: str | None = None  # why another path than the first answered it, if one did
    build: Mapping[str, Any] | None = None  # the artifact the model was loaded from (provenance.py)
    kept_tokens: int = 0  # how many of its input tokens were taken from an earlier request's reading (prefixes.py)
    # the milliseconds its rendering took and the pass that scored it ("render", "score"), when the caller timed
    # them (`timed`), as the decide API's response gives them
    timing_ms: Mapping[str, float] = field(default_factory=dict, compare=False)

    def timed(self, render_s: float, score_s: float) -> Result:
        """This result with the seconds its rendering and the pass that scored it took, as `timing_ms`: what
        batches.answer_each and Decider.sqwish both record."""
        return replace(self, timing_ms={"render": round(render_s * 1000, 3), "score": round(score_s * 1000, 3)})

    def __getitem__(self, question: str) -> Answer:
        return self.answers[question]

    def unsure(self, below: float) -> list[str]:
        """The questions whose answer's confidence (its most likely option's probability) is under `below`: the ones to
        send on, to a person or a larger model, while the rest are settled here."""
        return [q for q, a in self.answers.items() if a.confidence < below]

    @property
    def diagnostics(self) -> dict[str, Any]:
        """How it was answered: the build `Decider.load` read (its name, repository and commit, or GGUF file and
        sha256: provenance.py; absent for a decider made from loaded parts), the precision, whether input was cut to
        fit the model's window (the response's `truncated` lists what), the fallback's reason, if any, and how many
        input tokens came from an earlier request's reading instead of being read again."""
        return {
            **({"build": dict(self.build)} if self.build is not None else {}),
            "precision": self.precision,
            "truncated": "truncated" in self.response,
            "fallback": self.fallback,
            "kept_tokens": self.kept_tokens,
        }


def top_option(options: Sequence[str], probabilities: Sequence[float]) -> str:
    """The most likely of `options` (the first of a tie), by their `probabilities` in the same order."""
    return options[max(range(len(options)), key=probabilities.__getitem__)]


def tempered(probabilities: Sequence[float], temperature: float) -> list[float]:
    """`probabilities` with their log-probabilities divided by `temperature` and normalized again: softmax(z / T) for
    the logits z they are the softmax of, sharper below 1 and softer above, in log space so a tiny probability can't
    underflow. A probability of 0 stays 0; a temperature of 1 changes nothing."""
    if temperature == 1:
        return list(probabilities)
    logs = [math.log(p) if p > 0 else -math.inf for p in probabilities]
    top = max(logs)
    weights = [math.exp((x - top) / temperature) for x in logs]
    total = math.fsum(weights)
    return [w / total for w in weights]


def carried(response: Mapping[str, Any]) -> dict[str, Any]:
    """What a System One response says beside its answers that another API's response (the decide API's, SGLang's)
    carries as it is, so none hides it: the input cut to fit the model's window (`truncated`), video frame sampling
    (`videos`), and its `warnings` (low-bit drift or capped video sampling); {} when it says none."""
    return {key: response[key] for key in ("truncated", "videos", "warnings") if key in response}


def expected_level(probabilities: Sequence[float]) -> float:
    """A score's expected level, sum of level x probability, from its levels' probabilities lowest first."""
    return math.fsum(i * p for i, p in enumerate(probabilities))


def false_first(question: Question, scores: Sequence[float]) -> tuple[float, ...]:
    """Probabilities in question.options order from scores in the order a prompt shows the options, where a noul shows
    false before true (vllm-sr d3's and Perplexity's), each divided by their sum."""
    total = sum(scores)
    p = tuple(float(x) / total for x in scores)
    return (p[1], p[0]) if question.type == "noul" else p


def systemone_answer(
    q: Question,
    probabilities: Sequence[float],
    *,
    choice_confidence: Confidence = top_probability,
    score_confidence: Confidence = top_probability,
    legend: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """An answer in the System One response shape, from its probabilities in q.options order: noul -> P(true); choice
    -> the most likely label, its confidence and every probability; score -> the expected level, its confidence, the
    legend (the caller's level texts unless a creator shows them otherwise) and every probability. A creator's own
    confidence measure replaces the default, the most likely option's probability."""
    probs = dict(zip(q.options, probabilities, strict=True))
    if q.type == "noul":
        return {"type": "noul", "noul": probs["true"]}
    if q.type == "choice":
        return {
            "type": "choice",
            "choice": top_option(q.options, probabilities),
            "confidence": choice_confidence(probabilities),
            "probabilities": probs,
        }
    return {
        "type": "score",
        "score": expected_level(probabilities),
        "confidence": score_confidence(probabilities),
        "legend": dict(q.criteria) if legend is None else dict(legend),
        "probabilities": probs,
    }


def rounded(answer: dict[str, Any], digits: int) -> dict[str, Any]:
    """A System One answer with its numbers (P(true), the expected level, the confidence and every probability)
    rounded to `digits` decimals, as some creators' servers return them."""
    out = {k: round(v, digits) if k in ("noul", "score", "confidence") else v for k, v in answer.items()}
    if "probabilities" in out:
        out["probabilities"] = {k: round(x, digits) for k, x in out["probabilities"].items()}
    return out


def as_text(value: Any, none: str = "null") -> str:
    """A request's value (a state, instructions, a description) as creators' code writes it into a prompt: text as it
    is, any other JSON value written out with non-ASCII kept; a missing one (None) as `none`, JSON's null unless a
    creator writes it otherwise."""
    if value is None:
        return none
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
