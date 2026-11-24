"""`ippon lint REQUEST [--model M]`: what in a request may make a decision model's answers worse than they need
be, found before a model answers it.

Without a model it reads the request's own text: an empty state; a question with no instructions, too few words to
say what is asked, several questions in one, or a yes/no question asked as an open one; two questions asking the same;
options that are the same label twice or may overlap (near-identical labels, one label inside another, descriptions
saying nearly the same); a choice of one option; a choice of three or more options with no catch-all ("other",
"none") for a state that fits none of them; and options so long that reading them, not the state, sets the speed.
With a model it also renders the request as that model reads it (`Decider.inspect`, which runs no layer): each
prompt's tokens against the model's window, what would be cut to fit it, and what the model can't be asked at all
(more options than its readout has codes, a question it doesn't take).

The checks on the text are rules of thumb: each finding says why it matters and what to change, and a request may
keep what it flags on purpose. The starter requests in docs/evaluating.md pass them all.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ContextTooLong, InvalidRequest
from .schema import ANCHOR, BY_ANCHOR, Question, check_anchors, parse_request

CATCH_ALLS = frozenset(
    {
        "other",
        "others",
        "none",
        "neither",
        "unknown",
        "unclear",
        "na",
        "not applicable",
        "something else",
        "none of the above",
        "no match",
        "misc",
        "miscellaneous",
        "nothing",
    }
)
CATCH_ALL_WORDS = ("anything else", "everything else", "something else", "none of", "any other", "not listed")
OPEN = frozenset({"what", "which", "who", "whom", "whose", "when", "where", "why", "how"})  # not a yes/no question
LONG_OPTIONS = (
    4000  # characters of a choice's labels and descriptions past which its length, not its state, sets its speed
)
NEAR = 0.85  # difflib's ratio between two labels at and above which they may be one option written twice
MIN_WORDS = 3  # instructions shorter than this rarely say what to look at


@dataclass(frozen=True)
class Finding:
    """A problem in a request: its question's id ("" for the request as a whole) and what to change."""

    question: str
    message: str

    def __str__(self) -> str:
        return f"{self.question}: {self.message}" if self.question else self.message

    def to_json(self) -> dict[str, str]:
        return {"question": self.question, "message": self.message}


class NotARequest(Finding):
    """The finding that a body is no System One request at all, so no model is asked about it either."""


def lint(body: Any) -> list[Finding]:
    """The problems `lint` finds in a request body (decoded JSON) without a model: one NotARequest alone when it
    isn't a request."""
    try:
        request = parse_request(body)
        check_anchors(request)
    except InvalidRequest as error:
        return [NotARequest("", f"not a System One request a model can be asked: {error}")]
    found: list[Finding] = []
    stated = request.state if isinstance(request.state, str) else json.dumps(request.state, ensure_ascii=False)
    for kind in sorted({kind for kind, _ in ANCHOR.findall(stated or "")}):
        if not request.media(BY_ANCHOR[kind].field):  # read as the state's text: a forgotten attachment?
            found.append(
                Finding(
                    "",
                    f"the state places <{kind}:N>, but the request sends no {BY_ANCHOR[kind].field}: a model reads "
                    f"the tag as text. Add the {kind} to '{BY_ANCHOR[kind].field}', or leave it if the text is meant",
                )
            )
    if _empty(request.state) and not request.images and not request.audio and not request.videos:
        found.append(
            Finding(
                "",
                "the state is empty, so the questions have nothing to be answered about: put the text (or "
                "JSON) to decide on in 'state'",
            )
        )
    asked: dict[str, str] = {}
    for q in request.questions:
        found += _instructions(q)
        found += _options(q)
        key = _plain(q.instructions) if isinstance(q.instructions, str) else ""
        if key and key in asked:
            found.append(Finding(q.id, f"asks what {asked[key]} asks: ask it once, or say in each what sets it apart"))
        asked.setdefault(key, q.id)
    return found


def against(decider: Any, body: Any) -> tuple[list[Finding], dict[str, Any]]:
    """The problems a model finds in a request it renders (`Decider.inspect`): what it can't be asked, and what would be
    cut to fit its window; and what it read (its prompts' tokens and window), {} when it couldn't render it."""
    who = decider.name or decider.adapter
    try:
        seen: dict[str, Any] = decider.inspect(body)
    except InvalidRequest as error:  # more options than its readout has codes, a question it doesn't take, ...
        return [Finding("", f"{who} can't be asked this: {error}")], {}
    except ContextTooLong as error:
        return [Finding("", f"too long for {who}: {error}; shorten the state, or send the part the questions need")], {}
    found = []
    for cut in seen["truncated"]:
        part = "the state" if cut["part"] == "state" else f"its {cut['part']}" if cut["question"] else cut["part"]
        dropped, total = cut["tokens_dropped"], cut["tokens_dropped"] + cut["tokens_kept"]
        found.append(
            Finding(
                cut["question"],
                f"{part} would be cut to fit {who}'s window: {dropped} of its {total} tokens "
                "left out, unread; shorten it, or send only what the questions need",
            )
        )
    return found, seen


def _instructions(q: Question) -> list[Finding]:
    text = q.instructions
    if text is None or (isinstance(text, str) and not text.strip()):
        example = "Is the customer asking for money back?" if q.type == "noul" else "Which team should handle this?"
        return [Finding(q.id, f"no instructions: write the question the model answers, in full ('{example}')")]
    if not isinstance(text, str):
        return []  # instructions given as JSON: the model's format shows them as they are
    words = re.findall(r"[^\W_]+(?:['-][^\W_]+)*", text)
    found = []
    if len(words) < MIN_WORDS:
        found.append(
            Finding(q.id, f"{text!r} is too short to say what is asked: write a full question naming what to look at")
        )
    if text.count("?") > 1:
        found.append(
            Finding(
                q.id,
                f"asks {text.count('?')} things in one question: ask each as a question of its own, so each "
                "gets an answer of its own",
            )
        )
    if q.type == "noul" and words and words[0].lower() in OPEN:
        found.append(
            Finding(
                q.id,
                f"a noul is answered yes or no, and {text!r} asks for something else: ask it as 'Is ...?' or "
                "'Does ...?', or make it a choice among the answers",
            )
        )
    return found


def _options(q: Question) -> list[Finding]:
    """A choice's labels (and their descriptions) and a score's levels: twice the same, or possibly overlapping; a
    choice of one option, or of three or more with no catch-all."""
    if q.type == "noul":
        return []
    if q.type == "score":
        named = [(str(level), "") for _, level in q.criteria if isinstance(level, str)]
    else:
        named = [(label, d if isinstance(d, str) else "") for label, d in q.criteria]
    found = []
    what = "levels" if q.type == "score" else "options"
    for i, (a, about_a) in enumerate(named):
        for b, about_b in named[i + 1 :]:
            pa, pb = _plain(a), _plain(b)
            if pa == pb:
                found.append(Finding(q.id, f"{what} {a!r} and {b!r} are one written twice: keep one"))
            elif _overlap(pa, pb):
                found.append(
                    Finding(
                        q.id,
                        f"{what} {a!r} and {b!r} may overlap: merge them, or say in each one's description "
                        "what sets it apart",
                    )
                )
            elif about_a and about_b and _plain(about_a) == _plain(about_b):
                found.append(Finding(q.id, f"{a!r} and {b!r} have the same description: say what sets each apart"))
    if q.type != "choice":
        return found
    if len(named) == 1:
        found.append(
            Finding(q.id, "a choice of one option is always answered with it: add the alternatives, or ask a noul")
        )
    elif len(named) >= 3 and not any(_catch_all(label, about) for label, about in named):
        found.append(
            Finding(
                q.id,
                f"no catch-all among its {len(named)} options: a state that fits none of them still gets "
                "one of them; add 'other' (anything the rest don't cover) unless every state fits one",
            )
        )
    written = sum(len(label) + len(about) for label, about in named)
    if written > LONG_OPTIONS:
        found.append(
            Finding(
                q.id,
                f"its {len(named)} options run to {written:,} characters, and every state it's asked about reads "
                "them all: a request's time grows with its length, so a few words per option answer faster",
            )
        )
    return found


def _overlap(a: str, b: str) -> bool:
    """Whether two labels (`_plain`) may name one option: one's words all within the other's, or near-identical."""
    wa, wb = set(a.split()), set(b.split())
    return bool(wa and wb) and (wa <= wb or wb <= wa or difflib.SequenceMatcher(None, a, b).ratio() >= NEAR)


def _catch_all(label: str, about: str) -> bool:
    return _plain(label) in CATCH_ALLS or any(words in about.lower() for words in CATCH_ALL_WORDS)


def _plain(text: str) -> str:
    """`text` in lower case, its words only, one space apart: "Feature_Request!" -> "feature request"."""
    return " ".join(re.findall(r"[^\W_]+", text.lower().replace("/", " ")))


def _empty(state: Any) -> bool:
    if state is None:
        return True
    if isinstance(state, str):
        return not state.strip()
    return isinstance(state, Mapping | list) and not state
