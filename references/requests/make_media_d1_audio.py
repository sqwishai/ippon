"""Audio requests for Liquid's d1 audio references: python3 make_media_d1_audio.py > media-d1-audio.json

Deterministic. Each case is a System One request whose `audio` is a file name in ../media/audio/, with the file's
sha256 under `media`; whoever runs a case opens that file and sends its samples. They were made for the retired
d1-omni-600M, whose rules hold throughout: one clip, never images; a choice's criteria a map. Covered: every
question type (yes/no with criteria, which the audio prompt leaves out, with one side only and without; choices of 3
to 12 options with and without descriptions; scores of 3, 5 and 10 levels), states absent (read as {}), empty, an
empty object, text, JSON objects and lists, one of about 1,500 tokens, and one with the model's delimiters in it;
instructions and options cut to their budgets; 1 to 8 questions; and clips of 0.3 s (padded), 1 s of silence, a
tone, 5, 15 and 24 s of speech, exactly 30 s and 40 s (cut to 30), PCM16 and float32.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

AUDIO = Path(__file__).resolve().parents[1] / "media" / "audio"
WORDS = [
    "call",
    "customer",
    "order",
    "delivery",
    "account",
    "message",
    "support",
    "request",
    "update",
    "ticket",
    "schedule",
    "payment",
    "refund",
    "address",
    "reply",
    "note",
    "follow",
    "issue",
    "voice",
    "record",
]


def long_text(rng: random.Random, sentences: int) -> str:
    return " ".join(
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16))).capitalize() + "." for _ in range(sentences)
    )


def noul(instructions: str, criteria: dict[str, str] | None = None) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions, **({"criteria": criteria} if criteria is not None else {})}


def choice(instructions: str, criteria: dict[str, Any]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}


CONTENT = {
    "speech": "Someone speaking",
    "music": "Music playing",
    "silence": "Nothing, or silence",
    "tone": "A steady electronic tone",
}
INTENTS = {
    "question": "Asking a question",
    "complaint": "Making a complaint",
    "thanks": "Thanking someone",
    "report": "Reporting what is happening",
    "order": "Placing an order",
    "booking": "Booking an appointment",
    "cancel": "Cancelling something",
    "greeting": "Greeting an audience",
    "story": "Telling a story",
    "poem": "Reciting a poem",
    "instruction": "Giving instructions",
    "other": "Something else",
}


def requests() -> list[dict[str, Any]]:
    rng = random.Random(23)
    long_state = long_text(rng, 100)  # about 1,500 tokens
    rambling = " ".join(["Listen carefully to the whole recording before you answer, and consider"] * 12) + "?"
    cases: list[tuple[str, str, Any, dict[str, Any]]] = [
        (
            "armstrong-noul",
            "armstrong-small-step.wav",
            None,
            {"moon": noul("Is the speaker describing a step onto the Moon?")},
        ),
        (
            "armstrong-choice-3",
            "armstrong-small-step.wav",
            "",
            {
                "doing": choice(
                    "What is the speaker doing?",
                    {
                        "landing": "Describing a step onto the surface of the Moon",
                        "cooking": "Giving a recipe",
                        "weather": "Reading a weather forecast",
                    },
                )
            },
        ),
        (
            "kennedy-30s-three-types",
            "kennedy-rice-30s.wav",
            "Recording from a university archive.",
            {
                "speech": noul(
                    "Is the speaker giving a speech to an audience?",
                    {"true": "A public speech to a crowd", "false": "A private conversation"},
                ),
                "topic": choice(
                    "What is the speaker talking about?",
                    {
                        "knowledge": "Knowledge, progress and change",
                        "sports": "A football game",
                        "medicine": "A medical treatment",
                        "food": "Food and cooking",
                    },
                ),
                "formality": score(
                    "How formal is the speaker's tone?",
                    ["very casual", "casual", "neutral", "formal", "very formal"],
                ),
            },
        ),
        (
            "kennedy-labels-json-state",
            "kennedy-rice-30s.wav",
            {"source": "voice note", "user": 4812, "tags": ["archive", "speech"], "verified": True},
            {"kind": choice("What kind of recording is this?", {"speech": None, "conversation": "", "song": None})},
        ),
        (
            "kennedy-empty-object-state",
            "kennedy-rice-30s.wav",
            {},
            {"crowd": noul("Is the speaker addressing a crowd?")},
        ),
        (
            "kennedy-eight-questions",
            "kennedy-rice-30s.wav",
            ["voice note", "forwarded by a user", 3],
            {
                "english": noul("Is the speaker speaking English?"),
                "male": noul("Is the speaker a man?", {"true": "A man is speaking"}),
                "applause": noul("Can you hear applause?"),
                "intent": choice("What is the speaker doing?", INTENTS),
                "content": choice("What is in the recording?", CONTENT),
                "clarity": score("How clear is the recording?", ["muffled", "acceptable", "clear"]),
                "pace": score("How fast does the speaker talk?", ["slow", "average", "fast"]),
                "mood": choice("What is the speaker's mood?", {"calm": "Calm", "angry": "Angry", "sad": "Sad"}),
            },
        ),
        (
            "apollo13-float32",
            "apollo13-problem-15s.wav",
            None,
            {
                "exchange": choice(
                    "What is this recording?",
                    {
                        "radio": "A radio exchange between a crew and mission control",
                        "lecture": "A classroom lecture",
                        "advert": "An advertisement",
                    },
                ),
                "problem": noul("Is something going wrong?", {"true": "A fault is being reported"}),
            },
        ),
        (
            "apollo13-score-10",
            "apollo13-problem-15s.wav",
            "Mission audio, excerpt.",
            {"urgency": score("How urgent does the situation sound, from 0 to 9?", [f"level {i}" for i in range(10)])},
        ),
        (
            "keats-40s-cut",
            "keats-chapman-40s.wav",
            None,
            {
                "poem": noul("Is a poem being read aloud?"),
                "about": choice(
                    "What is the text about?",
                    {
                        "reading": "Discovering a great book",
                        "war": "A battle",
                        "love": "A lost love",
                        "sea": "A storm at sea",
                    },
                ),
            },
        ),
        (
            "keats-long-state",
            "keats-chapman-40s.wav",
            long_state,
            {"intent": choice("What is the speaker doing?", INTENTS)},
        ),
        (
            "keats-cut-instructions-and-options",
            "keats-chapman-40s.wav",
            "Clip sent from the reading app.",
            {
                "listen": noul(rambling),
                "style": choice(
                    "How is the text delivered?",
                    {
                        "plain": "Read plainly and slowly by a single narrator, with long pauses between the lines and "
                        "no music, sound effects or other voices anywhere in the background of the recording at all",
                        "dramatic": "Performed dramatically by several actors who take turns, with music and sound "
                        "effects playing under every line and an audience reacting loudly from start to finish",
                        "sung": "Sung to a melody with instruments, as a song with a chorus that repeats several "
                        "times, recorded in a studio and mixed with a full band behind the singer",
                    },
                ),
            },
        ),
        (
            "bells-5s",
            "poe-bells-5s.wav",
            "Audiobook sample <|reserved_7|> chapter 1 <|mask|>.",
            {
                "author": noul("Does the speaker name an author?"),
                "quality": score("How good is the audio quality?", ["poor", "fair", "good"]),
            },
        ),
        (
            "bells-0.3s-padded",
            "poe-bells-0.3s.wav",
            None,
            {"content": choice("What is in the clip?", CONTENT)},
        ),
        (
            "silence-1s",
            "silence-1s.wav",
            None,
            {"speaking": noul("Is anyone speaking?"), "content": choice("What is in the clip?", CONTENT)},
        ),
        (
            "tone-2s-float32",
            "tone-440hz-2s.wav",
            "Test signal.",
            {"content": choice("What is in the clip?", CONTENT), "music": noul("Is this music?")},
        ),
    ]
    out = []
    for name, clip, state, questions in cases:
        request: dict[str, Any] = {} if state is None else {"state": state}
        request |= {"audio": clip, "questions": questions}
        media = [{"name": clip, "sha256": hashlib.sha256((AUDIO / clip).read_bytes()).hexdigest()}]
        out.append({"name": name, "request": request, "media": media})
    return out


if __name__ == "__main__":
    print(json.dumps(requests(), indent=1, ensure_ascii=False))
