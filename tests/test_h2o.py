"""H2O-Lightning-4B: its prompts and usage against the checkpoint's own server (h2o_lightning_shim.py, references/h2o),
with special-token text, combining marks, Greek labels and object instructions; its labels as the server probes
them; its yes/no floor and score confidence; its pictures as the server sends them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from adapter_fixtures import question as q
from adapter_fixtures import renderings, snapshot
from ippon.adapters.h2o import PIXELS_PER_IMAGE_TOKEN, WHITE, H2OFormat
from ippon.adapters.h2o_select import MARK, EvidenceSelectionUnsupported, select_state
from ippon.adapters.qwen import MIN_PIXELS
from ippon.media import load_rgb
from ippon.media.qwen import ImageConfig, QwenImages
from ippon.readout import answer_tokens
from ippon.schema import Question, parse_request
from ippon.tokenizer import Tokenizer
from ippon.verify import load_reference, media_files, reference_request
from media_references import load, no_tower

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/h2o/h2o-lightning-4b.tokens.json.gz"
MEDIA = ROOT / "references/h2o/h2o-lightning-4b-media.json.gz"


def h2o_format() -> H2OFormat:
    model = load_reference(REFERENCE)["model"]
    d = snapshot(model["repo"], model["revision"], "tokenizer.json", "tokenizer_config.json", "serve_config.json")
    config = json.loads((d / "serve_config.json").read_text())
    tok = Tokenizer.from_file(d / "tokenizer.json", json.loads((d / "tokenizer_config.json").read_text()))
    labels = tuple(config["labels"])
    fmt = H2OFormat(tok, None, config["prompt"]["system"], config["prompt"]["prefill"], labels, (), {}, 0.801)
    probe = Question("q", "choice", "q", tuple((f"o{i}", None) for i in range(len(labels))))
    fmt.label_ids = answer_tokens(tok, fmt.prompt_text("s", probe), [" " + label for label in labels])
    assert fmt.label_ids[:3] == (357, 417, 351)  # " A", " B", " C", as the server's probe finds them
    return fmt


def test_prompts_and_usage_are_the_servers() -> None:
    found = load_reference(REFERENCE)
    refused = [c["name"] for c in found["requests"] if "error" in c]
    assert refused == ["choice-list"]  # the server takes a choice's criteria as a map only; ippon also takes a list
    answered = [c for c in found["requests"] if "error" not in c]
    for case, rendering in renderings(h2o_format(), answered, 26):
        assert [r.question for p in rendering.prompts for r in p.readouts] == case["keys"], case["name"]
        assert all(r.position == len(p.tokens) - 1 for p in rendering.prompts for r in p.readouts)


def test_a_weak_yes_or_no_moves_to_the_floor_and_a_scores_confidence_is_its_top_level() -> None:
    fmt = H2OFormat(None, None, "", "", (), (), {}, 0.801)  # type: ignore[arg-type]
    probabilities, answer = fmt.answer(q({"type": "noul"}), [0.6, 0.4])
    assert answer == {"type": "noul", "noul": pytest.approx(0.801)}
    assert probabilities == pytest.approx((0.6, 0.4))  # the model's, which verify compares
    _, score = fmt.answer(q({"type": "score", "criteria": ["low", {"level": 2}, "high"]}), [0.2, 0.5, 0.3])
    assert score["confidence"] == pytest.approx(0.5)
    assert score["legend"] == {"0": "low", "1": '{"level": 2}', "2": "high"}


def test_pictures_lead_the_user_turn_as_the_server_sends_them() -> None:
    """The image requests against the server with vLLM's image loading (references/h2o): the pictures ahead of the
    record, each expanded to its rows; their pixels as the pinned vLLM build loads them (an EXIF turn, transparency
    composited on white) within the server's ceiling of 1,048,576 pixels (12 MP photos scaled down); the usage with
    the shared head once; five pictures, past the shipped server's cap of four (its image.max_images), as the server
    reads them with the cap raised. Read one way (the GGUF build), a prompt is kept just after its pictures, where a
    request with one more picture parts from it. No tower runs."""
    found = load(MEDIA)
    d = snapshot(found["model"]["repo"], found["model"]["revision"], "preprocessor_config.json", "serve_config.json")
    ceiling = json.loads((d / "serve_config.json").read_text())["image"]["max_pixels"]
    image = ImageConfig.from_hf(json.loads((d / "preprocessor_config.json").read_text())).pixels(MIN_PIXELS, ceiling)
    fmt = h2o_format()
    token = found["media_format"]["image_token"]
    fmt.images = QwenImages(image, True, token, no_tower, matte=WHITE)
    fmt.image_bound = ceiling // PIXELS_PER_IMAGE_TOKEN + 2  # as the loader sets it from the same ceiling
    fmt.causal = True
    files = media_files(MEDIA, found)
    for case in found["requests"]:
        request = reference_request(case, files(case))
        rendering = fmt.render(request)
        assert [[t if t >= 0 else token for t in p.tokens] for p in rendering.prompts] == case["prompts"], case["name"]
        assert fmt.usage(rendering)["input_tokens"] == case["input_tokens"], case["name"]
        for p in rendering.prompts:  # the last placeholder, its <|vision_end|>, then the mark
            assert p.marks[0] == max(i for i, t in enumerate(p.tokens) if t < 0) + 2, case["name"]
        patches = [fmt.images.patches(load_rgb(m, exif=True, matte=WHITE)) for m in request.images]
        pixels = np.concatenate([p.pixels for p in patches]).astype("<f4").tobytes()
        for qid in case["keys"]:
            assert [list(p.grid) for p in patches] == case["image_grids"][qid], case["name"]
            assert hashlib.sha256(pixels).hexdigest() == case["pixel_values_sha256"][qid], case["name"]
    assert not [c["name"] for c in found["requests"] if "error" in c]
    assert next(c for c in found["requests"] if c["name"] == "five-images")["shim_env"] == {"SHIM_MAX_IMAGES": "5"}


def test_a_record_over_the_context_keeps_the_passages_that_answer_the_question() -> None:
    """As the server fits it (h2o_lightning_shim.py, `fit_states`): the creator's BM25 selection keeps the passage
    about the question among the filler, the prompt fits the context, and `truncated` says how much was left out."""
    fmt = h2o_format()
    fmt.window = 700
    filler = [f"Paragraph {i}: the weather was mild and the meeting notes were filed as usual." for i in range(60)]
    filler[37] = "Paragraph 37: the customer asked for a refund of the duplicate charge on invoice 4471."
    request = parse_request(
        {
            "state": "\n\n".join(filler),
            "questions": {
                "refund": {"type": "noul", "instructions": "Does the customer ask for a refund on invoice 4471?"}
            },
        }
    )
    rendering = fmt.render(request)
    (prompt,) = rendering.prompts
    text = fmt.tokenizer.decode(list(prompt.tokens))
    assert len(prompt.tokens) <= fmt.window
    assert "refund of the duplicate charge on invoice 4471" in text
    assert text.count("Paragraph") < 60
    ((question, part, dropped, kept),) = [(c.question, c.part, c.dropped, c.kept) for c in rendering.cuts]
    assert (question, part) == ("refund", "state")
    assert dropped > 0 < kept


def words(text: str) -> list[int]:
    """A tokenizer of one token per whitespace-separated word."""
    return [0] * len(text.split())


def bytes4(text: str) -> list[int]:
    """A tokenizer of one token per four bytes."""
    return [0] * (len(text.encode()) // 4 + 1)


def test_a_json_record_is_cut_at_whole_entries_and_kept_whole_only_when_it_fits() -> None:
    """A JSON record is cut at its entries: what is kept fits the budget, holds the entry that answers the question,
    and every line is one whole entry under its path. A record of one entry is sent whole when its exact count fits,
    though its first 16,000 characters promised more, and refused when it doesn't."""
    tickets = [{"id": i, "note": f"ticket {i} was closed after the usual follow-up call"} for i in range(80)]
    tickets[53] = {"id": 53, "note": "the customer wants a refund of the duplicate charge on invoice 4471"}
    lines = {f"tickets[{i}]: {json.dumps(t, separators=(',', ':'))}" for i, t in enumerate(tickets)}
    kept = select_state({"tickets": tickets}, "Is a refund asked on invoice 4471?", ["yes", "no"], 300, bytes4)
    assert len(bytes4(kept)) <= 300
    assert "invoice 4471" in kept
    entries = [line for part in kept.split(MARK) for line in part.split("\n") if line != "tickets"]
    assert 2 < len(entries) < 80
    assert set(entries) <= lines

    record = {"a": "w " * 28 + "x" * 24000}  # 29 words, all in the first 16,000 characters
    whole = json.dumps(record, separators=(",", ":"))
    assert len(words(whole)) == 29
    assert select_state(record, "w?", None, 30, words) == whole
    with pytest.raises(EvidenceSelectionUnsupported, match="29 tokens can't be cut to 28"):
        select_state(record, "w?", None, 28, words)
