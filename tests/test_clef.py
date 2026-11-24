"""Clef-Flash: its prompt and spans against the checkpoint's own encode_record (references/clef), with images too, its
joint schema head against the creator's PyTorch head on a tiny random instance (tests/fixtures/clef_head), its answers'
shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from adapter_fixtures import cases as reference_cases
from adapter_fixtures import question as q
from adapter_fixtures import renderings, snapshot
from ippon.adapters.clef import ClefFormat, JointSchemaHead, Spans
from ippon.schema import parse_request
from ippon.tokenizer import Tokenizer
from ippon.weights import Weights

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
FIXTURE = ROOT / "tests/fixtures/clef_head"


def clef_format() -> ClefFormat:
    d = snapshot(REPO, REVISION, "tokenizer.json")
    return ClefFormat(Tokenizer.from_file(d / "tokenizer.json"), None, None)  # type: ignore[arg-type]  # renders only


def cases() -> list[dict[str, Any]]:
    return reference_cases(ROOT / "references/clef/clef-flash.tokens.json.gz")


def test_the_prompt_and_its_spans_match_clefs_own_code() -> None:
    fmt = clef_format()
    for case, rendering in renderings(fmt, cases(), 20):
        spans = [[list(s.question), [list(o) for o in s.options], list(s.option_ids)] for s in rendering.meta["spans"]]
        assert spans == case["readouts"], case["name"]


def test_image_prompts_spans_and_positions_match_clefs_own_code() -> None:
    """encode_record with the processor, in the card's transformers 5.10.2 (references/clef/make_media_reference.py):
    the images after "STATE:\\n" with their rows, the state cut to what they leave of the window, or the request
    refused when they leave none; each question's spans; every id's multimodal position; each image's pixels."""
    from ippon.media import qwen
    from ippon.media.qwen import QwenImages
    from media_references import check_image_requests, hub_files, load, no_tower

    reference = load(ROOT / "references/clef/clef-flash-media.tokens.json.gz")
    files = hub_files(reference["model"], ("tokenizer.json", "processor_config.json", "config.json"))
    image = qwen.ImageConfig.from_hf(json.loads(files["processor_config.json"].read_text()))
    tok = Tokenizer.from_file(files["tokenizer.json"])
    images = QwenImages(image, False, json.loads(files["config.json"].read_text())["image_token_id"], no_tower)
    fmt = ClefFormat(tok, None, None, images=images)  # type: ignore[arg-type]  # renders only
    assert check_image_requests(fmt, reference, image) >= 16


def test_an_option_named_question_is_an_option() -> None:
    fmt = clef_format()
    body = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "Which?",
                                              "criteria": {"question": "asks something", "answer": None}}}}  # fmt: skip
    (spans,) = fmt.render(parse_request(body)).meta["spans"]
    assert spans.option_ids == ("answer", "question")  # Clef sorts a choice's options
    assert len(spans.options) == 2
    assert spans.question[0] < spans.options[0][0]  # the instruction's span, before the options'


def test_the_head_matches_the_makers_pytorch_head() -> None:
    case = json.loads((FIXTURE / "case.json").read_text())
    head = JointSchemaHead.load(Weights.from_dir(FIXTURE, ["head.safetensors"]), case["config"])
    spans = [
        Spans(
            tuple(s["question"]),
            tuple(tuple(o) for o in s["options"]),
            tuple(str(i) for i in range(len(s["options"]))),
            s["type"],
        )
        for s in case["spans"]
    ]
    got = head(mx.array(case["hidden"]), case["ids"], spans, mx.array(case["output"]))
    for g, want in zip(got, case["logits"], strict=True):
        np.testing.assert_allclose(np.array(g), want, atol=1e-4, rtol=1e-4)


def test_the_imported_head_loads_only_its_own_floating_point_tensors() -> None:
    """From a checkpoint's whole weights (an Ollama build's): the vision tensors beside it are left unread, and a head
    tensor packed into integers is refused."""
    from ippon.errors import UnsupportedModel

    weights = Weights.from_dir(FIXTURE, ["head.safetensors"])
    weights.tensors["model.visual.unused.weight"] = mx.zeros((8, 8), mx.uint32)
    case = json.loads((FIXTURE / "case.json").read_text())
    head = JointSchemaHead.load(weights, case["config"])
    assert "model.visual.unused.weight" in weights.tensors
    assert all(not name.startswith("model.") for name in head.w)
    weights = Weights.from_dir(FIXTURE, ["head.safetensors"])  # loading took the head's tensors out
    weights.tensors["memory_projection.weight"] = weights.tensors["memory_projection.weight"].astype(mx.uint32)
    with pytest.raises(UnsupportedModel, match="floating-point weights"):
        JointSchemaHead.load(weights, case["config"])


def test_answers_take_clefs_shape() -> None:
    fmt = ClefFormat(None, None, None)  # type: ignore[arg-type]
    _, answer = fmt.answer(q({"type": "choice", "criteria": {"b": "x", "a": "y"}}), [0.25, 0.75])  # request order
    assert answer == {"type": "choice", "choice": "a", "confidence": 0.75, "probabilities": {"b": 0.25, "a": 0.75}}
    probs, answer = fmt.answer(q({"type": "noul"}), [0.6, 0.4])
    assert probs == (0.6, 0.4)
    assert answer == {"type": "noul", "noul": 0.6}


def tiny_clef(tmp_path: Path, vision: bool = False) -> None:
    """A tiny Clef checkpoint with Clef-Flash's tokenizer (and, with `vision`, a tiny vision tower whose patches are
    4 pixels wide, and its processor_config.json)."""
    import shutil

    from qwen35_numpy import tiny_config, tiny_weights

    source = snapshot(REPO, REVISION, "tokenizer.json")
    shutil.copy(source / "tokenizer.json", tmp_path / "tokenizer.json")
    vocab = Tokenizer.from_file(tmp_path / "tokenizer.json").backend.get_vocab_size(with_added_tokens=True)
    text = tiny_config(["linear_attention", "full_attention"]) | {"vocab_size": vocab, "tie_word_embeddings": False}
    config: dict[str, Any] = {"model_type": "qwen3_5", "text_config": text}
    rng = np.random.default_rng(1)
    weights = {"model." + k: v for k, v in tiny_weights(text).items()}
    weights["lm_head.weight"] = rng.standard_normal((vocab, 32)).astype(np.float32)
    if vision:
        from qwen_vision_numpy import tiny_config as tower_config
        from qwen_vision_numpy import tiny_weights as tower_weights

        config |= {"image_token_id": 248056, "vision_config": tower_config()["vision_config"] | {"out_hidden_size": 32}}
        weights |= {"model.visual." + k: v for k, v in tower_weights(config).items()}
        processor = {
            "image_processor_type": "Qwen2VLImageProcessor",
            "patch_size": 4,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "size": {"shortest_edge": 256, "longest_edge": 4096},
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1 / 255,
            "resample": 3,
        }
        (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": processor}))
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    case = json.loads((FIXTURE / "case.json").read_text())
    (tmp_path / "joint_head_config.json").write_text(json.dumps(case["config"] | {"hidden_size": 32}))
    fixture = Weights.from_dir(FIXTURE, ["head.safetensors"]).tensors
    head = {  # the fixture's head, its 24-wide inputs made 32-wide for the tiny backbone
        k: rng.standard_normal([32 if d == 24 else d for d in v.shape]).astype(np.float32) * 0.3
        if 24 in v.shape
        else np.array(v)
        for k, v in fixture.items()
    }
    mx.save_safetensors(str(tmp_path / "joint_head.safetensors"), {k: mx.array(v) for k, v in head.items()})


def test_a_clef_checkpoint_loads_by_directory_and_answers_as_its_reference_path(tmp_path: Path) -> None:
    from ippon import Decider

    tiny_clef(tmp_path)
    decider = Decider.load(tmp_path, precision="fp32", pin_memory=False, keep_awake=None)
    assert decider.adapter == "cloudflare/clef"
    reference = Decider.load(
        tmp_path, precision="fp32", pin_memory=False, keep_awake=None, strategy="reference", warm_up=False
    )
    body = next(c for c in cases() if c["name"] == "several-questions")["request"]
    fast, slow = decider.systemone(body), reference.systemone(body)
    assert fast == slow
    assert fast["usage"]["input_tokens"] == len(
        next(c for c in cases() if c["name"] == "several-questions")["prompts"][0]
    )


def test_images_reach_the_tower_and_every_pass_places_their_rows(tmp_path: Path) -> None:
    """A checkpoint with a vision tower takes images: their rows come from the tower, shaped by their grid, and the
    answers read them; requests whose prompts the runtime splits in chunks of 16 through the images, or resumes
    after them from a kept prefix, answer as the reference path reads each prompt whole."""
    from ippon import Decider

    tiny_clef(tmp_path, vision=True)
    decider = Decider.load(tmp_path, precision="fp32", pin_memory=False, keep_awake=None, chunk=16)
    reference = Decider.load(
        tmp_path, precision="fp32", pin_memory=False, keep_awake=None, strategy="reference", warm_up=False
    )
    assert isinstance(decider.format, ClefFormat)
    assert decider.format.media == frozenset({"images"})
    rng = np.random.default_rng(4)
    images = [rng.integers(0, 256, (24, 40, 3), dtype=np.uint8), rng.integers(0, 256, (32, 16, 3), dtype=np.uint8)]
    questions = {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "one", "b": "two", "c": "three"}}}
    asked = [
        {"state": "two photos", "questions": questions, "images": images},
        {"state": "two photos", "questions": questions | {"r": {"type": "noul"}}, "images": images},  # resumes
        {"state": "two photos", "questions": questions, "images": images[::-1]},
    ]
    for body in asked:
        fast, slow = decider.decide_many([body])[0].response, reference.decide_many([body])[0].response
        assert fast == slow
    assert decider.prefixes is not None
    assert decider.prefixes.stats.tokens_reused > 0
    (with_images,) = reference.decide_many([asked[0]])
    rendering = reference.render(with_media_of(asked[0]))
    assert with_images.response["usage"]["input_tokens"] == len(rendering.prompts[0].tokens)
    assert sum(t < 0 for t in rendering.prompts[0].tokens) == 3 * 5 + 4 * 2  # (24 x 40) / 8^2 and (32 x 16) / 8^2
    text_only = reference.decide("two photos", questions)
    assert text_only.response["answers"] != with_images.response["answers"]


def with_media_of(body: dict[str, Any]) -> Any:
    from ippon.questions import with_media

    return with_media(parse_request({k: v for k, v in body.items() if k != "images"}), body["images"], None)
