"""Perplexity and d3: pinned creator prompts, code readouts, calibration and admission contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from adapter_fixtures import cases, renderings, snapshot
from adapter_fixtures import question as q
from ippon import hub, locate
from ippon.adapter import AdapterBase, Hidden
from ippon.adapters.d3 import D3Format
from ippon.adapters.perplexity import (
    DEFAULT_MAX_LENGTH,
    MARKER,
    PROTOCOL,
    SEPARATOR,
    Perplexity,
    PerplexityFormat,
    descriptions,
    escapes,
    sequence_limit,
)
from ippon.errors import ContextTooLong, UnsupportedModel
from ippon.model_dir import ModelDir
from ippon.readout import CODE_ROWS, CodeReadout
from ippon.schema import parse_request
from ippon.tokenizer import Tokenizer
from ippon.verify import load_reference
from media_references import hub_files

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "references" / "perplexity" / "pplx-decider-flash-v1.3.json.gz"
CODES = tuple([chr(65 + i) for i in range(26)] + [chr(65 + i // 26) + chr(65 + i % 26) for i in range(CODE_ROWS - 26)])
CONFIG = {  # the special-token fields of flash v1.3's tokenizer_config.json
    "bos_token": None,
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "unk_token": None,
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "audio_token": "<|audio_pad|>",
    "vision_bos_token": "<|vision_start|>",
    "model_specific_special_tokens": {"audio_bos_token": "<|audio_start|>", "audio_eos_token": "<|audio_end|>"},
    "pad_token_type_id": 0,
}


def text_format() -> PerplexityFormat:
    """The format without a tokenizer: enough for the prompt's text and the answers."""
    return PerplexityFormat(None, CodeReadout(mx.zeros((CODE_ROWS, 4)), CODES, 1.5), escapes(CONFIG), 0, 1)  # type: ignore[arg-type]


def test_escapes_are_read_from_every_special_token_field_longest_first() -> None:
    """transformers' all_special_tokens, the marker and the separator (the set flash v1.3's config gives is held
    against the creator's by test_prompts_and_escapes_match_the_makers_path): an extra token given as text or as an
    object, longest first, so a token that contains another is neutralized whole."""
    lengths = [len(t) for t in escapes(CONFIG)]
    assert lengths == sorted(lengths, reverse=True)
    assert escapes({"extra_special_tokens": ["<x>", {"content": "<yy>"}]}) == (SEPARATOR, MARKER, "<yy>", "<x>")


def test_option_texts_follow_the_maker() -> None:
    assert descriptions(q({"type": "choice", "criteria": {"a": None, "b": {"why": 1}}})) == ["a", 'b: {"why": 1}']
    assert descriptions(q({"type": "noul", "instructions": "x"})) == ["No / false", "Yes / true"]
    assert descriptions(q({"type": "noul", "instructions": "x", "criteria": {"true": "It is"}})) == [
        "No / false",
        "It is",
    ]


def test_answers_have_the_makers_shape() -> None:
    fmt = text_format()
    probabilities, yes = fmt.answer(q({"type": "noul", "instructions": "x"}), [0.25, 0.75])
    assert probabilities == (0.75, 0.25)  # option order: true, false
    assert yes == {"type": "noul", "noul": 0.75}
    _, choice = fmt.answer(q({"type": "choice", "criteria": {"a": None, "b": None, "c": None}}), [0.2, 0.6, 0.2])
    assert choice["choice"] == "b"
    assert choice["confidence"] == pytest.approx((0.6 - 1 / 3) / (2 / 3))
    _, score = fmt.answer(q({"type": "score", "criteria": ["low", "mid", "high"]}), [0.0, 0.5, 0.5])
    assert score["score"] == pytest.approx(1.5)
    assert score["confidence"] == pytest.approx(1.0 - 0.5 / (2 / 3))
    assert list(score["legend"].values()) == ["low", "mid", "high"]


def test_scores_take_the_first_codes_at_the_calibrated_temperature() -> None:
    from ippon.adapter import Rendering

    readout = mx.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    fmt = PerplexityFormat(None, CodeReadout(readout, CODES, 2.0), escapes({}), 0, 1)  # type: ignore[arg-type]
    request = parse_request({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}})
    out = fmt.scores(Rendering(request, ()), Hidden({"q": mx.array([[9.0, 9.0], [3.0, 1.0]])}))
    assert mx.allclose(out["q"], mx.softmax(mx.array([3.0, 1.0]) / 2.0)).item()


def test_recognised_by_its_protocol(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert not Perplexity().matches(ModelDir.open(tmp_path))
    (tmp_path / "joint_config.json").write_text(json.dumps({**PROTOCOL, "max_questions": None}))
    assert Perplexity().matches(ModelDir.open(tmp_path))
    (tmp_path / "joint_config.json").write_text(json.dumps({**PROTOCOL, "version": 3}))
    assert not Perplexity().matches(ModelDir.open(tmp_path))


def real_format(file: Path = FULL) -> tuple[PerplexityFormat, dict[str, Any]]:
    reference = load_reference(file)
    files = hub_files(reference["model"], ("tokenizer.json", "tokenizer_config.json"))
    tok = Tokenizer.from_file(Path(files["tokenizer.json"]))
    config = json.loads(Path(files["tokenizer_config.json"]).read_text())
    marker, separator = tok.token_id(MARKER), tok.token_id(SEPARATOR)
    assert marker is not None
    assert separator is not None
    readout = CodeReadout(mx.zeros((CODE_ROWS, 4)), CODES, 1.5)
    window = reference["maker"].get("max_length", DEFAULT_MAX_LENGTH)
    return PerplexityFormat(tok, readout, escapes(config), marker, separator, window=window), reference


def test_overlength_plans_match_the_maker_before_any_inference(tmp_path: Path) -> None:
    for name in ("overlength.json.gz", "overlength-default.tokens.json.gz"):
        fmt, reference = real_format(FULL.with_name(name))
        for case in reference["requests"]:
            request = parse_request(case["request"])
            if "error" in case:
                with pytest.raises(ContextTooLong, match="with the state and images"):
                    fmt.render(request)
                continue
            rendering = fmt.render(request)
            assert [list(p.tokens) for p in rendering.prompts] == case["prompts"], case["name"]
            assert fmt.input_tokens(rendering) == case["input_tokens"]
            assert [r.question for p in rendering.prompts for r in p.readouts] == [q.id for q in request.questions]

    fmt, reference = real_format(FULL.with_name("overlength.json.gz"))
    request = parse_request(reference["requests"][0]["request"])
    length = max(len(p.tokens) for p in fmt.render(request).prompts)
    fmt.window = length
    assert len(fmt.render(request).prompts) == len(request.questions)  # equal to the limit is allowed
    fmt.window -= 1
    with pytest.raises(ContextTooLong):
        fmt.render(request)
    fmt.window = None
    assert len(fmt.render(request).prompts) == len(request.questions)

    (tmp_path / "config.json").write_text("{}")
    model = ModelDir.open(tmp_path)
    assert sequence_limit(model) == DEFAULT_MAX_LENGTH
    for value in (False, 0, -1, 1.5, "256"):
        with pytest.raises(UnsupportedModel, match="max_length"):
            sequence_limit(ModelDir.open(tmp_path, {"ippon.json": {"max_length": value}}))


def test_prompts_and_escapes_match_the_makers_path() -> None:
    fmt, reference = real_format()
    assert set(fmt.neutralized) == set(reference["maker"]["escapes"])
    renderings(fmt, reference["requests"], 12)


def test_image_prompts_and_positions_match_the_makers_path() -> None:
    """flash v1.3's own Encoder on image requests (references/perplexity/make_media_reference.py): every sequence's
    ids with each image's rows, their multimodal positions, each image's pixels, and the usage."""
    from ippon.adapters.qwen import MIN_PIXELS
    from ippon.media import qwen
    from ippon.media.qwen import QwenImages
    from media_references import check_image_requests, hub_files, load, no_tower

    reference = load(FULL.with_name("pplx-decider-flash-v1.3-media.json.gz"))
    names = ("tokenizer_config.json", "processor_config.json", "config.json", "decision_config.json")
    files = hub_files(reference["model"], ("tokenizer.json", *names))
    tok = Tokenizer.from_file(files["tokenizer.json"])
    configs = {n: json.loads(files[n].read_text()) for n in names}
    marker, separator = tok.token_id(MARKER), tok.token_id(SEPARATOR)
    assert marker is not None
    assert separator is not None
    ceiling = configs["decision_config.json"]["max_pixels"]
    image = qwen.ImageConfig.from_hf(configs["processor_config.json"]).pixels(MIN_PIXELS, ceiling)
    assert reference["maker"]["pixels"] == {"min": MIN_PIXELS, "max": ceiling}
    images = QwenImages(image, False, configs["config.json"]["image_token_id"], no_tower)
    assert images.token == tok.token_id("<|image_pad|>")
    readout = CodeReadout(mx.zeros((CODE_ROWS, 4)), CODES, 1.5)
    fmt = PerplexityFormat(tok, readout, escapes(configs["tokenizer_config.json"]), marker, separator, images)
    assert check_image_requests(fmt, reference, image) >= 25


SIZES = {  # the checkpoints the references were made from
    "mini": "c5aa751c277d26905c28747706023709ea56324b",
    "flash": "13dc1cc8ef23caefa765907e1d764751e5629f7d",
}


def d3_format(size: str = "mini") -> D3Format:
    directory = snapshot(f"vllm-sr/d3-{size}", SIZES[size], "tokenizer.json", "decision_config.json")
    config = json.loads((directory / "decision_config.json").read_text())
    readout = CodeReadout(mx.zeros((255, 1)), tuple(config["codes"]), 1.0)
    return D3Format(Tokenizer.from_file(directory / "tokenizer.json"), readout)


@pytest.mark.parametrize("size", list(SIZES))
def test_prompts_and_token_counts_match_vllm_srs_own_runtime(size: str) -> None:
    fmt = d3_format(size)
    renderings(fmt, cases(ROOT / f"references/d3/d3-{size}.json.gz"), 18)


def test_answers_take_d3s_shape() -> None:
    fmt = D3Format(None, CodeReadout(mx.zeros((255, 1)), ("A",), 1.0))  # type: ignore[arg-type]
    probs, answer = fmt.answer(q({"type": "noul"}), [0.25, 0.75])  # prompt order: false, true
    assert probs == (0.75, 0.25)
    assert answer == {"type": "noul", "noul": 0.75}
    _, answer = fmt.answer(q({"type": "choice", "criteria": ["a", "b"]}), [0.5, 0.5])
    assert answer["confidence"] == pytest.approx(0.0)  # 1 - normalized entropy
    _, answer = fmt.answer(q({"type": "score", "criteria": ["low", "mid", "high"]}), [0.0, 0.0, 1.0])
    assert answer["score"] == 2.0
    assert answer["confidence"] == 1.0


class Lora:
    """An adapter whose model is a LoRA on a pinned base, as Kev's `base` names it from head.pt."""

    def base(self, model: ModelDir) -> tuple[str, str]:
        return "Qwen/Qwen3.5-9B", "c2022362"


def test_the_base_is_downloaded_at_its_revision_before_the_model_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[tuple[str, tuple[str, ...]]] = []
    (tmp_path / "base").mkdir()

    def fetch(name: str, patterns: Any) -> Path:
        fetched.append((name, tuple(patterns)))
        return tmp_path / "base"

    monkeypatch.setattr(hub, "fetch", fetch)
    (tmp_path / "lora").mkdir()
    lora = ModelDir.open(tmp_path / "lora")
    base = locate.open_base(Lora(), lora)  # type: ignore[arg-type]
    assert base is not None
    assert base.path == tmp_path / "base"
    assert [name for name, _ in fetched] == ["Qwen/Qwen3.5-9B@c2022362"]  # the base at its pinned revision
    assert locate.open_base(AdapterBase(), lora) is None  # type: ignore[arg-type]  # no base: nothing fetched
