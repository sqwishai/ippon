"""Builds below 8 bits load only when asked for, and every such load says how far it drifts; formats ippon doesn't
run are refused before a model opens, naming a build it does."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from gguf_writer import write
from ippon import Decider, drift, locate, names, onboarding, recommend
from ippon.cli import main
from ippon.device import Device
from ippon.drift import LowBitWarning
from ippon.errors import InvalidRequest, UnsupportedModel
from ippon.gguf.builds import find
from ippon.quant import Stacked, quantize
from ippon.quantization import inspect
from ippon.safetensors import Tensor
from ippon.schema import parse_request
from ippon.weights import Weights
from test_generic_adapters import LABELS, qwen3_classifier

POLICY = "full precision or 8-bit, and at 4 to 6 bits or fp4 only when asked"
SMALL_MAC = Device("Apple M1", "applegpu_g13g", 8 << 30, 6 << 30, (26, 0), True)
BIG_MAC = Device("Apple M5 Pro", "applegpu_g17s", 48 << 30, 36 << 30, (26, 2), True, gpu_cores=20)
REQUEST = {"state": "w1 w2", "questions": {"team": {"type": "choice", "criteria": list(LABELS.values())}}}
DECIDE = {
    "context": "w1 w2",
    "decisions": [{"id": "team", "question": "Which team?", "outcomes": list(LABELS.values())}],
}


def test_no_bare_name_recommendation_or_setup_picks_a_build_below_8_bits() -> None:
    low = [n for n in names.named() if n.low_bit]
    assert low  # d3-mini:nvfp4, which loads by its name only
    for family, builds in names.families().items():
        chosen = names.recommended(builds)
        assert chosen is None or not chosen[0].low_bit, family
        assert not names.find(family)[0].low_bit
    for device in (SMALL_MAC, BIG_MAC):
        picks = recommend.recommend("text", device)
        assert not any(p.named.low_bit or any(s.low_bit for s in p.smaller) for p in picks)
        listed = onboarding.choices(device, every_build=True)
        assert not listed[onboarding.first_usable(listed, device)].named.low_bit
    # a listing where the low-bit build comes first and fits: setup still starts with the 8-bit one
    first = [onboarding.Choice(low[0]), onboarding.Choice(names.find("d1-2b:q8_0")[0])]
    assert onboarding.first_usable(first, BIG_MAC) == 1
    # asked for by name, it loads: a low-bit build's own tag, or a family at q4 from its full-precision build
    assert names.find(low[0].name)[0] == low[0]
    assert names.named_precision("kettei-core:q4", "auto") == "q4"
    assert names.find("kettei-core:q4")[0].tag == "bf16"


def quantized_checkpoint(path: Path, mode: str, bits: int, group: int) -> None:
    """The tiny classifier as a published MLX build: its layers' matrices packed in `mode`."""
    config, weights = qwen3_classifier(path, intermediate_size=64)
    tensors: dict[str, mx.array] = {}
    for name, value in weights.items():
        array = mx.array(value)
        if "model.layers." in name and name.endswith("proj.weight"):
            base = name.removesuffix(".weight")
            packed = mx.quantize(array, group_size=group, bits=bits, mode=mode)
            tensors |= {name: packed[0], base + ".scales": packed[1]}
            if mode == "affine":
                tensors[base + ".biases"] = packed[2]
        else:
            tensors[name] = array
    mx.save_safetensors(str(path / "model.safetensors"), tensors)
    quantization = {"mode": mode, "bits": bits, "group_size": group}
    (path / "config.json").write_text(json.dumps(config | {"quantization": quantization}))


def load(model: Path | str, precision: str = "auto") -> tuple[Decider, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decider = Decider.load(model, precision=precision, keep_awake=None, pin_memory=False)
    return decider, [str(w.message) for w in caught if issubclass(w.category, LowBitWarning)]


@pytest.mark.parametrize(
    ("mode", "bits", "group", "form"),
    [("affine", 4, 32, "4-bit"), ("affine", 6, 32, "6-bit"), ("nvfp4", 4, 16, "NVFP4"), ("mxfp4", 4, 32, "MXFP4")],
)
def test_a_published_low_bit_build_loads_as_given_and_warns_in_every_answer(
    tmp_path: Path, mode: str, bits: int, group: int, form: str
) -> None:
    quantized_checkpoint(tmp_path, mode, bits, group)
    decider, said = load(tmp_path)
    unmeasured = drift.SAYS["noticeable" if form == "6-bit" else "a lot"]  # every 4-bit build measured moved a lot
    text = f"{tmp_path} runs at {form}. {unmeasured}"
    assert said == [text]
    assert decider.details["drift"] == text
    assert decider.systemone(REQUEST)["warnings"] == [text]
    assert decider.sqwish(DECIDE)["warnings"] == [text]
    decider.close()


def test_quantizing_below_8_bits_warns_with_the_measured_drift_and_8_bits_never_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    qwen3_classifier(tmp_path, intermediate_size=64)
    eight, said = load(tmp_path, precision="q8")
    assert not said
    assert "drift" not in eight.details
    assert "warnings" not in eight.systemone(REQUEST)
    eight.close()
    four, said = load(tmp_path, precision="q4")
    assert said == [f"{tmp_path}:q4 runs at 4-bit. {drift.SAYS['a lot']}"]
    four.close()
    monkeypatch.setitem(drift.MEASURED, f"{tmp_path}:q6", "close")
    measured = f"{tmp_path}:q6 runs at 6-bit. {drift.SAYS['close']}"
    six, said = load(tmp_path, precision="q6")
    assert said == [measured]
    assert six.details["drift"] == measured
    six.close()
    assert main(["decide", str(tmp_path), json.dumps(REQUEST), "--precision", "q6", "--json"]) == 0
    printed = capsys.readouterr()
    assert printed.err.count(measured) == 1
    assert json.loads(printed.out)["warnings"] == [measured]


@pytest.mark.parametrize("named", [False, True])
def test_a_decider_reports_the_precision_it_resolved_by_its_short_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: bool
) -> None:
    """fp16 asked for by its long name, or carried by a short name (kettei-core:fp16) with precision left at auto,
    is reported as "fp16", never as the text the caller gave."""
    qwen3_classifier(tmp_path, intermediate_size=64)
    if named:  # the toy as a short name whose tag carries fp16
        monkeypatch.setattr(names, "precision_of", lambda model: "fp16")
    decider, _ = load(tmp_path, precision="auto" if named else "float16")
    assert decider.precision == "fp16"
    assert decider.answer(parse_request(REQUEST)).precision == "fp16"
    decider.close()


@pytest.mark.parametrize("precision", ["w8a8", "q2", "q3"])
def test_a_precision_ippon_does_not_run_never_opens_a_model(precision: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(InvalidRequest, match=POLICY):
        Decider.load("/does/not/exist", precision=precision)
    assert main(["decide", "/does/not/exist", "{}", "--precision", precision]) == InvalidRequest.exit_code
    assert "--precision auto, bf16, fp16, fp32, q8, q6 or q4" in capsys.readouterr().err
    # asked by name, the refusal lists the names that do run, the precisions ippon computes first
    assert main(["decide", f"kettei-core:{precision}", "{}"]) == UnsupportedModel.exit_code
    assert "use kettei-core:q8, kettei-core:q6, kettei-core:q4 or kettei-core:bf16" in capsys.readouterr().err


@pytest.mark.parametrize("bits", [2, 3])
def test_two_and_three_bit_builds_are_refused_by_preflight_and_loading(bits: int) -> None:
    q = mx.zeros((2, bits * 2), mx.uint32)
    s = mx.ones((2, 1), mx.float16)
    config = {"mode": "affine", "bits": bits, "group_size": 64}
    with pytest.raises(UnsupportedModel, match=POLICY):
        Weights({"x.weight": q, "x.scales": s, "x.biases": s}, quantization=config).matrix("x.weight")
    headers = [
        Tensor("x.weight", "U32", q.shape),
        Tensor("x.scales", "F16", s.shape),
        Tensor("x.biases", "F16", s.shape),
    ]
    with pytest.raises(UnsupportedModel, match=POLICY):
        inspect(headers, config)
    with pytest.raises(UnsupportedModel, match=POLICY):
        quantize(mx.ones((2, 64)), bits)
    with pytest.raises(UnsupportedModel, match=POLICY):
        Stacked(mx.ones((2, 2, 64))).quantized(bits)


def test_formats_ippon_does_not_run_name_the_replacement(tmp_path: Path) -> None:
    path = tmp_path / "unread.gguf"
    write(path, {}, {"x.weight": (np.zeros((2, 256), np.float32), "Q3_K")})
    with pytest.raises(UnsupportedModel, match=POLICY + ".*:Q8_0"):
        locate.open_model(path)
    with pytest.raises(UnsupportedModel, match=POLICY + ".*LiquidAI/d1-2B-GGUF:Q8_0"):
        find("LiquidAI/d1-2B-GGUF:Q3_K_M")
    with pytest.raises(UnsupportedModel, match=POLICY + ".*d3-mini:mxfp8"):
        names.resolve("d3-mini:w8a8")


def test_a_name_carries_its_precision_but_a_creators_build_keeps_its_own() -> None:
    assert names.resolve("kettei-core:q8") == names.resolve("kettei-core")
    assert names.named_precision("Kettei-Core:Q8", "auto") == "q8"
    assert names.named_precision("kettei-core", "auto") == "auto"
    # a creator's own verified 8-bit build is what :q8 takes (Liquid's Q8_0, 2 GB, never 3.8 GB quantized here)
    assert (names.find("d1-2b:q8")[0].name, names.precision_of("d1-2b:q8")) == ("d1-2b:q8_0", None)
    assert names.find("h2o:q8")[0].kind == "safetensors"  # its own Q8_0 unverified: ippon quantizes the checkpoint
    assert names.find("d1-2b:q8_0")[0].kind == "gguf"  # Liquid's own 8-bit file, loaded as published
    assert names.find("clef-flash:fp16")[0].name == "clef-flash:bf16"  # not Cloudflare's 8-bit MLX build
    assert names.precision_of("d1-2b:q8_0") is None
    with pytest.raises(InvalidRequest, match="give one of them"):
        names.named_precision("kettei-core:q8", "bf16")
