"""`ippon profile`: a request timed step by step (`--phases`, on the toy model), and every operation timed under
its label."""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import pytest

from ippon import cli
from ippon.adapter import Loaded
from ippon.arch.ops import for_switches
from ippon.decider import Decider
from ippon.kernels import qgemm
from ippon.profile import PHASES, TimedOps, matrix_label, phases
from ippon.profiles import Switches
from ippon.quant import QuantizedMatrix
from toy import ToyBackbone, ToyFormat
from toy import decider as toy


def body(state: str, question: str) -> dict[str, object]:
    return {"state": state, "questions": {"q": {"type": "noul", "instructions": question}}}


@pytest.mark.parametrize("foreign", [False, True])
def test_phases_report_each_step_without_changing_answers(foreign: bool) -> None:
    backbone = ToyBackbone()
    fmt = ToyFormat(backbone)
    if foreign:
        fmt.tokenizer = type("ForeignTokenizer", (), {"backend": "tokenizers"})()
    with Decider(Loaded(backbone, fmt)) as decider:
        requests = [body("a b c", "x"), body("d e f", "y")] if foreign else [body("a b c", q) for q in ("x", "y", "z")]
        with toy() as reference:
            expected = reference.systemone(requests[0])
        scenario = "fresh" if foreign else "changed"
        found = phases(decider, requests, scenario)
        assert set(found.median) == set(found.first) == set(found.cpu) == set(PHASES)
        assert all(ms >= 0 for ms in found.median.values())
        assert found.input_tokens > 0
        assert found.table().startswith(f"{scenario}: ")
        assert all(name in found.table() for name in PHASES)
        assert decider.systemone(requests[0]) == expected
        if foreign:
            assert found.tokenize_ms == 0


@pytest.mark.metal
@pytest.mark.nax
def test_a_packed_gate_up_with_its_own_fused_tile_is_timed_as_one_operation(gpu: None) -> None:
    """A quantized [gate; up]'s fused GLU (choices.py's tile, no full-precision one) is timed under its own label,
    not left out of the table."""
    n, k = 512, 1024
    packed, scales, biases = mx.quantize((mx.random.normal((n, k)) * 0.05).astype(mx.float16), group_size=64, bits=8)
    tile = next(t for t in qgemm.CANDIDATES if qgemm.glu_fits(t, n, k, 8, 64))
    gate_up = QuantizedMatrix(packed, scales, biases, 8, 64, glu=("silu", ((1, None), (16, tile))))
    timed = TimedOps.of(for_switches(Switches(), {}))
    timed.glu_mm(mx.random.normal((64, k)).astype(mx.float16), gate_up)
    assert dict(timed.calls) == {f"{matrix_label(gate_up)} + silu (fused)": 1}


def test_phases_time_the_request_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps(body("a b c", "x")))
    monkeypatch.setattr(cli, "load", lambda args: toy())
    model = str(tmp_path)  # a folder: the toy decider stands in for loading it
    assert cli.main(["profile", model, "--phases", "--request", str(request), "--repeat", "2"]) == 0
    found = re.search(r"(\d+) input tokens", capsys.readouterr().out)
    assert found is not None
    read = int(found.group(1))
    assert read < 100  # the three-word request, not bench's 1,000-word one
    assert cli.main(["profile", model, "--phases", "--request", str(request), "--scenario", "changed"]) != 0
