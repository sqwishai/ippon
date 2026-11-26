"""Whether a model fits before it loads: sizes from safetensors headers, a refusal naming what fits, auto stepping
down to quantized weights."""

from __future__ import annotations

import pytest

from ippon.admission import admit, resident_bytes
from ippon.errors import InsufficientMemory
from ippon.names import smaller_build
from ippon.safetensors import Tensor

GIB = 1 << 30


def model(gib_fp16: float) -> list[Tensor]:
    """An embedding and weight matrices of about `gib_fp16` GiB in fp16, as headers list them."""
    n = int(gib_fp16 * GIB / 2 / (4096 * 4096))
    return [Tensor("embed", "BF16", (32000, 1024)), *[Tensor(f"w{i}", "BF16", (4096, 4096)) for i in range(n)]]


def test_sizes_follow_the_precision_and_stored_quantized_tensors_count_as_stored() -> None:
    """Floats at the compute width, weight matrices at `bits` (plus a scale and bias per 64; the embedding as it is),
    integer and encoded tensors (packed weights, their scales, global multipliers) as stored at any width, and the
    vocabulary tables a load reads from the file not at all."""
    found = [Tensor("embed", "F32", (1000, 64)), Tensor("w", "BF16", (128, 128)), Tensor("packed", "U32", (128, 32))]
    assert resident_bytes(found, 2) == 1000 * 64 * 2 + 128 * 128 * 2 + 128 * 32 * 4
    q8 = resident_bytes(found, 2, 8)
    assert q8 == 1000 * 64 * 2 + 128 * 128 + (128 * 128 // 64) * 4 + 128 * 32 * 4
    encoded = [
        Tensor("x.weight", "U32", (64, 8)),
        Tensor("x.scales", "U8", (64, 4)),
        Tensor("x.global_scale", "F32", ()),
    ]
    assert resident_bytes(encoded, 2) == resident_bytes(encoded, 4) == 64 * 8 * 4 + 64 * 4 + 4
    tables = [  # read from the file, never held: a checkpoint's (packed too) and a GGUF file's
        Tensor("model.embed_tokens.weight", "BF16", (1000, 64)),
        Tensor("lm_head.weight", "U32", (1000, 16)),
        Tensor("lm_head.scales", "BF16", (1000, 1)),
        Tensor("token_embd.weight.biases", "F16", (1000, 2)),
    ]
    assert resident_bytes([*found, *tables], 2, 8) == q8


def test_a_precision_that_does_not_fit_is_refused_naming_what_fits() -> None:
    """Explicit or auto, a precision past the budget is refused with the precisions that fit; auto never quantizes on
    its own, and an imported build (which can't be quantized again) is pointed at smaller published builds."""
    with pytest.raises(InsufficientMemory, match="Precisions that fit: q8"):
        admit(model(19), "fp16", working_set=16 * GIB)
    assert admit(model(19), "fp16", working_set=None) > 0  # nothing known: no check
    assert admit(model(6), "auto", 16 * GIB) > 0
    with pytest.raises(InsufficientMemory, match=r"q8 quantizes the model's weights, which changes its answers"):
        admit(model(19), "auto", 16 * GIB)
    with pytest.raises(InsufficientMemory, match=r"No precision fits\.$"):
        admit(model(60), "auto", 8 * GIB)
    with pytest.raises(InsufficientMemory, match="Choose a smaller published build") as caught:
        admit(model(19), "auto", 16 * GIB, can_quantize=False)
    assert "q4" not in str(caught.value)
    # a refusal names the creator's smaller build of a named model (d1-2B's Q8_0 file), else where builds are listed
    assert smaller_build("LiquidAI/d1-2B").endswith("ippon pull d1-2b:q8_0.")
    assert "ippon recommend" in smaller_build("/a/folder/of/weights")
