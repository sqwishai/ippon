"""The fused Metal kernels against the MLX compositions they replace (arch/ops.py), on the GPU: the same numbers,
for every dtype, width and layout an architecture passes them, and whole tiny models run either way."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from helpers import assert_mlx_close
from ippon.arch.ops import FUSED, MLX

pytestmark = [pytest.mark.metal, pytest.mark.usefixtures("gpu")]

TOLERANCE = {mx.float32: 2e-5, mx.float16: 2e-3, mx.bfloat16: 1.6e-2}  # about one unit of the dtype's last place


def close(a: mx.array, b: mx.array, dtype: mx.Dtype) -> None:
    assert_mlx_close(a, b, atol=TOLERANCE[dtype], rtol=TOLERANCE[dtype])


@pytest.mark.parametrize(
    ("dtype", "width", "kind", "groups"),
    [(d, w, k, 1) for d in (mx.float32, mx.float16, mx.bfloat16) for w in (64, 100, 4100) for k in ("rms", "layer")]
    + [(d, w, "rms", g) for d in (mx.float32, mx.bfloat16) for w, g in ((2560, 2), (256, 4))],
)
def test_norms_equal_the_mlx_composition(dtype: mx.Dtype, width: int, kind: str, groups: int) -> None:
    """Partial threadgroups, multiple passes and groups with different scales; residual rounding stays exact."""
    grouped = groups > 1
    mx.random.seed(groups if grouped else width)
    shape = (2, 5, width) if grouped else (3, 7, width)
    x = (mx.random.normal(shape) * 3 + (mx.arange(width) / width if grouped else 0)).astype(dtype)
    delta = mx.random.normal(shape).astype(dtype)
    w = mx.random.normal((width,)) * 0.5 + 1.0
    epsilon = 1e-6 if grouped else 1e-5
    h1, n1 = FUSED.add_norm(x, delta, w, epsilon, kind, groups=groups)
    h2, n2 = MLX.add_norm(x, delta, w, epsilon, kind, groups=groups)
    assert mx.array_equal(h1, h2)
    close(n1, n2, dtype)
    close(FUSED.norm(x, w, 1e-6, kind, groups=groups), MLX.norm(x, w, 1e-6, kind, groups=groups), dtype)


def test_a_strided_input_is_read_as_its_values() -> None:
    q = mx.random.normal((2, 5, 4, 128)).astype(mx.bfloat16)[..., :64]  # a head's first half, as Qwen3.5 slices q
    w = mx.random.normal((64,)) + 1.0
    close(FUSED.norm(q, w, 1e-6), MLX.norm(q, w, 1e-6), mx.bfloat16)


def test_whole_models_match_the_numpy_models_with_and_without_the_fused_kernels() -> None:
    """In a process with M4-family kernels, whose fp32 matmuls are exact (the M5's round to about TF32)."""
    import json
    import os
    import subprocess
    import sys

    script = Path(__file__).with_name("fused_models.py")
    env = {**os.environ, "MLX_METAL_GPU_ARCH": "applegpu_g16s"}
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, check=True)
    for arch, errors in json.loads(out.stdout.strip().splitlines()[-1]).items():
        assert errors["mlx"] < 1e-4, (arch, errors)
        assert errors["fused"] < 1e-4, (arch, errors)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("silu", [True, False])
@pytest.mark.parametrize("taps", [3, 4])
def test_causal_conv_equals_the_mlx_composition(dtype: mx.Dtype, silu: bool, taps: int) -> None:
    mx.random.seed(taps)
    channels, extra = 96, 40  # the convolution reads the first 96 columns of 136-wide rows
    proj = mx.random.normal((3, 11, channels + extra)).astype(dtype)
    w = mx.random.normal((channels, taps))
    for window in (mx.random.normal((1, taps - 1, channels)).astype(dtype), mx.random.normal((3, taps - 1, channels))):
        close(FUSED.causal_conv(proj, window, w, silu), MLX.causal_conv(proj, window, w, silu), dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_gated_conv_equals_the_mlx_composition(dtype: mx.Dtype) -> None:
    mx.random.seed(1)
    d = 64
    proj = mx.random.normal((2, 13, 3 * d)).astype(dtype)
    w = mx.random.normal((d, 3))
    window = mx.random.normal((1, 2, d)).astype(dtype)
    close(FUSED.gated_conv(proj, window, w), MLX.gated_conv(proj, window, w), dtype)
    close(FUSED.gated_conv(proj, None, w), MLX.gated_conv(proj, None, w), dtype)


@pytest.mark.parametrize(
    ("dtype", "weight"), [(mx.bfloat16, mx.bfloat16), (mx.float16, mx.float32), (mx.bfloat16, mx.float32)]
)
def test_gated_norm_equals_the_mlx_composition(dtype: mx.Dtype, weight: mx.Dtype) -> None:
    """The norm weight in the compute dtype, or in fp32 as some checkpoints store it: w * normed then stays fp32,
    unrounded."""
    mx.random.seed(4)
    heads, width, offset = 4, 128, 96
    o = mx.random.normal((2, 7, heads, width)) * 2
    proj = mx.random.normal((2, 7, offset + heads * width + 40)).astype(dtype)  # z sits inside wider rows
    w = (mx.random.normal((width,)) + 1).astype(weight)
    fused, plain = FUSED.gated_norm(o, proj, offset, w, 1e-6), MLX.gated_norm(o, proj, offset, w, 1e-6)
    assert fused.dtype == plain.dtype == dtype
    close(fused, plain, dtype)


def test_no_kernel_is_compiled_again_for_another_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every request has its own length; a length among a kernel's template constants would compile the kernel
    again for each (about 60 ms on an M5 Pro). Lengths and row counts go in as inputs."""
    from ippon.kernels import conv, norm

    seen: dict[str, set[str]] = {}

    def recording(name: str, make: object) -> object:
        def kernel(**kwargs: object) -> object:
            seen.setdefault(name, set()).add(repr(kwargs["template"]))
            return make(**kwargs)  # type: ignore[operator]

        return kernel

    for module, attr in ((conv, "_causal"), (conv, "_gated")):
        real = getattr(module, attr)
        monkeypatch.setattr(module, attr, lambda *a, real=real, attr=attr: recording(attr, real(*a)))
    for length in (5, 9, 17):
        x = mx.random.normal((1, length, 48)).astype(mx.float16)
        mx.eval(conv.causal_conv(x, mx.zeros((1, 3, 32), mx.float16), mx.random.normal((32, 4)), True))
        mx.eval(conv.gated_conv(x, None, mx.random.normal((16, 3))))
        mx.eval(norm.add_norm(mx.random.normal((length, 64)).astype(mx.float16), None, mx.ones((64,)), 1e-6, "rms"))
    assert set(seen) == {"_causal", "_gated"}
    assert all(len(templates) == 1 for templates in seen.values()), seen


@pytest.mark.parametrize(
    ("hk", "hv", "dk", "dv"),
    [
        (8, 16, 128, 128),
        (24, 48, 128, 128),
        (4, 4, 64, 96),
        (16, 32, 128, 128),
        (8, 8, 128, 128),
        (20, 20, 128, 128),
        (16, 16, 64, 64),
        (16, 16, 256, 256),
        (16, 80, 128, 128),  # fallback: too wide, unsupported head ratio
    ],
)
@pytest.mark.parametrize("start", [False, True])
def test_a_padded_recurrence_equals_mlx_on_the_shape_itself(hk: int, hv: int, dk: int, dv: int, start: bool) -> None:
    """Native, padded, split and fallback shapes all preserve the real heads' outputs and recurrence states."""
    key = mx.random.key(hk * hv + dk)
    k1, k2, k3, k4, k5 = mx.random.split(key, 5)
    t = 37
    q = mx.random.normal((1, t, hk, dk), key=k1)
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * dk**-0.5
    k = mx.random.normal((1, t, hk, dk), key=k2)
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = mx.random.normal((1, t, hv, dv), key=k3)
    gamma = mx.exp(-mx.abs(mx.random.normal((1, t, hv), key=k4)) * 0.2)
    beta = mx.sigmoid(mx.random.normal((1, t, hv), key=k5))
    state = mx.random.normal((1, hv, dv, dk), key=key) * 0.1 if start else None
    mx.set_default_device(mx.gpu)
    try:
        want_o, want_s = mx.fast.gated_delta_update(q, k, v, gamma, beta, initial_state=state)
        got_o, got_s = MLX.gated_delta(q, k, v, gamma, beta, state)
        mx.eval(want_o, want_s, got_o, got_s)
    finally:
        mx.set_default_device(mx.cpu)
    assert got_o.shape == want_o.shape
    assert got_s.shape == want_s.shape
    scale = float(mx.abs(want_o).max())
    np.testing.assert_allclose(np.array(got_o), np.array(want_o), atol=3e-3 * scale, rtol=0)
    np.testing.assert_allclose(np.array(got_s), np.array(want_s), atol=3e-3 * float(mx.abs(want_s).max()), rtol=0)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(("hk", "dk", "hv", "dv"), [(4, 64, 8, 96), (2, 48, 2, 48)])
def test_delta_inputs_equal_the_mlx_composition(dtype: mx.Dtype, hk: int, dk: int, hv: int, dv: int) -> None:
    key = mx.random.key(hk * dk)
    mixed = (mx.random.normal((2, 7, 2 * hk * dk + hv * dv), key=key) * 3).astype(dtype)
    proj = (mx.random.normal((2, 7, 40 + 2 * hv), key=mx.random.key(dv)) * 4).astype(dtype)  # [.., z.. | b | a]
    neg_a = -mx.exp(mx.random.normal((hv,), key=mx.random.key(1)))
    dt_bias = mx.random.normal((hv,), key=mx.random.key(2))
    args = (mixed, proj, neg_a, dt_bias, hk, dk, hv, dv, dk**-0.5)
    want, got = list(MLX.delta_inputs(*args)), FUSED.delta_inputs(*args)
    # beta: sigmoid in fp32 rounded once to the dtype (MLX's half and bf16 sigmoids round at every step, a unit or two
    # away)
    want[3] = mx.sigmoid(proj[..., -2 * hv : -hv].astype(mx.float32)).astype(dtype).astype(mx.float32)
    for name, g, w in zip(("q", "k", "v", "beta", "gamma"), got, want, strict=True):
        assert g.dtype == mx.float32, name
        assert g.shape == w.shape, name
        np.testing.assert_allclose(np.array(g), np.array(w), atol=1e-6, rtol=1e-5, err_msg=name)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    ("width", "stride", "dims", "branch"),
    [(128, 128, 128, False), (64, 64, 64, True), (256, 512, 64, True)],
)
def test_norm_rope_equals_the_norm_then_mlx_rope(
    dtype: mx.Dtype, width: int, stride: int, dims: int, branch: bool
) -> None:
    """kernels/norm.norm_rope against the composition it replaces (Ops.norm_rope with the switch off): a dense
    model's queries (stride = width), Qwen3.5's (each head's query then its gate: stride 2 x width, partial rotary),
    a prefill from an offset and a branch's rows at their own positions."""
    from ippon.arch.ops import Ops
    from ippon.arch.rope import RoPE

    mx.random.seed(5)
    heads, n, offset = 3, 9, 32
    proj = (mx.random.normal((1 if branch else 2, n, offset + heads * stride + 16)) * 3).astype(dtype)
    w = mx.random.normal((width,)) * 0.5 + 1
    positions = mx.array([4, 5, 9, 9, 10, 20, 21, 22, 23]) if branch else None
    rope = RoPE(dims, 1e6)
    plain = Ops(norm_kernel=False).norm_rope(proj, offset, stride, heads, width, w, 1e-6, 7, positions, rope)
    fused = Ops(norm_rope_kernel=True).norm_rope(proj, offset, stride, heads, width, w, 1e-6, 7, positions, rope)
    assert fused.shape == plain.shape
    assert fused.dtype == plain.dtype == dtype
    close(fused, plain, dtype)
