"""Deterministic inputs shared by tests; numerical reference implementations remain independent."""

from __future__ import annotations

import base64
import hashlib

import mlx.core as mx
import numpy as np

from ippon.profiles import Switches

# Hermes 0.21's `hermes mcp add NAME --env K=V ... --command C --args A ...` and `hermes mcp remove NAME`, as a fake
# on PATH ({python} formatted in): each asks on stdin before it changes config.yaml (under HERMES_HOME, else
# ~/.hermes) and cancels without a yes, and removing a server it doesn't have says so and changes nothing. A fake
# that also starts a chat continues after this block
FAKE_HERMES_MCP = """#!{python}
import os, sys
from pathlib import Path
import yaml
if sys.argv[1:2] == ["mcp"]:
    path = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes") / "config.yaml"
    config = (yaml.safe_load(path.read_text()) if path.exists() else None) or {{}}
    servers = config.setdefault("mcp_servers", {{}})
    verb, name = sys.argv[2], sys.argv[3]
    if verb == "remove" and name not in servers:
        sys.exit(print(f"Server {{name!r}} not found in config."))
    if sys.stdin.readline().strip().lower() not in ("y", "yes"):
        sys.exit(print("Cancelled."))
    if verb == "add":
        at = sys.argv.index
        told = dict(pair.split("=", 1) for pair in sys.argv[at("--env") + 1 : at("--command")])
        servers[name] = {{"command": sys.argv[at("--command") + 1], "args": sys.argv[at("--args") + 1 :], "env": told}}
    else:
        del servers[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    sys.exit(0)
"""

# every operation on MLX's own path: the reference a test turns single switches on over
MLX_SWITCHES = Switches(
    add_norm="mlx",
    conv="mlx",
    gemm="mlx",
    gated_delta="mlx",
    delta_inputs="mlx",
    attention="mlx",
    pipeline="off",
    compile="off",
)


def b64(data: bytes) -> str:
    """Bytes as a request carries them: base64 text."""
    return base64.b64encode(data).decode()


def tokens(n: int, seed: int, low: int = 0) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(low, 64, n)]


def encoder_tokens(n: int, seed: int) -> list[int]:
    return tokens(n, seed, low=3)


def pattern(height: int, width: int, seed: int) -> np.ndarray:
    """The fixture generators' hashed-noise image, using portable uint32 arithmetic."""
    y = np.arange(height, dtype=np.uint32)[:, None, None]
    x = np.arange(width, dtype=np.uint32)[None, :, None]
    c = np.arange(3, dtype=np.uint32)[None, None, :]
    h = (x * np.uint32(73856093)) ^ (y * np.uint32(19349663)) ^ (c * np.uint32(83492791)) ^ np.uint32(seed)
    h = (h ^ (h >> np.uint32(15))) * np.uint32(2246822519)
    image: np.ndarray = ((h >> np.uint32(24)) & np.uint32(255)).astype(np.uint8)
    return image


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require_gemm() -> None:
    """Skip absent hardware; a supported kernel that fails its self-check is a regression."""
    import pytest

    from ippon.device import detect
    from ippon.kernels import gemm

    if not detect().neural_accelerators:
        pytest.skip("needs an M5's neural accelerators")
    assert gemm.available(), "matmul failed compilation or its numerical self-check"


def assert_mlx_equal(actual: mx.array, expected: mx.array) -> None:
    """Compare every value exactly, including bf16 arrays NumPy cannot represent directly."""
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)))


def assert_mlx_close(
    actual: mx.array, expected: mx.array, *, atol: float, rtol: float = 1e-7, err_msg: str = ""
) -> None:
    """Keep a test's explicit bounds while comparing MLX arrays through NumPy's diagnostics."""
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
    )
