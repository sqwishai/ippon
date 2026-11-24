"""Unit tests run on the CPU, so they give the same results on any Mac and in CI and leave the GPU free
for benchmarks; tests marked `metal` need the GPU and are skipped without one, and those marked `nax` need an M5's
neural accelerators and are skipped without them, as on GitHub's
macOS runners (an M1 or M2 Pro in a VM)."""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import pytest

from ippon import pressure
from ippon.device import detect

mx.set_default_device(mx.cpu)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu = mx.metal.is_available()
    nax = gpu and detect().neural_accelerators
    for item in items:
        if "metal" in item.keywords and not gpu:
            item.add_marker(pytest.mark.skip(reason="needs an Apple GPU"))
        elif "nax" in item.keywords and not nax:
            item.add_marker(pytest.mark.skip(reason="needs an M5's neural accelerators (macOS 26.2 or later)"))


@pytest.fixture(autouse=True)
def own_cache(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test's ippon cache (remembered hashes, kept prefixes, licences) apart from the user's; a test that sets
    IPPON_CACHE itself overrides it."""
    monkeypatch.setenv("IPPON_CACHE", str(tmp_path_factory.mktemp("ippon-cache")))


@pytest.fixture(autouse=True)
def any_free_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loads in tests wire by the fixed bounds alone (residency.wired_limit reads no memory pressure): a test's answer
    must not depend on how busy the Mac running it is."""
    monkeypatch.setattr(pressure, "now", lambda: None)


@pytest.fixture
def gpu() -> Iterator[None]:
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)
