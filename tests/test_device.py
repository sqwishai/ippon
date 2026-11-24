from __future__ import annotations

import pytest

from ippon.device import Device, detect, macos_version
from ippon.pressure import Reading


def device(architecture: str | None, macos: tuple[int, ...] = (27, 0), gpu: bool = True) -> Device:
    return Device(
        name="Apple M", architecture=architecture, memory_bytes=None, working_set_bytes=None, macos=macos, gpu=gpu
    )


@pytest.mark.parametrize(
    ("architecture", "macos", "gpu", "expected"),
    [
        ("applegpu_g17g", (26, 2), True, True),  # M5, from macOS 26.2
        ("applegpu_g17s", (26, 1), True, False),  # before macOS 26.2
        ("applegpu_g16s", (27, 0), True, False),  # M4 generation and before
        ("applegpu_g17p", (27, 0), True, False),  # phone parts need g18
        ("applegpu_g18p", (27, 0), True, True),
        ("applegpu_g17s", (27, 0), False, False),  # no GPU
        (None, (27, 0), True, False),
    ],
)
def test_neural_accelerators(architecture: str | None, macos: tuple[int, ...], gpu: bool, expected: bool) -> None:
    assert device(architecture, macos, gpu).neural_accelerators is expected


def test_macos_version() -> None:
    assert macos_version("26.2.1") == (26, 2, 1)
    assert macos_version("") == (0,)


def test_detect_reads_the_gpu_while_the_default_device_is_the_cpu() -> None:
    d = detect()  # the tests' default device is the CPU (conftest.py)
    if not d.gpu:
        pytest.skip("no Apple GPU")
    if d.generation is None and not (d.architecture or "").startswith("applegpu_"):
        pytest.skip(f"a virtual GPU ({d.architecture}): GitHub's macOS runners are VMs")
    assert d.architecture is not None
    assert d.architecture.startswith("applegpu_")
    assert d.memory_bytes


def test_the_limits_cover_the_model_and_scale_with_the_mac() -> None:
    """The wired limit covers the model within 90% of the working set and half the Mac's memory; MLX's cache
    keeps 10% of the working set (1 GiB at least), never more than a lower limit already set."""
    from ippon.residency import cache_limit, wired_limit

    gib = 1 << 30
    assert wired_limit(40 * gib, 10 * gib) == 18 * gib  # the model and 8 GB of room
    assert wired_limit(20 * gib, 15 * gib) == 18 * gib  # at most 90% of the working set
    assert wired_limit(64 * gib, 50 * gib, memory=48 * gib) == 24 * gib  # a raised working set
    assert wired_limit(12 * gib, int(9.7 * gib), memory=16 * gib) == 8 * gib  # Torchcast on a 16 GB Mac, not 10.8 GB

    def now(available: float, kernel: int = 1, memory: int = 48) -> Reading:
        return Reading(0, memory * gib, int(available * gib), 0, kernel)

    assert wired_limit(40 * gib, 10 * gib, memory=48 * gib, reading=now(30)) == 18 * gib  # plenty: as before
    assert wired_limit(40 * gib, 10 * gib, memory=48 * gib, reading=now(9.76)) == 14 * gib  # 4 GB above the line
    assert wired_limit(40 * gib, 10 * gib, memory=48 * gib, reading=now(5)) == 0  # under the line: nothing wired
    assert wired_limit(40 * gib, 10 * gib, memory=48 * gib, reading=now(30, kernel=2)) == 0  # macOS says warning
    assert wired_limit(12 * gib, int(9.7 * gib), memory=16 * gib, reading=now(1.5, memory=16)) == 0  # the panic's Mac
    assert cache_limit(40 * gib) == gib  # the M5 Pro's 48 GB: the cap, not 10% of it
    assert cache_limit(int(10.7 * gib)) == gib  # a 16 GB Mac: the cap
    assert cache_limit(int(5.3 * gib)) == int(0.53 * gib)  # an 8 GB Mac: 10% of its working set
    assert cache_limit(40 * gib, lower=gib // 2) == gib // 2  # a lower limit set before stays


@pytest.mark.metal
def test_keep_awake_waits_for_the_model_and_rests_after_its_window() -> None:
    import threading
    import time

    from ippon.residency import KeepAwake

    lock = threading.Lock()
    with lock:  # the model is running: no beat
        awake = KeepAwake(lock, period=0.01, window=0.3)
        time.sleep(0.1)
        assert awake.beats == 0
    time.sleep(0.4)  # beats, then the window after the start closes
    rested = awake.beats
    assert rested > 0
    time.sleep(0.1)
    assert awake.beats == rested
    awake.busy()  # an answer: it beats again
    time.sleep(0.1)
    awake.close()
    assert awake.beats > rested
    with pytest.raises(ValueError, match="seconds"):
        KeepAwake(lock, period=0)


def test_an_m5_max_is_its_own_chip_though_mlx_reports_the_pros_family(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx

    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    for reported, emulated in (("applegpu_g17s", None), ("applegpu_g16s", "applegpu_g16s")):
        info = {"device_name": "Apple M5 Max", "architecture": reported}
        monkeypatch.setattr(mx, "device_info", lambda _device, info=info: info)
        d = detect()
        assert (d.architecture, d.tier, d.emulated_architecture) == ("applegpu_g17c", "Max", emulated)
