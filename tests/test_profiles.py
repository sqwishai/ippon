from __future__ import annotations

import pytest

from ippon import UnsupportedProfile
from ippon._emulate import architecture_for
from ippon.device import Device, chip_architecture
from ippon.profiles import resolve


def mac(architecture: str | None, name: str = "Apple M", macos: tuple[int, ...] = (27, 0), gpu: bool = True) -> Device:
    return Device(name=name, architecture=architecture, memory_bytes=None, working_set_bytes=None, macos=macos, gpu=gpu)


@pytest.mark.parametrize(
    ("architecture", "generation", "tier", "profile"),
    [
        ("applegpu_g13g", 1, "base", "m1"),
        ("applegpu_g16d", 4, "Ultra", "m4"),
        ("applegpu_g17s", 5, "Pro", "m5"),
    ],
)
def test_the_chip_picks_its_profile(
    architecture: str, generation: int, tier: str, profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IPPON_PROFILE", raising=False)
    device = mac(architecture)
    assert device.generation == generation
    assert device.tier == tier
    p = resolve(device=device)
    assert p.name == profile
    assert not p.forced


def test_a_lower_profile_can_be_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    p = resolve("m2", device=mac("applegpu_g17s"))
    assert (p.name, p.detected, p.forced) == ("m2", "m5", True)
    monkeypatch.setenv("IPPON_PROFILE", "m3")
    assert resolve(device=mac("applegpu_g17s")).name == "m3"
    assert resolve("m4", device=mac("applegpu_g17s")).name == "m4"  # the argument wins over the environment


@pytest.mark.parametrize(
    ("override", "device", "message"),
    [
        ("m5", mac("applegpu_g16s", "Apple M4 Pro"), "M5-generation chip"),
        ("m9", mac("applegpu_g17s"), "unknown profile"),
        ("m5", mac("applegpu_g17s", macos=(26, 1)), "macOS 26.2"),
    ],
)
def test_impossible_profiles_are_refused(override: str, device: Device, message: str) -> None:
    with pytest.raises(UnsupportedProfile, match=message):
        resolve(override, device=device)


def test_no_apple_gpu_runs_the_portable_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IPPON_PROFILE", raising=False)
    p = resolve(device=mac(None, gpu=False))
    assert p.name == "m1"
    assert p.detected is None


def test_a_chip_names_its_gpu_family_and_only_an_older_ones_is_emulated() -> None:
    assert chip_architecture("Apple M1") == "applegpu_g13g"
    assert chip_architecture("Apple M5 Pro") == "applegpu_g17s"
    assert chip_architecture("Intel(R) Core(TM) i7") is None
    assert architecture_for("m4", "Apple M5 Pro") == "applegpu_g16s"
    assert architecture_for(" M1 ", "Apple M3 Max") == "applegpu_g13c"
    assert architecture_for("m5", "Apple M5 Pro") is None  # not below the chip
    assert architecture_for("m3", "Apple M3") is None
    assert architecture_for("m2", None) is None


def test_a_profile_says_whether_mlx_runs_its_chips_kernels_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IPPON_PROFILE", raising=False)
    plain = mac("applegpu_g17s", "Apple M5 Pro")
    emulating = Device(
        "Apple M5 Pro", "applegpu_g17s", None, None, (27, 0), True, emulated_architecture="applegpu_g16s"
    )
    assert not resolve("m4", device=plain).emulated
    assert resolve("m4", device=emulating).emulated
    assert not resolve("m3", device=emulating).emulated  # MLX takes the M4's kernels, not the M3's


def test_switches_given_to_load_override_the_environments(monkeypatch: pytest.MonkeyPatch) -> None:
    m5 = mac("applegpu_g17s", "Apple M5 Pro")
    monkeypatch.setenv("IPPON_SWITCHES", "add_norm=mlx,conv=mlx")
    p = resolve(device=m5, switched="conv=kernel,gemm=mlx")
    assert (p.switches.add_norm, p.switches.conv, p.switches.gemm) == ("mlx", "kernel", "mlx")
    with pytest.raises(UnsupportedProfile, match="unknown switch"):
        resolve(device=m5, switched="attention=fast")
