from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ippon.device import Device
from ippon.doctor import Check, checks, report
from ippon.errors import RuntimeUnavailable
from ippon.profiles import resolve

GIB = 1 << 30


def m5(macos: tuple[int, ...] = (27, 0)) -> Device:
    return Device("Apple M5 Pro", "applegpu_g17s", 48 * GIB, 36 * GIB, macos, True)


def statuses(found: list[Check]) -> dict[str, str]:
    return {c.what: c.status for c in found}


@pytest.fixture(autouse=True)
def _no_profile_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IPPON_PROFILE", raising=False)


def test_a_mac_at_full_speed_has_nothing_to_change() -> None:
    found = checks(m5(), resolve(device=m5()), "0.32.4", {})
    assert all(c.status == "ok" for c in found), report(found)
    assert any("up to ~16B in bf16" in c.what for c in found)  # admission.budget(36 GiB) / 2 bytes


def test_an_untested_mlx_an_old_macos_and_a_broken_adapter_are_named_with_their_fix() -> None:
    device = m5(macos=(26, 1))
    found = checks(device, resolve("m4", device=device), "0.31.0", {"acme/x": "ImportError: no acme"})
    warned = {c.what.split(":")[0] for c in found if c.status == "warn"}
    assert {"MLX 0.31.0 is untested here", "macOS 26.1"} <= warned
    assert not any("is forced" in c.what for c in found)  # an M5 before macOS 26.2 runs the M4's paths by default
    assert [c.what for c in found if c.status == "fail"] == ["adapter 'acme/x' failed to load: ImportError: no acme"]
    assert "-> update to macOS 26.2 or later" in report(found)
    # read as PEP 440 versions: a development build before the lowest tested release is untested too
    assert [c.status for c in checks(device, resolve("m4", device=device), "0.32.4.dev1", {}) if "MLX" in c.what] == [
        "warn"
    ]
    assert [c.status for c in checks(device, resolve("m4", device=device), "0.32.4", {}) if "MLX" in c.what] == ["ok"]


def test_no_gpu_and_no_mlx_fail() -> None:
    cpu = Device(None, None, None, None, (0,), False)
    found = checks(cpu, resolve(device=cpu), None, {})
    assert sum(c.status == "fail" for c in found) == 2


def test_an_m5_before_macos_26_2_runs_the_m4_paths_and_doctor_explains_a_forced_m5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from ippon import doctor

    old = m5(macos=(26, 1))
    assert (resolve(device=old).name, resolve(device=old).forced) == ("m4", False)
    monkeypatch.setattr(doctor, "detect", lambda: old)
    monkeypatch.setenv("IPPON_PROFILE", "m5")
    found = doctor.run()  # it used to fall back to m5 and fail again
    failed = [c.what for c in found if c.status == "fail"]
    assert any("macOS 26.2" in w for w in failed)
    assert any(c.what.startswith("macOS 26.1") and c.status == "warn" for c in found)


def run_without_mlx(code: str) -> subprocess.CompletedProcess[str]:
    """`code` in a fresh Python where `import mlx` fails, as under Rosetta or with a broken install."""
    blocked = "import sys; sys.modules['mlx'] = None; sys.modules['mlx.core'] = None\n"
    return subprocess.run(
        [sys.executable, "-c", blocked + code], capture_output=True, text=True, timeout=60, check=False
    )


def test_help_and_doctor_run_where_mlx_does_not_load() -> None:
    """With `import mlx` failing, importing ippon, its command line and what `ippon mcp` and `recommend` read before a
    model loads still works (MLX is imported only where it is used), --help and doctor run, and a command that needs
    MLX says to run doctor."""
    imported = run_without_mlx("import ippon.admission, ippon.mcp, ippon.recommend")
    assert imported.returncode == 0, imported.stderr
    helped = run_without_mlx("from ippon.cli import main\ntry:\n    main(['--help'])\nexcept SystemExit:\n    pass")
    assert "ippon" in helped.stdout, helped.stderr
    doctored = run_without_mlx("from ippon.cli import main\nraise SystemExit(main(['doctor']))")
    assert doctored.returncode == 1
    assert "MLX doesn't load" in doctored.stdout, doctored.stderr
    request = '{"state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}'
    ran = run_without_mlx(
        f"from ippon.__main__ import main\nraise SystemExit(main(['run', 'org/name', {request!r}, '--json']))"
    )
    assert ran.returncode == RuntimeUnavailable.exit_code
    assert "ippon doctor" in ran.stderr
    assert json.loads(ran.stdout)["error"]["type"] == "runtime_unavailable"
    imported = run_without_mlx("from ippon import Decider")
    assert "RuntimeUnavailable" in imported.stderr


def test_a_missing_startup_dependency_reports_json_without_a_traceback() -> None:
    code = (
        "import sys; sys.modules['numpy'] = None\nfrom ippon.__main__ import main\n"
        "raise SystemExit(main(['doctor', '--json']))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == RuntimeUnavailable.exit_code
    assert json.loads(result.stdout)["error"]["type"] == "runtime_unavailable"
    assert "reinstall" in result.stderr
    assert "numpy" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("contents", [None, b"not a dynamic library"])
def test_doctor_detects_a_missing_or_broken_native_library(tmp_path: Path, contents: bytes | None) -> None:
    source = Path(__file__).resolve().parents[1] / "src/ippon"
    package = tmp_path / "ippon"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__", "*.dylib"))
    path = package / "libippon_native.dylib"
    if contents is not None:
        path.write_bytes(contents)
    result = subprocess.run(
        [sys.executable, "-m", "ippon", "doctor", "--json"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    checks = json.loads(result.stdout)
    assert any(c["status"] == "fail" and "native library" in c["what"] and c["fix"] for c in checks)
    assert "Traceback" not in result.stderr
