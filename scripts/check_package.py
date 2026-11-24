"""Check a built wheel in fresh Python environments, including broken dependencies and server recovery.

    uv run python scripts/check_package.py --wheel dist/ippon-VERSION.whl --out /tmp/ippon-package-checks --tests

Defaults to Python 3.11 through 3.15, one at a time. Dependencies follow uv.lock. Reports and subprocess logs remain
under --out, temporary environments are removed. --tests also runs the full suite against the installed wheel.
Use --install-only to check installation and dependency repair without starting inference or servers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[str], log: Path, *, cwd: Path, expected: int = 0, timeout: float = 300) -> str:
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "IPPON_PROFILE", "IPPON_API_KEY"):
        environment.pop(name, None)
    environment["MLX_ENABLE_TF32"] = "0"  # the tiny NumPy fixture checks precise fp32 on M5 too
    with subprocess.Popen(
        arguments,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            process.terminate()  # the lifecycle worker stops its detached servers before exiting
            try:
                stdout, stderr = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            with log.open("a") as output:
                output.write(f"$ {arguments!r}\ninterrupted\n{stdout}\n{stderr}\n")
            raise
    with log.open("a") as output:
        output.write(f"$ {arguments!r}\n{stdout}\n{stderr}\nexit={process.returncode}\n")
    if process.returncode != expected:
        raise AssertionError(f"{arguments[0]} exited {process.returncode}, expected {expected}, see {log}")
    return stdout


def environment_check(args: argparse.Namespace, requested: str, fixture: Path, requirements: Path) -> dict[str, Any]:
    log = args.out / f"python-{requested}.log"
    log.write_text("")
    report: dict[str, Any] = {
        "requested_python": requested,
        "passed": False,
        "scope": "installation" if args.install_only else "lifecycle",
        "wheel_sha256": args.wheel_sha256,
    }
    with tempfile.TemporaryDirectory(prefix="ippon-package-") as temporary:
        scratch = Path(temporary)
        environment = scratch / "venv"
        python, ippon = str(environment / "bin/python"), str(environment / "bin/ippon")

        def call(arguments: list[str], expected: int = 0, timeout: float = 300) -> str:
            return run(arguments, log, cwd=scratch, expected=expected, timeout=timeout)

        def checked(arguments: list[str], code: int, kind: str) -> None:
            result = json.loads(call(arguments, code))
            assert result["error"]["type"] == kind, result
            assert result["error"]["exit_code"] == code, result

        try:
            call(["uv", "venv", str(environment), "--python", requested])
            call(["uv", "pip", "install", "--python", python, "-r", str(requirements), str(args.wheel)])
            installed = json.loads(
                call(
                    [
                        python,
                        "-I",
                        "-c",
                        "import importlib.metadata as m, json, platform, ippon; "
                        "print(json.dumps({'python': platform.python_version(), 'package': ippon.__file__, "
                        "'dependencies': {d.metadata['Name']: d.version for d in m.distributions()}}))",
                    ]
                )
            )
            assert Path(installed["package"]).resolve().is_relative_to(environment.resolve()), installed
            report.update(installed)
            request = json.loads((fixture / "reference.json").read_text())["requests"][0]["request"]
            config = fixture / "config.json"
            original = config.read_bytes()
            try:
                config.write_text("{")
                checked([ippon, "decide", str(fixture), json.dumps(request), "--json"], 4, "unsupported_model")
            finally:
                config.write_bytes(original)
            for package in ("mlx", "numpy"):
                pinned = installed["dependencies"][package]
                try:
                    call(["uv", "pip", "uninstall", "--python", python, package])
                    for entry in ([ippon], [python, "-I", "-m", "ippon"]):
                        checked(
                            [*entry, "decide", str(fixture), json.dumps(request), "--json"], 14, "runtime_unavailable"
                        )
                        doctor = json.loads(call([*entry, "doctor", "--json"], 1 if package == "mlx" else 14))
                        if package == "mlx":
                            assert any(check["status"] == "fail" and check["fix"] for check in doctor), doctor
                            call([*entry, "--help"])
                        else:
                            assert doctor["error"]["type"] == "runtime_unavailable", doctor
                finally:
                    call(["uv", "pip", "install", "--python", python, f"{package}=={pinned}"])
            native = Path(installed["package"]).parent / "libippon_native.dylib"
            original_native = native.read_bytes()
            try:
                for contents in (None, b"broken dynamic library"):
                    if contents is None:
                        native.unlink()
                    else:
                        native.write_bytes(contents)
                    doctor = json.loads(call([ippon, "doctor", "--json"], 1))
                    assert any(check["status"] == "fail" and "native library" in check["what"] for check in doctor), (
                        doctor
                    )
            finally:
                native.write_bytes(original_native)
            call([ippon, "--version"])
            call([ippon, "--help"])
            if not args.install_only:  # a runner without an Apple GPU fails doctor's GPU check by design
                doctor = json.loads(call([ippon, "doctor", "--json"]))
                assert all(check["status"] != "fail" for check in doctor), doctor

                call(
                    [
                        python,
                        "-I",
                        str(ROOT / "scripts/serving_lifecycle.py"),
                        str(fixture),
                        "--reference",
                        str(fixture / "reference.json"),
                        "--precision",
                        "fp32",
                        "--seconds",
                        str(args.seconds),
                        "--out",
                        str(args.out / f"python-{requested}-lifecycle.json"),
                    ],
                    timeout=args.seconds + 300,
                )
            if args.tests:
                run([python, "-I", "-m", "pytest", "-q", str(ROOT / "tests")], log, cwd=ROOT, timeout=1800)
            report.update(passed=True, serving=not args.install_only, full_suite=args.tests)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            (args.out / f"python-{requested}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--python", action="append", choices=("3.11", "3.12", "3.13", "3.14", "3.15"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--install-only", action="store_true", help="check installation and repair without inference")
    args = parser.parse_args()
    args.wheel, args.out = args.wheel.resolve(), args.out.resolve()
    if not args.wheel.is_file() or args.seconds < 1:
        parser.error("give an existing wheel and at least one second of load")
    if args.install_only and args.tests:
        parser.error("--install-only excludes --tests, since the full suite includes GPU work")
    args.wheel_sha256 = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    requirements = args.out / "requirements.txt"
    export = ["uv", "export", "--frozen", "--no-emit-project", "--no-hashes", "--output-file", str(requirements)]
    export += ["--all-extras", "--group", "dev"] if args.tests else ["--no-dev"]
    run(export, args.out / "setup.log", cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="ippon-package-fixture-") as temporary:
        fixture = Path(temporary)
        run([sys.executable, str(ROOT / "tests/package_fixture.py"), str(fixture)], args.out / "setup.log", cwd=ROOT)
        reports = []
        for requested in args.python or ["3.11", "3.12", "3.13", "3.14", "3.15"]:
            print(f"Checking installed package on Python {requested}", flush=True)
            report = environment_check(args, requested, fixture, requirements)
            reports.append(report)
            print(
                json.dumps(
                    {
                        "python": requested,
                        "scope": report["scope"],
                        "passed": report["passed"],
                        "error": report.get("error"),
                    }
                ),
                flush=True,
            )
    (args.out / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    raise SystemExit(0 if all(report["passed"] for report in reports) else 1)


if __name__ == "__main__":
    main()
