"""Builds ippon's native library (native/, Rust) into every wheel, and records the git
revision in wheels built from a checkout (src/ippon/REVISION), so a regular install, which has no checkout to ask,
still says which commit it runs: bench and suite results carry it (bench.revision).

The library is plain C ABI loaded through ctypes, with no Python in it, so one wheel serves every Python ippon
supports: py3-none, Apple silicon, the macOS the library is built for. An editable install gets it copied into
src/ippon, where the checkout's package looks for it."""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

LIBRARY = "libippon_native.dylib"
MACOS = "11.0"  # the oldest macOS on Apple silicon; the wheel's tag says it


class BuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        self._dir: str | None = None
        if self.target_name != "wheel":
            return
        built = self._native()
        if version == "editable":
            shutil.copy2(built / LIBRARY, Path(self.root) / "src/ippon" / LIBRARY)
            return  # an editable install asks its checkout for the revision
        build_data["force_include"][str(built / LIBRARY)] = f"ippon/{LIBRARY}"
        for reference in self._references():  # what `ippon verify NAME` checks each named build against
            build_data["force_include"][str(Path(self.root) / "references" / reference)] = (
                f"ippon/references/{reference}"
            )
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-macosx_{MACOS.replace('.', '_')}_arm64"
        git = ["git", "-C", self.root]
        rev = subprocess.run(
            [*git, "rev-parse", "--show-prefix", "--short", "HEAD"], capture_output=True, text=True, check=False
        )
        if rev.returncode != 0 or rev.stdout.split("\n")[0] != "":  # not built from ippon's own checkout
            return
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--", "src/ippon", "native"], capture_output=True, text=True, check=False
        )
        self._dir = tempfile.mkdtemp()
        stamp = Path(self._dir) / "REVISION"
        stamp.write_text(rev.stdout.split("\n")[1] + ("+" if dirty.stdout.strip() else "") + "\n")
        build_data["force_include"][str(stamp)] = "ippon/REVISION"

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)

    def _references(self) -> list[str]:
        """The `reference` files of verified.py's builds. The catalog needs only the standard library, so it runs
        here without importing ippon, whose dependencies a build doesn't have."""
        catalog = runpy.run_path(str(Path(self.root) / "src/ippon/verified.py"))
        return sorted({build.reference for build in catalog["BUILDS"] if build.reference})

    def _native(self) -> Path:
        """Builds the library and the hook; the directory they are in."""
        manifest = Path(self.root) / "native/Cargo.toml"
        target = manifest.parent / "target"  # here whatever CARGO_TARGET_DIR says, since the build looks here
        command = [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(manifest),
            "--target-dir",
            str(target),
        ]
        try:
            subprocess.run(command, check=True, env={**os.environ, "MACOSX_DEPLOYMENT_TARGET": MACOS})
        except FileNotFoundError:
            raise RuntimeError("building ippon needs Rust's cargo on the PATH: https://rustup.rs") from None
        return target / "release"
