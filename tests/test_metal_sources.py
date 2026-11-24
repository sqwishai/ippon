"""Bundled Metal remains readable without MLX, a source checkout or a filesystem-backed package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import ippon


def test_metal_resources_load_from_a_zip_without_mlx_or_the_original_working_directory(tmp_path: Path) -> None:
    package = Path(ippon.__file__).parent
    resources = sorted((package / "kernels/metal").glob("*.metal"))
    assert resources
    archive = tmp_path / "package.zip"
    with ZipFile(archive, "w") as z:
        for name in (
            "__init__.py",
            "_emulate.py",
            "device.py",
            "errors.py",
            "kernels/__init__.py",
            "kernels/_source.py",
        ):
            z.write(package / name, f"ippon/{name}")
        for path in resources:
            z.write(path, f"ippon/kernels/metal/{path.name}")
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from ippon.kernels import _source
assert _source.__file__.startswith(sys.argv[1])
assert 'mlx' not in sys.modules
names = json.loads(sys.argv[2])
print(json.dumps({name: _source.metal_source(name) for name in names}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(archive), json.dumps([p.name for p in resources])],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {p.name: p.read_text(encoding="utf-8") for p in resources}
