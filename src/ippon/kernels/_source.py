"""Load bundled Metal source independently of the working directory or package installation layout."""

from functools import cache
from importlib.resources import files


@cache
def metal_source(name: str) -> str:
    """Read a UTF-8 source file from kernels/metal, preserving its whitespace for MLX's source cache."""
    return files("ippon.kernels").joinpath("metal", name).read_text(encoding="utf-8")
