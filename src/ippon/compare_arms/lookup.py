"""Which arms run a model, as its adapter lists them (`compare_arms`), and where each one's script is."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..errors import InvalidRequest
from ..model_dir import ModelDir
from ..registry import select
from . import ARMS


def fitting(model: str | Path) -> tuple[str, ...]:
    """The arms the adapter of the model at `model` lists as running it."""
    return tuple(select(ModelDir.open(model)).compare_arms)


def arms_for(model: str, wanted: Sequence[str]) -> dict[str, tuple[str, str]]:
    """{name: (python, script)} for `wanted` ("name=python"), each one the model's adapter lists as fitting it."""
    fits = fitting(model)
    out = {}
    for spec in wanted:
        name, _, python = spec.partition("=")
        if name not in ARMS or name not in fits:
            raise InvalidRequest(
                f"{name!r} is not an arm for {model}: its arms are {', '.join(fits) or 'none'}"
                f" (all arms: {', '.join(ARMS)})"
            )
        if not python:
            raise InvalidRequest(f"name the Python that has {name} installed: --compare {name}=/path/to/bin/python")
        arm = ARMS[name]
        if arm.needs and not (Path(model) / arm.needs).is_file():
            raise InvalidRequest(f"{model} has no {arm.needs}, which {name} runs")
        out[name] = (str(Path(python).expanduser()), str(Path(__file__).parent / arm.script))
    return out
