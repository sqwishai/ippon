"""ippon: open decision models on Apple silicon, fast and with the creator's answers.

The names below are imported on first use, so that `import ippon` (and `ippon --help`, and
`ippon doctor`, which must explain an MLX that doesn't load) doesn't load MLX.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import _emulate  # noqa: F401  first, before anything imports MLX (IPPON_PROFILE=m1..m4)
from .errors import (
    ContextTooLong,
    DecisionError,
    InvalidRequest,
    RuntimeUnavailable,
    UnsupportedModel,
    UnsupportedProfile,
)

if TYPE_CHECKING:
    from .adapter import Hidden, LoadContext, Loaded, Rendering
    from .decider import Decider
    from .model_dir import API_VERSION, ModelDir
    from .questions import Choice, Score, YesNo
    from .registry import register
    from .schema import Answer, Question, Request, Result, parse_request

__version__ = "0.2.1"

logging.getLogger(__name__).addHandler(logging.NullHandler())  # a library's log, shown where the caller configures it

_WHERE = {  # each lazily imported name's module
    **dict.fromkeys(("Hidden", "LoadContext", "Loaded", "Rendering"), "adapter"),
    **dict.fromkeys(("API_VERSION", "ModelDir"), "model_dir"),
    "Decider": "decider",
    **dict.fromkeys(("Choice", "Score", "YesNo"), "questions"),
    "register": "registry",
    **dict.fromkeys(("Answer", "Question", "Request", "Result", "parse_request"), "schema"),
}


def __getattr__(name: str) -> Any:
    if name not in _WHERE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(f".{_WHERE[name]}", __name__), name)
    except ImportError as error:
        failure = RuntimeUnavailable.of(error)
        if failure is None:  # a fault inside ippon, not the user's installation
            raise
        raise failure from None
    globals()[name] = value  # once
    return value


__all__ = [
    "API_VERSION",
    "Answer",
    "Choice",
    "ContextTooLong",
    "Decider",
    "DecisionError",
    "Hidden",
    "InvalidRequest",
    "LoadContext",
    "Loaded",
    "ModelDir",
    "Question",
    "Rendering",
    "Request",
    "Result",
    "RuntimeUnavailable",
    "Score",
    "UnsupportedModel",
    "UnsupportedProfile",
    "YesNo",
    "__version__",
    "parse_request",
    "register",
]
