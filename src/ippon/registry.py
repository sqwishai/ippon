"""Which adapter runs a model.

Adapters come from the `ippon.adapters` entry-point group (the built-in ones included), and from
`register()` for an adapter defined in a script or notebook. Each is checked once: it must name the API version
it was written for and be named as its entry point is. A model directory is matched against every adapter; no
match, or more than one, is an error that names the candidates. Code from a model repository is never imported.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from .errors import UnsupportedModel
from .model_dir import API_VERSION, ModelDir

if TYPE_CHECKING:
    from .adapter import Adapter

GROUP = "ippon.adapters"
_registered: dict[str, Adapter] = {}
_discovered: dict[str, Adapter] | None = None
_broken: dict[str, str] = {}


def _checked(adapter: Any, name: str | None = None) -> Adapter:
    if getattr(adapter, "api", None) != API_VERSION:
        raise UnsupportedModel(
            f"adapter {name or adapter!r} is written for API {getattr(adapter, 'api', None)}; "
            f"this ippon has API {API_VERSION}"
        )
    if name is not None and adapter.name != name:
        raise UnsupportedModel(f"adapter {adapter.name!r} is registered as {name!r}")
    return adapter  # type: ignore[no-any-return]


def register(adapter: Adapter) -> Adapter:
    """Register an adapter defined outside a package (a script, a notebook); returns it."""
    _registered[adapter.name] = _checked(adapter)
    return adapter


def adapters() -> dict[str, Adapter]:
    """Every adapter: the installed ones (entry points) and the registered ones. An installed adapter that fails to
    load is left out with a warning, so that one broken package doesn't stop every model (`broken()` lists them)."""
    global _discovered
    if _discovered is None:
        found: dict[str, Adapter] = {}
        for entry in entry_points(group=GROUP):
            try:
                obj = entry.load()
                found[entry.name] = _checked(obj() if isinstance(obj, type) else obj, entry.name)
            except Exception as error:  # a third party's package: report it, run the others
                _broken[entry.name] = f"{type(error).__name__}: {error}"
                warnings.warn(f"ippon: adapter {entry.name!r} failed to load: {error}", stacklevel=2)
        _discovered = found
    return {**_discovered, **_registered}


def read_safely() -> frozenset[str]:
    """The pickles the adapters name exactly (their `read_safely`), which only ptfile reads: the only ones ever
    downloaded."""
    return frozenset(name for adapter in adapters().values() for name in adapter.read_safely)


def broken() -> dict[str, str]:
    """Installed adapters that failed to load, and why."""
    adapters()
    return dict(_broken)


def select(model: ModelDir, name: str | None = None) -> Adapter:
    """The adapter for `model`: the one named, or the only one that matches it (a model's own adapter before a
    generic one, which reads a standard checkpoint format)."""
    available = adapters()
    if name is not None:
        if name not in available:
            raise UnsupportedModel(f"no adapter named {name!r} (installed: {', '.join(sorted(available)) or 'none'})")
        return available[name]
    matching = [a for a in available.values() if a.matches(model)]
    own = [a for a in matching if not a.generic]
    matching = own or matching
    if not matching:
        failed = f"; failed to load: {', '.join(sorted(_broken))}" if _broken else ""
        raise UnsupportedModel(
            f"{model.path}: no installed adapter matches this model "
            f"(installed: {', '.join(sorted(available)) or 'none'}{failed})"
        )
    if len(matching) > 1:
        raise UnsupportedModel(
            f"{model.path}: several adapters match ({', '.join(a.name for a in matching)}); pass adapter=<name>"
        )
    return matching[0]
