"""Model architectures: each family's layers behind arch.base.Backbone, and which module loads a config.json's
model_type (DECODERS, ENCODERS), for the adapters that take any checkpoint of a family (bring-your-own models:
adapters/classifier.py's `classifier_arch`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from . import bert, deberta, dense, gemma3, gemma4, lfm2, modernbert, qwen35
from .base import Backbone, Cache

# model_type -> the module that loads it (each module's FAMILIES); decoders read causally, encoders both ways
DECODERS = {kind: module for module in (dense, gemma3, gemma4, qwen35, lfm2) for kind in module.FAMILIES}
ENCODERS = {kind: module for module in (bert, deberta, modernbert) for kind in module.FAMILIES}


B_co = TypeVar("B_co", covariant=True)


class Architecture(Protocol[B_co]):
    """A module of this package that loads a family's layers: `load(model_dir, config, *, dtype, bits, weights,
    ...its own options)`, returning its backbone."""

    @property
    def load(self) -> Callable[..., B_co]: ...


__all__ = ["DECODERS", "ENCODERS", "Architecture", "Backbone", "Cache"]
