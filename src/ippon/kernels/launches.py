"""What a kernel launch needs besides its inputs, kept by shape."""

from typing import Any, TypeVar

V = TypeVar("V")

SHAPES = 4096  # a few per matrix shape and row count; an entry is a few hundred bytes


class Launches(dict[tuple[Any, ...], V]):
    """Each shape's launch arguments, worked out once: working them out again cost about as long as the launch, and
    a request launches a few kernels a layer. Cleared when full."""

    def __setitem__(self, key: tuple[Any, ...], value: V) -> None:
        if len(self) >= SHAPES:
            self.clear()
        super().__setitem__(key, value)
