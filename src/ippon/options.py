"""`Decider.load`'s options that a command line or server checks before it downloads, allocates or starts anything:
their types, the keep-awake heartbeat's timing, and their validation, without importing MLX."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, get_args

from .errors import InvalidRequest

Strategy = Literal["auto", "reference"]  # how a plan is read (runtime.py): the fast paths, or every prompt whole
Truncation = Literal["creator", "error"]  # input past the window: cut as the creator's code cuts it, or refused
STRATEGIES: tuple[Strategy, ...] = get_args(Strategy)
TRUNCATIONS: tuple[Truncation, ...] = get_args(Truncation)

PERIOD_S = 1.0  # seconds between two of the keep-awake heartbeat's GPU commands (`Decider.load(keep_awake=)`)
AWAKE_WINDOW_S = 300.0  # seconds after the last answer that the heartbeat goes on (residency.py, a server's policy)


@dataclass(frozen=True)
class Reads:
    """The load options that change what a model answers, at their defaults unless given: special tokens' strings in
    a state read as text (`literal_state`), input past the window cut as its creator's code cuts it or refused
    (`truncation`), a window in place of the creator's (`window`), and a bring-your-own model's ippon.json from
    elsewhere (`settings`). A server lists them for each model it serves, so a client that reuses it can tell whether
    it answers as the client asks (`unlike`)."""

    literal_state: bool = False
    truncation: Truncation = "creator"
    window: int | None = None
    settings: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def unlike(self, asked: Mapping[str, Any]) -> list[str]:
        """The options `asked` gives (by name, each one a client left out absent) that these are not: a client that
        leaves an option out takes the server's, one that gives it gets it or another server, even when it gives the
        default (--truncation creator never reuses a server that refuses long input)."""
        return [f.name for f in fields(self) if f.name in asked and asked[f.name] != getattr(self, f.name)]

    @classmethod
    def of(cls, options: Mapping[str, Any]) -> Reads:
        """These options among `options` (a server's listing, or `Decider.load`'s keywords), each left out at its
        default."""
        return cls(**{f.name: options[f.name] for f in fields(cls) if f.name in options})


def precision_argv(precision: str | None) -> list[str]:
    """`--precision` as a load or `ippon serve` takes it on a command line: nothing for auto, the default."""
    return ["--precision", precision] if precision and precision != "auto" else []


def validate_load_options(
    *,
    chunk: int,
    window: int | None = None,
    prefix_budget: int | None = None,
    answer_cache: int = 0,
    keep_awake: float | None = None,
    strategy: str = "auto",
    truncation: str = Reads.truncation,
) -> None:
    """Raise InvalidRequest with the argument's name and accepted values when a load option cannot be used."""
    for name, value, minimum in (
        ("chunk", chunk, 1),
        ("window", window, 1),
        ("prefix_budget", prefix_budget, 0),
        ("answer_cache", answer_cache, 0),
    ):
        if value is None and name in ("window", "prefix_budget"):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            bound = "a positive integer" if minimum else "a non-negative integer (0 keeps none)"
            raise InvalidRequest(f"{name} (--{name.replace('_', '-')}) must be {bound}, got {value!r}")
    if keep_awake is not None and (
        isinstance(keep_awake, bool) or not isinstance(keep_awake, (int, float)) or not 0 <= keep_awake < float("inf")
    ):
        raise InvalidRequest(
            f"keep_awake (--keep-awake) must be a finite, non-negative number of seconds, got {keep_awake!r}"
        )
    if strategy not in STRATEGIES:
        raise InvalidRequest(f"strategy must be {' or '.join(map(repr, STRATEGIES))}, got {strategy!r}")
    if truncation not in TRUNCATIONS:
        raise InvalidRequest(f"truncation must be {' or '.join(map(repr, TRUNCATIONS))}, got {truncation!r}")
