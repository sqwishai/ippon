"""The exceptions ippon raises. Every one derives from DecisionError, so callers can catch the
library's failures without catching unrelated ones. Each names its kind for machine-readable output (`kind`: the type
of the error body) and the exit code the command line returns for it (`exit_code`):

    1  decision_error       any other failure ippon reports (and a failed `verify` or `doctor`)
    2                       the command line itself was misused (argparse)
    3  invalid_request      a request that isn't a System One request, or asks for something unsupported
    4  unsupported_model    no installed adapter runs the model, or not this variant of it
    5  context_too_long     input longer than the model or the configured limit takes
    6  insufficient_memory  the model doesn't fit this Mac at the precision asked for (the message names those that do)
    7  unsupported_profile  a chip profile or code path this Mac can't run
    8  overloaded           a server's queue is full or memory pressure is hard: retry later
    9  unavailable          a server that is closing, or whose model thread stopped
    10 model_not_found      a request names a model the server doesn't answer for
    11 license_not_accepted the model licence needs acceptance before new weights are downloaded
    12 insufficient_disk    the cache disk cannot hold the remaining download
    13 download_failed      a download stopped or arrived incomplete: repeat the command to resume it
    14 runtime_unavailable  a required dependency, native library or coding agent is missing or cannot load

Each also carries the HTTP status `serve` answers it with (`http_status`, 400 unless said otherwise) and, for a 503,
the seconds its caller waits before sending the request again (`retry_after`). `to_json` builds the error body that
`serve`, `mcp` and the command line's `--json` send, and `from_json` reads one back as its class. The kinds only a
server sends declare no exit code of their own: timeout (504), unauthorized (401), forbidden (403), not_found (404)
and internal (500); nor do the 405, 413 and 415 of a request in the wrong method, with a body too large or in an
encoding the server can't read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, ParamSpec, TypeVar


class DecisionError(Exception):
    """Base class of every error ippon raises on purpose."""

    kind = "decision_error"
    exit_code = 1
    http_status = 400
    retry_after: str | None = None  # seconds a 503 asks its caller to wait before sending the request again

    def to_json(self, exit_code: bool = False) -> dict[str, Any]:
        """The error body, {"error": {"type", "message"}}; with `exit_code`, the code the command line returns too."""
        error: dict[str, Any] = {"type": self.kind, "message": str(self)}
        if exit_code:
            error["exit_code"] = self.exit_code
        return {"error": error}


class InvalidRequest(DecisionError, ValueError):
    """A request that does not follow the System One schema, or asks for something not supported."""

    kind, exit_code = "invalid_request", 3


class UnsupportedModel(DecisionError):
    """A model directory or repository that no registered decision format can run."""

    kind, exit_code = "unsupported_model", 4


class ContextTooLong(DecisionError):
    """A rendered prompt longer than the model or the configured limit allows."""

    kind, exit_code = "context_too_long", 5


class UnsupportedProfile(DecisionError):
    """A chip profile this Mac can't run: one above its chip, or a switch its chip doesn't have."""

    kind, exit_code = "unsupported_profile", 7


class InsufficientMemory(DecisionError):
    """A model too big for this Mac's GPU working set at the precision asked for; the message names those that fit."""

    kind, exit_code = "insufficient_memory", 6
    http_status, retry_after = 503, "10"


class Overloaded(DecisionError):
    """A server with a full queue or hard memory pressure: send the request again later."""

    kind, exit_code = "overloaded", 8
    http_status, retry_after = 503, "1"


class Unavailable(DecisionError):
    """A server that can't answer: it is closing, or its model thread has stopped."""

    kind, exit_code = "unavailable", 9
    http_status, retry_after = 503, "1"


class ModelNotFound(DecisionError):
    """A request names a model the server doesn't answer for (`ippon serve MODEL...`)."""

    kind, exit_code = "model_not_found", 10
    http_status = 404


class LicenseNotAccepted(DecisionError):
    """A new weight download whose licence has not been accepted."""

    kind, exit_code = "license_not_accepted", 11


class InsufficientDisk(DecisionError):
    """The cache disk has less free space than the remaining download needs."""

    kind, exit_code = "insufficient_disk", 12


class DownloadFailed(DecisionError):
    """A download that stopped, or arrived incomplete or corrupt: repeating the command resumes it."""

    kind, exit_code = "download_failed", 13


class RuntimeUnavailable(DecisionError, ImportError):
    """A dependency ippon needs is missing, or MLX cannot load on this Mac. Still an ImportError, so code that guards
    `from ippon import Decider` with `except ImportError` keeps working."""

    kind, exit_code = "runtime_unavailable", 14

    @classmethod
    def of(cls, error: ImportError) -> RuntimeUnavailable | None:
        """The error a user can act on behind `error`: a third-party package that isn't installed (reinstall ippon),
        or MLX that doesn't load (`ippon doctor` says why). None for anything else, such as a fault inside ippon
        itself, which its caller raises as it is."""
        text = str(error)
        if "mlx" in text.lower() or "metal::load_device" in text:
            return cls(f"MLX doesn't load ({error}); `ippon doctor` says what to change")
        missing = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
        if missing and missing.split(".")[0] != "ippon":
            return cls(
                f"the package {missing!r} that ippon needs isn't installed ({error}); reinstall ippon in this Python "
                "environment with `python -m pip install --force-reinstall ippon`, or `uv sync --reinstall` in a "
                "checkout"
            )
        return None


class MemoryShort(Overloaded):
    """A request for a model that isn't loaded, refused while this Mac's memory is very short (`pressure.Level.HARD`):
    503, to send again later."""

    retry_after = "5"  # seconds: memory is read every second, and is plenty again only well above the line


class TooLarge(InvalidRequest):
    """A request body larger than the server takes."""

    http_status = 413


class UnsupportedEncoding(InvalidRequest):
    """A request body in a Content-Encoding the server doesn't read."""

    http_status = 415


class MethodNotAllowed(InvalidRequest):
    """A GET on a route that takes a POST, or a POST on one that takes a GET."""

    http_status = 405


class Timeout(Overloaded):
    """A request the server didn't answer within its `--timeout-s`: it is busy, send the request again later."""

    kind, http_status = "timeout", 504


class Abandoned(Timeout):
    """A reading stopped between passes because nobody waits for its answer any more (every request in it past its
    deadline, or its caller gone)."""


class Unauthorized(DecisionError):
    """A request without the server's API key."""

    kind, http_status = "unauthorized", 401


class Forbidden(DecisionError):
    """A request from a Host or an Origin the server doesn't answer."""

    kind, http_status = "forbidden", 403


class NotFound(DecisionError):
    """A request for a route the server doesn't have."""

    kind, http_status = "not_found", 404


class Internal(DecisionError):
    """A failure ippon didn't raise on purpose, as a server or an MCP tool reports it."""

    kind, http_status = "internal", 500


class NonFiniteAnswer(Internal):
    """An answer whose probabilities aren't finite: the model's activations overflowed its compute dtype. A decider
    that chose fp16 itself answers the request again in bf16 (`Decider.load`'s precision "auto")."""


_CLASSES = [c for c in list(globals().values()) if isinstance(c, type) and issubclass(c, DecisionError)]
KINDS: dict[str, type[DecisionError]] = {c.kind: c for c in _CLASSES if "kind" in vars(c)}  # by the kind each declares
BY_EXIT_CODE: dict[int, type[DecisionError]] = {c.exit_code: c for c in _CLASSES if "exit_code" in vars(c)}


def from_json(body: Mapping[str, Any]) -> DecisionError:
    """The error an error body (`DecisionError.to_json`) describes, as its kind's class (DecisionError for a kind this
    ippon doesn't know). KeyError or TypeError for a body that isn't one."""
    error = body["error"]
    return KINDS.get(str(error["type"]), DecisionError)(str(error["message"]))


def as_decision_error(error: BaseException) -> DecisionError:
    """`error` as its caller is told it: itself when a DecisionError, InsufficientMemory when the GPU ran out of
    memory, else Internal, naming the exception."""
    if isinstance(error, DecisionError):
        return error
    if out_of_memory(error):
        return InsufficientMemory(
            f"not enough memory ({error}): close other apps or reduce the request size, then retry"
        )
    return Internal(f"{type(error).__name__}: {error}")


OUT_OF_MEMORY = ("[metal::malloc]", "[malloc] Unable to allocate", "Insufficient Memory", "OutOfMemory")  # MLX's words


def out_of_memory(error: BaseException) -> bool:
    """Whether `error` is MLX's allocator or Metal running out of memory."""
    return isinstance(error, MemoryError) or (
        isinstance(error, RuntimeError) and any(words in str(error) for words in OUT_OF_MEMORY)
    )


P = ParamSpec("P")
T = TypeVar("T")


def memory_errors(function: Callable[P, T]) -> Callable[P, T]:
    """Report allocator failures at public load and inference boundaries, preserving unrelated exceptions."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except (MemoryError, RuntimeError) as error:
            if not out_of_memory(error):
                raise
            reported = as_decision_error(error)
            error.__traceback__ = None  # a retained error must not keep a failed load's weights alive
            raise reported from None

    return wrapped
