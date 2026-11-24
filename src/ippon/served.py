"""What an ippon server and its clients agree on: where a server on this Mac listens and is reached (`url_of`), and
each model it serves as `GET /v1/models` lists it (`ServedModel`), with the one rule for whether it answers a name a
request or a client gives (`ServedModel.answers`). serve.py produces the list; daemon.py, the CLI, the MCP server and
the agent commands read it."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import names, provenance, quantization
from .options import Reads
from .quantization import SHORT

DEFAULT_PORT = 8000
CLIENT_HEADER = "X-Ippon-Client"  # the name a client goes by in a server's log and status, such as "history-reader"
CLIENT_ENV = "IPPON_CLIENT"  # the name ippon's own clients send as CLIENT_HEADER; `ippon AGENT` sets it to the agent's
LOOPBACK = ("localhost", "127.0.0.1", "::1")  # the names of this Mac's loopback a local server is reached at


def reach(host: str) -> str:
    """The address a client on this Mac reaches a server listening on `host` at: the loopback for every address."""
    return {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "localhost": "127.0.0.1", "::": "::1"}.get(host, host)


def url_of(port: int, host: str = "127.0.0.1") -> str:
    """The URL a client on this Mac reaches a server listening on `host`:`port` at."""
    address = reach(host)
    return f"http://[{address}]:{port}" if ":" in address else f"http://{address}:{port}"


def is_loopback(address: str) -> bool:
    """Whether a server listening on `address` can be reached from this Mac only."""
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return address == "localhost"


@dataclass(frozen=True)
class ServedModel:
    """A model a server answers for, as `GET /v1/models` lists it in `data`: the name requests use (`id`), what it
    loads (a Hub id, a folder), the weights it runs on ("full", or "q8", "q6", "q4" when quantized as it loads:
    names.weights_at), how its load options change its answers (`reads`; None when the server doesn't say), once
    loaded its adapter, chip profile, details and the media a request may carry for it (`media`: "images", "audio",
    "spoken", "videos"), and the precision it was asked to load at (`precision`: "auto", "fp32", "q8", ...; None when
    the server doesn't say)."""

    id: str
    loads: str
    weights: str = "full"
    reads: Reads | None = None
    loaded: bool = False
    adapter: str | None = None
    profile: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    precision: str | None = None
    media: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "object": "model",
            "loads": self.loads,
            "weights": self.weights,
            "precision": self.precision,
            "reads": {} if self.reads is None else self.reads.to_json(),
            "loaded": self.loaded,
        }
        if self.loaded:
            out |= {
                "adapter": self.adapter,
                "profile": self.profile,
                "details": dict(self.details),
                "media": [*self.media],
            }
        elif self.media:
            out["media"] = [*self.media]
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ServedModel:
        name = str(data["id"])
        reads = data.get("reads")
        return cls(
            id=name,
            loads=str(data.get("loads") or name),
            weights=str(data.get("weights") or "full"),
            reads=Reads.of(reads) if reads else None,
            loaded=bool(data.get("loaded")),
            adapter=data.get("adapter"),
            profile=data.get("profile"),
            details=dict(data.get("details") or {}),
            precision=data.get("precision"),
            media=tuple(data.get("media") or ()),
        )

    def answers(self, asked: str, weights: str | None = None) -> bool:
        """Whether this model answers for `asked`: the name it is served as, or a name of what it loads (a short name,
        Hub id or folder of it, at the revision it pins if any: names.answers_for); and, when `weights` is given, only
        on those weights. A server routes a request by the name it gives (`serve.Models.named`), a client finds the
        server and name of the model it asks for (`daemon.match`), and `ippon stop MODEL` the server running it, all
        by this rule."""
        same = asked == self.id or names.answers_for(self.loads, asked)
        return same and (weights is None or weights == self.weights)

    def unlike(self, precision: str, reads: Mapping[str, Any]) -> list[str]:
        """The load options a client asks for that this model was loaded otherwise with, as their flags: a compute
        precision named outright (fp32 or float32 alike, as quantization.get reads it; auto and the quantized ones,
        which `answers` matches by weights, take any) other than the one it loads at, and each option `reads` gives
        (by name) that its own are not (`Reads.unlike`). A client reuses only a server that answers as it asks
        (`daemon.match`); an unknown precision is refused (InvalidRequest)."""
        dtype = quantization.get(precision).dtype
        flags = ["--precision"] if dtype is not None and SHORT[dtype] != self.precision else []
        return flags + ["--" + name.replace("_", "-") for name in (self.reads or Reads()).unlike(reads)]

    @property
    def licence(self) -> str | None:
        """The name of the licence of the build it loaded, once loaded (provenance.py)."""
        found = provenance.licence(self.details)
        return found.get("name") if found else None
