"""How a server takes requests and keeps its models ready between them: how many it answers together and keeps
waiting, and (`ippon serve --policy`) how long the GPU heartbeat goes on after a request and how long a model may
answer nothing before it unloads (the heartbeat's own timing is options.py's). batches.py and serve.py run these; this
module only names them, without MLX, so the command line can describe them."""

from __future__ import annotations

from typing import Any

from .options import AWAKE_WINDOW_S

MAX_BATCH = 64  # requests answered together at most
# Multimodal requests (images, videos, audio) are heavier per item; capping their batch prevents a burst of distinct
# photos from blocking every caller until the entire burst finishes.
MAX_MEDIA_BATCH = 4
# requests a bulk client keeps at a shared server at once: one, so each answer takes the model's own time and other
# callers are not held behind a queue. A request's time is those in flight over the rate, and the rate barely moves:
# 300 commits of about 300 tokens, two questions each, M5 Pro, Kettei Core read 11.0 a second at 1, 4 and 16 in
# flight, in 91 ms, 0.4 s and 1.5 s a commit, and three callers at once read 11.5 and 11.6 a second at 1 and 16
# each, in 0.26 s and 4.1 s. On 2,000 commits of 346 tokens Core read 9.7 a second at 1 and 12.0 at 16, Kettei Dot
# 35 and 42. `--batch` raises it for a server nobody shares
IN_FLIGHT = 1
MAX_QUEUE = 256  # requests waiting at most; past that a request is refused (503)

# each policy's heartbeat window (None: no heartbeat; infinity: as long as the server runs) and keep-alive, the
# seconds a model may answer nothing before it unloads (infinity: never)
POLICIES: dict[str, tuple[float | None, float]] = {
    "latency": (float("inf"), float("inf")),
    "balanced": (AWAKE_WINDOW_S, float("inf")),
    "battery": (None, AWAKE_WINDOW_S),
}
DEFAULT = "balanced"  # what `ippon serve` and a server MCP starts keep ready, unless --policy says otherwise


def policy_json(name: str, every: float, window: float | None, keep_alive: float) -> dict[str, Any]:
    """How a server keeps its models ready, in its record: the policy's name, its heartbeat (every `every` seconds,
    for `for_s` after a request; null for_s: as long as it runs; a null heartbeat: none) and keep-alive (seconds a
    model may answer nothing before it unloads; null: never)."""

    def finite(seconds: float) -> float | None:
        return None if seconds == float("inf") else seconds

    heartbeat = None if window is None else {"every_s": every, "for_s": finite(window)}
    return {"name": name, "heartbeat": heartbeat, "keep_alive_s": finite(keep_alive)}
