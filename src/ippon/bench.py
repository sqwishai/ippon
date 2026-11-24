"""Timing a model on this Mac, the way every number ippon publishes is measured.

    ippon bench MODEL [--sizes 300,1000,2000,4000] [--questions 1,3,5] [--scenario fresh] [--n 20]
                            [--warmup 2] [--seed 1] [--out results.json]

For each state size and question count: `warmup` requests, then `n` timed ones, timed around the whole call on the
caller's side. What the requests are is the scenario:

- **fresh**: every request a new state that starts with a unique identifier, so no cache can carry anything
  between requests (a service answering many different tickets);
- **repeat**: one state, asked a new set of questions each time (an agent asking about the same state again);
- **grow**: a state that grows by a turn of GROW words before each request (an agent's conversation);
- **idle**: fresh requests after `gap` seconds idle each (a service answering now and then);
- **concurrent**: `clients` clients sending fresh requests at once through serve's batcher, in this process (a
  service answering many callers), reported with the requests answered per second.

A cell reports the median (p50), p95 and p99, and a bootstrap 95% interval of the median, with the machine: chip, macOS
and runtime versions. Live output names busy processes; published JSON retains only their CPU shares. The requests
depend only on the scenario, the shape and `seed`, so two runs with the same seed time the same requests (`ab.py` relies
on it), and the answers are kept, so two runs' answers can be compared too.
"""

from __future__ import annotations

import random
import statistics
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import machine, policy
from ._portable import PRIVATE_REPORT_ENV, dumps, private_report, record
from .asks import Asker, standard
from .errors import InvalidRequest
from .schema import Result, parse_request
from .stats import median_interval, quantile

if TYPE_CHECKING:
    from .decider import Decider

WORDS = [
    "account",
    "order",
    "refund",
    "shipping",
    "invoice",
    "delay",
    "password",
    "warranty",
    "discount",
    "address",
    "payment",
    "card",
    "email",
    "return",
    "delivery",
    "support",
    "manager",
    "receipt",
    "subscription",
    "cancel",
    "upgrade",
    "customer",
    "charged",
    "twice",
    "please",
    "confirm",
]
SCENARIOS = ("fresh", "repeat", "grow", "idle", "concurrent")
GROW = 40  # words a grown state gains before each request: about one agent turn


def state(words: int, seed: int) -> str:
    rng = random.Random(seed)
    return f"Ticket {seed:08d}. " + " ".join(rng.choice(WORDS) for _ in range(words))


def bodies(
    scenario: str, words: int, count: int, seed: int, total: int, ask: Asker = standard
) -> Iterator[dict[str, Any]]:
    """The `total` requests of one cell of `scenario`, from `seed`: the same for every runtime given the same seed;
    `ask` gives the questions (`asks.asker`)."""
    if scenario not in SCENARIOS:
        raise InvalidRequest(f"scenario must be one of {', '.join(SCENARIOS)}")
    if scenario in ("fresh", "idle", "concurrent"):
        for i in range(total):
            yield {"state": state(words, seed + i), "questions": ask(count, 0)}
    elif scenario == "repeat":
        text = state(words, seed)
        for i in range(total):
            yield {"state": text, "questions": ask(count, i)}
    else:  # grow
        text, rng = state(words, seed), random.Random(seed + 1)
        for i in range(total):
            text += f" Update {i + 1}: " + " ".join(rng.choice(WORDS) for _ in range(GROW))
            yield {"state": text, "questions": ask(count, 0)}


Key = tuple[str, int, int, float]  # a cell's scenario, words, questions, and its idle gap or concurrent clients


def fields(key: Key) -> dict[str, Any]:
    """A cell's shape as every timing report writes it: the gap only for idle cells, the clients only for concurrent
    ones."""
    scenario, words, questions, extra = key
    out: dict[str, Any] = {"scenario": scenario, "words": words, "questions": questions}
    if scenario == "idle":
        out["gap_s"] = extra
    elif scenario == "concurrent":
        out["clients"] = extra
    return out


def key(cell: Mapping[str, Any]) -> Key:
    """A reported cell's shape (`fields`), to pair it with the same cell of another run."""
    return cell["scenario"], cell["words"], cell["questions"], cell.get("gap_s", cell.get("clients", 0.0))


def label(cell: Mapping[str, Any]) -> str:
    """A reported cell's scenario as a table names it."""
    if cell["scenario"] == "idle":
        return f"idle {cell['gap_s']:g} s"
    if cell["scenario"] == "concurrent":
        return f"{cell['clients']} clients"
    return str(cell["scenario"])


def split_rounds(n: int, rounds: int, seed: int) -> list[tuple[int, int]]:
    """`n` timed requests per cell spread over `rounds` fresh processes: each round's count and seed, a seed of its
    own so rounds time other requests, the same for every arm in a round."""
    if rounds < 1 or n < rounds:
        raise InvalidRequest("timing in rounds needs at least one round and one request per round")
    return [(n // rounds + (r < n % rounds), seed + 100_000 * r) for r in range(rounds)]


@dataclass(frozen=True)
class Cell:
    words: int
    questions: int
    milliseconds: tuple[float, ...]
    input_tokens: int
    scenario: str = "fresh"
    gap: float = 0.0
    warmup_ms: tuple[float, ...] = ()  # the first is this shape's first request (compiles, first reads)
    answers: tuple[tuple[float, ...], ...] = ()  # every timed request's probabilities, its questions in order
    kept_hits: int = 0
    busy: tuple[str, ...] = field(default=())
    clients: int = 1  # concurrent: how many clients sent requests at once
    throughput: float = 0.0  # concurrent: requests answered per second, over the cell
    batches: tuple[int, ...] = ()  # concurrent: how many requests each pass of the model answered

    @property
    def median(self) -> float:
        return statistics.median(self.milliseconds)

    def percentile(self, q: float) -> float:
        return quantile(self.milliseconds, q)

    def to_json(self) -> dict[str, Any]:
        lo, hi = median_interval(self.milliseconds)
        extra = self.clients if self.scenario == "concurrent" else self.gap
        return {
            **fields((self.scenario, self.words, self.questions, extra)),
            **(
                {"throughput_rps": round(self.throughput, 2), "batches": list(self.batches)}
                if self.scenario == "concurrent"
                else {}
            ),
            "input_tokens": self.input_tokens,
            "median_ms": round(self.median, 2),
            "median_ci_ms": [round(lo, 2), round(hi, 2)],
            "p95_ms": round(self.percentile(0.95), 2),
            "p99_ms": round(self.percentile(0.99), 2),
            "ms": [round(t, 2) for t in self.milliseconds],
            "warmup_ms": [round(t, 2) for t in self.warmup_ms],
            "kept_prefix_hits": self.kept_hits,
            "busy": list(self.busy),
            "answers": [[round(p, 6) for p in a] for a in self.answers],
        }


@dataclass(frozen=True)
class Spec:
    """What a timing run sends: for each state size (words) and question count, `warmup` untimed then `n` timed
    requests of `scenario`, generated from `seed`; `gaps` are the idle scenario's pauses before each request,
    `clients` the concurrent scenario's senders. bench and compare read it alike, so they send the same requests."""

    sizes: tuple[int, ...]
    questions: tuple[int, ...]
    scenario: str = "fresh"
    n: int = 20
    warmup: int = 2
    seed: int = 1
    gaps: tuple[float, ...] = (1.0,)
    clients: tuple[int, ...] = (1, 4, 8)

    @classmethod
    def from_args(cls, args: Any) -> Spec:
        """From bench's options (cli.py); InvalidRequest when a list isn't comma-separated numbers."""
        try:
            return cls(
                tuple(int(x) for x in args.sizes.split(",")),
                tuple(int(x) for x in args.questions.split(",")),
                args.scenario,
                args.n,
                args.warmup,
                args.seed,
                tuple(float(x) for x in args.gaps.split(",")),
                tuple(int(x) for x in args.clients.split(",")),
            )
        except ValueError as error:
            raise InvalidRequest(
                f"--sizes, --questions, --gaps and --clients are comma-separated numbers ({error})"
            ) from None

    def cells(self) -> Iterator[tuple[int, int, float, int]]:
        """Every cell's words, questions, gap and the seed of its first request (after the cells before it), in
        order; the concurrent scenario's cells are `concurrent`'s."""
        seed = self.seed
        for words in self.sizes:
            for count in self.questions:
                for gap in self.gaps if self.scenario == "idle" else (0.0,):
                    yield words, count, gap, seed
                    seed += self.warmup + self.n

    def to_json(self) -> dict[str, Any]:
        return {"scenario": self.scenario, "n": self.n, "warmup": self.warmup, "seed": self.seed}


def run(decider: Decider, spec: Spec, window_ms: float = 0.0, max_batch: int | None = None) -> list[Cell]:
    """The spec's cells timed on `decider`, its kept answers set aside: a timed request is always read, the repeat
    scenario's identical ones included."""
    with decider.uncached():
        return _run(decider, spec, window_ms, max_batch)


def timed(decider: Decider, body: Mapping[str, Any]) -> tuple[Result, float]:
    """`body` answered by `decider`, and the milliseconds the call took on the caller's side: parsed, rendered, read
    and shaped, as every number bench reports is timed."""
    start = time.perf_counter()
    result = decider.answer_rendered([decider.render(parse_request(body))])[0]
    return result, (time.perf_counter() - start) * 1000


def flat(result: Result) -> tuple[float, ...]:
    """A result's probabilities, its questions in order and each one's options in order, as a cell keeps them."""
    return tuple(p for answer in result.answers.values() for p in answer.probabilities)


def _run(decider: Decider, spec: Spec, window_ms: float = 0.0, max_batch: int | None = None) -> list[Cell]:
    n, warmup = spec.n, spec.warmup
    if spec.scenario == "concurrent":
        shapes = ((w, c, k) for w in spec.sizes for c in spec.questions for k in spec.clients)
        return [
            concurrent(decider, words, count, k, n, warmup, spec.seed + 1000 * i, window_ms, max_batch)
            for i, (words, count, k) in enumerate(shapes)
        ]
    cells = []
    for words, count, gap, seed in spec.cells():
        times: list[float] = []
        warm: list[float] = []
        answers: list[tuple[float, ...]] = []
        seen: set[str] = set()
        tokens = 0
        hits = decider.prefixes.stats.hits if decider.prefixes else 0
        for i, body in enumerate(bodies(spec.scenario, words, count, seed, warmup + n, decider.asks)):
            if gap:
                time.sleep(gap)
            if i == warmup and decider.prefixes:
                hits = decider.prefixes.stats.hits
            result, elapsed = timed(decider, body)
            (times if i >= warmup else warm).append(elapsed)
            if i >= warmup:
                tokens = int(result.response["usage"]["input_tokens"])
                answers.append(flat(result))
                seen.update(machine.busy() if i % 5 == 0 else ())
        kept = (decider.prefixes.stats.hits if decider.prefixes else 0) - hits
        busy = tuple(sorted(seen))
        cells.append(
            Cell(words, count, tuple(times), tokens, spec.scenario, gap, tuple(warm), tuple(answers), kept, busy)
        )
    return cells


def concurrent(
    decider: Decider,
    words: int,
    count: int,
    clients: int,
    n: int,
    warmup: int,
    seed: int,
    window_ms: float = 0.0,
    max_batch: int | None = None,
) -> Cell:
    """`clients` clients each sending its share of `n` fresh requests (after `warmup` each) one after another, as
    fast as they are answered, through `serve`'s batcher (in this process, no HTTP): each request's time from sent to
    answered, and the requests answered per second over the timed ones. `max_batch`: serve's --max-batch."""
    from .serve import Batcher  # serve and batches load MLX: bench is imported by `ippon --help`

    batcher = Batcher(  # every batch kept
        decider, max_batch or policy.MAX_BATCH, window_ms=window_ms, history=n + clients * warmup + 1
    )
    ask = decider.asks
    per = -(-n // clients)
    times: dict[tuple[int, int], float] = {}  # by (client, request): the answers line up across runs
    answers: dict[tuple[int, int], tuple[float, ...]] = {}
    lock = threading.Lock()
    warmed, start_line = threading.Barrier(clients + 1), threading.Barrier(clients + 1)

    def client(c: int) -> None:
        own = list(bodies("fresh", words, count, seed + c * (warmup + per), warmup + per, ask))
        for body in own[:warmup]:
            batcher.submit(body).result()
        warmed.wait()
        start_line.wait()
        for i, body in enumerate(own[warmup:]):
            t = time.perf_counter()
            result = batcher.submit(body).result()
            elapsed = (time.perf_counter() - t) * 1000
            with lock:
                times[c, i] = elapsed
                answers[c, i] = flat(result)

    threads = [threading.Thread(target=client, args=(c,)) for c in range(clients)]
    for t in threads:
        t.start()
    warmed.wait()  # every warm-up answered: the batches after this are the timed requests'
    warm_batches = batcher.answered
    start_line.wait()
    began = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - began
    timed = batcher.answered - warm_batches
    batches = tuple(list(batcher.batches)[len(batcher.batches) - timed :])
    batcher.close()
    body = next(bodies("fresh", words, count, seed, 1, ask))
    tokens = int(decider.format.usage(decider.render(parse_request(body)))["input_tokens"])
    order = sorted(times)
    return Cell(
        words, count, tuple(times[k] for k in order), tokens, "concurrent", answers=tuple(answers[k] for k in order),
        busy=tuple(sorted(set(machine.busy()))), clients=clients, throughput=len(times) / wall, batches=batches,
    )  # fmt: skip


def report(model: str, decider: Decider, cells: Sequence[Cell], spec: Spec) -> dict[str, Any]:
    return record(
        {
            "model": model,
            "adapter": decider.adapter,
            **machine.describe(),
            "profile": decider.profile.name,
            "forced_profile": decider.profile.forced,
            "switches": decider.profile.switches.as_dict(),
            "precision": decider.precision,
            "weights": decider.details.get("weights"),
            "kernels": decider.details.get("kernels"),  # how its matmuls' kernels were chosen (choices.py)
            "chunk": decider.chunk,
            "prefix_budget": decider.prefixes.budget if decider.prefixes else 0,
            "env": {k: v for k, v in machine.environment().items() if k != PRIVATE_REPORT_ENV},
            "n": spec.n,
            "warmup": spec.warmup,
            "seed": spec.seed,
            "kept_prefix_hits": decider.prefixes.stats.hits if decider.prefixes else 0,  # 0: every request read whole
            "cells": [c.to_json() for c in cells],
        },
        process_names=private_report(),
    )


def table(cells: Sequence[Cell]) -> str:
    lines = [
        "| scenario | words | questions | input tokens | median | 95% CI | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        lo, hi = median_interval(c.milliseconds)
        name = label(c.to_json())
        if c.scenario == "concurrent":
            batch = statistics.mean(c.batches) if c.batches else 0.0
            name += f", {c.throughput:,.1f} req/s, batches of {batch:.1f}"
        lines.append(
            f"| {name} | {c.words:,} | {c.questions} | {c.input_tokens:,} | {c.median:,.1f} ms | "
            f"{lo:,.1f} to {hi:,.1f} | {c.percentile(0.95):,.1f} ms |"
        )
    if any(c.busy for c in cells):
        lines.append("\nbusy while timing: " + ", ".join(sorted({b for c in cells for b in c.busy})))
    return "\n".join(lines)


def save(path: str, data: dict[str, Any]) -> None:
    """Write portable evidence for bench, A/B and runtime comparisons."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(data, process_names=private_report(path), indent=1))
