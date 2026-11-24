"""`ippon profile`: where a request's time goes, operation by operation, or phase by phase.

    ippon profile MODEL --size 1000 --questions 3          # bench's request of that shape
    ippon profile MODEL --request request.json
    ippon profile MODEL --phases --scenario fresh          # parse, render, plan, graph, GPU, readout, answer
    ippon profile MODEL --phases --scenario changed        # new questions about a state already read

With --phases, each request is answered by the decider's own answer path, which tells each step's end to a hook
(runtime.PHASE), and each step's time is summed: parsing the body; rendering its prompts (the template and the
tokenizer; tokenizing is also timed apart, by encoding the prompts' text again, an estimate of its share); planning
the passes (and encoding a request's images or audio); building the passes' graph (`runtime.run` returning: for a
prompt of one chunk the GPU starts on it while Python builds the rest, every few layers with the pipeline switch on,
so this is Python's time with the GPU at work beneath it, and for a longer prompt it includes waiting for earlier
chunks); waiting for the GPU to finish the hidden states; the readout (the head's matmuls and their wait); and
shaping the answers on the CPU. With `--switch pipeline=off` nothing runs until the graph is built, which separates
Python's time (graph) from the GPU's (gpu). "fresh" requests each have a state of their own; "changed" ones ask new
questions about one state, so after the first they start from its kept reading. The first request of each is
reported apart from the rest.

The request is answered as the decider answers it (after the warm-up) for its own time, then again with every operation
of the backbone's `ops` (each matmul by shape and kind, the norms, GLUs and convolutions, attention, the
recurrence) evaluated on its own and timed. Each timed operation first waits for its inputs, so its time is its
own; from each, the round trip of an empty synchronization (measured first) is taken off, so what is left is
about its GPU time. "Everything else" is the request's own time less those: embeddings, RoPE, the recurrence's
inputs, masks, the readout, and Python. The table shows where time goes, not what removing an operation would
save, since in a request the GPU overlaps the tail of one kernel with the start of the next.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import mlx.core as mx

from . import quant
from .arch.ops import Ops
from .decider import Decider
from .errors import InvalidRequest
from .kernels import gemm
from .runtime import PHASE
from .schema import Request, parse_request

T = TypeVar("T")


def _arrays(values: Any) -> list[mx.array]:
    if isinstance(values, mx.array):
        return [values]
    if isinstance(values, tuple | list):
        return [a for v in values for a in _arrays(v)]
    return []


def matrix_label(w: quant.Matrix) -> str:
    if isinstance(w, quant.QuantizedMatrix):
        mode = f"q{w.bits}" if w.mode == "affine" else w.mode
        return f"matmul {w.shape[0]}x{w.shape[1]} {mode}{' scaled' if w.multiplier() is not None else ''}"
    if isinstance(w, quant.JoinedMatrix):
        return "joined " + " + ".join(matrix_label(p) for p in w.parts)
    return f"matmul {w.shape[0]}x{w.shape[1]}"


@dataclass(frozen=True)
class TimedOps(Ops):
    """The same operations as the Ops it copies, each evaluated on its own and timed into `clock` (seconds)."""

    clock: dict[str, float] = field(default_factory=lambda: defaultdict(float), hash=False, compare=False)
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int), hash=False, compare=False)
    _depth: list[int] = field(default_factory=lambda: [0], hash=False, compare=False)  # operations being timed

    def _timed(self, label: str, run: Callable[[], T], inputs: Any) -> T:
        """`run()`, evaluated on its own and timed as `label`; an operation it calls in turn (a biased matmul's
        matmul) is part of its time, not timed again."""
        if self._depth[0]:
            return run()
        mx.eval(_arrays(inputs))
        start = time.perf_counter()
        self._depth[0] += 1
        try:
            out = run()
            mx.eval(_arrays(out))
        finally:
            self._depth[0] -= 1
        self.clock[label] += time.perf_counter() - start
        self.calls[label] += 1
        return out

    def mm(self, x: mx.array, w: quant.Matrix) -> mx.array:
        return self._timed(matrix_label(w), lambda: super(TimedOps, self).mm(x, w), x)

    def mm_bias(self, x: mx.array, w: quant.Matrix, b: mx.array) -> mx.array:
        return self._timed(matrix_label(w), lambda: super(TimedOps, self).mm_bias(x, w, b), (x, b))

    def glu_mm(self, x: mx.array, w: quant.Matrix, activation: str = "silu") -> mx.array:
        m = x.size // x.shape[-1]
        fused = (
            self.gemm_tiles and gemm.tile_for(self.gemm_tiles, x.dtype, w.shape[0], x.shape[-1], m, activation)
        ) or (
            isinstance(w, quant.QuantizedMatrix) and w.glu_tile(m, activation) is not None  # its own fused tile
        )
        if not fused:  # timed as its matmul and its GLU
            return super().glu_mm(x, w, activation)
        return self._timed(
            f"{matrix_label(w)} + {activation} (fused)", lambda: super(TimedOps, self).glu_mm(x, w, activation), x
        )

    def attention(self, q: mx.array, k: mx.array, v: mx.array, scale: float, mask: mx.array | str | None) -> mx.array:
        return self._timed("attention", lambda: super(TimedOps, self).attention(q, k, v, scale, mask), (q, k, v, mask))

    def gated_delta(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        gamma: mx.array,
        beta: mx.array,
        state: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        return self._timed(
            "gated delta recurrence",
            lambda: super(TimedOps, self).gated_delta(q, k, v, gamma, beta, state),
            (q, k, v, gamma, beta, state),
        )

    def norm(self, x: mx.array, w: mx.array, eps: float, kind: str = "rms", groups: int = 1) -> mx.array:
        return self._timed("norm", lambda: super(TimedOps, self).norm(x, w, eps, kind, groups), (x, w))

    def add_norm(
        self, x: mx.array, delta: mx.array, w: mx.array, eps: float, kind: str = "rms", groups: int = 1
    ) -> tuple[mx.array, mx.array]:
        return self._timed(
            "add + norm", lambda: super(TimedOps, self).add_norm(x, delta, w, eps, kind, groups), (x, delta)
        )

    def add_layer_norm(self, x: mx.array, delta: mx.array, w: mx.array, b: mx.array, eps: float) -> mx.array:
        return self._timed(
            "add + layer norm", lambda: super(TimedOps, self).add_layer_norm(x, delta, w, b, eps), (x, delta)
        )

    def delta_inputs(
        self,
        mixed: mx.array,
        proj: mx.array,
        neg_a: mx.array,
        dt_bias: mx.array,
        k_heads: int,
        k_dim: int,
        v_heads: int,
        v_dim: int,
        scale: float,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        return self._timed(
            "gated delta inputs",
            lambda: super(TimedOps, self).delta_inputs(
                mixed, proj, neg_a, dt_bias, k_heads, k_dim, v_heads, v_dim, scale
            ),
            (mixed, proj),
        )

    def glu(self, gate_up: mx.array, activation: str = "silu") -> mx.array:
        return self._timed("glu", lambda: super(TimedOps, self).glu(gate_up, activation), gate_up)

    def route(
        self,
        x: mx.array,
        router: quant.Matrix,
        top_k: int,
        inputs: mx.array | None = None,
        expert_scale: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        return self._timed(
            "experts: routing and grouping",
            lambda: super(TimedOps, self).route(x, router, top_k, inputs, expert_scale),
            x,
        )

    def expert_mm(self, x: mx.array, w: quant.Stacked | quant.JoinedStacked, experts: mx.array) -> mx.array:
        experts_n, rows, columns = w.shape
        parts = w.parts if isinstance(w, quant.JoinedStacked) else (w,)
        modes = " + ".join(
            ("dense" if p.scales is None else f"q{p.bits}" if p.mode == "affine" else p.mode)
            + (" scaled" if p.global_scale is not None else "")
            for p in parts
        )
        label = f"experts: {experts_n} x {rows}x{columns}" + ("" if modes == "dense" else f" {modes}")
        return self._timed(label, lambda: super(TimedOps, self).expert_mm(x, w, experts), (x, experts))

    def combine(self, y: mx.array, order: mx.array, top: mx.array) -> mx.array:
        return self._timed("experts: combine", lambda: super(TimedOps, self).combine(y, order, top), (y, order, top))

    def gated_norm(self, o: mx.array, proj: mx.array, offset: int, w: mx.array, eps: float) -> mx.array:
        return self._timed("gated norm", lambda: super(TimedOps, self).gated_norm(o, proj, offset, w, eps), (o, proj))

    def causal_conv(self, x: mx.array, window: mx.array, taps: mx.array, silu: bool) -> mx.array:
        return self._timed("convolution", lambda: super(TimedOps, self).causal_conv(x, window, taps, silu), (x, window))

    def gated_conv(self, proj: mx.array, window: mx.array | None, taps: mx.array) -> mx.array:
        return self._timed("convolution", lambda: super(TimedOps, self).gated_conv(proj, window, taps), (proj, window))


@dataclass(frozen=True)
class Profile:
    request_ms: float  # the request's own time, median
    synchronized_ms: float  # the request with every operation timed on its own, median
    round_trip_ms: float  # an empty synchronization's, median
    operations: Mapping[str, tuple[int, float]]  # (calls, ms) per request, by operation, before the round trips

    def net(self, name: str) -> float:
        calls, ms = self.operations[name]
        return max(ms - calls * self.round_trip_ms, 0.0)

    def table(self) -> str:
        net = {name: self.net(name) for name in self.operations}
        rows = sorted(net.items(), key=lambda kv: -kv[1])
        rest = self.request_ms - sum(net.values())
        width = max(len("everything else"), *(len(name) for name in net))
        lines = [
            f"request {self.request_ms:.1f} ms (synchronized {self.synchronized_ms:.1f} ms; round trip "
            f"{self.round_trip_ms * 1000:.0f} us taken off each operation)",
            f"{'operation':<{width}}  {'calls':>5}  {'ms':>8}  share",
        ]
        lines += [
            f"{name:<{width}}  {self.operations[name][0]:5}  {ms:8.2f}  {ms / self.request_ms:6.1%}"
            for name, ms in rows
        ]
        lines.append(f"{'everything else':<{width}}  {'':5}  {rest:8.2f}  {rest / self.request_ms:6.1%}")
        return "\n".join(lines)


def round_trip(repeat: int = 50) -> float:
    """The median wall time, in ms, of evaluating a trivial GPU operation on inputs already evaluated."""
    x = mx.zeros((1,))
    mx.eval(x)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        mx.eval(x + 1)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def profile(decider: Decider, body: Mapping[str, Any], repeat: int = 5) -> Profile:
    """The request's own time and its time by operation, medians of `repeat` (after one untimed answer each way).
    Kept prefixes are cleared before every answer, so each one reads the whole request, and kept answers are not
    used."""

    def answer() -> float:
        if decider.prefixes is not None:
            decider.prefixes.clear()
        start = time.perf_counter()
        with decider.uncached():
            decider.systemone(body)
        return (time.perf_counter() - start) * 1000

    answer()
    own = statistics.median(answer() for _ in range(repeat))
    floor = round_trip()
    timed = TimedOps.of(decider.backbone.ops)
    with decider.with_ops(timed):
        answer()
        totals, per_op, calls = [], defaultdict(list), {}
        for _ in range(repeat):
            timed.clock.clear()
            timed.calls.clear()
            totals.append(answer())
            for name, seconds in timed.clock.items():
                per_op[name].append(seconds * 1000)
            calls = dict(timed.calls)
    ops = {k: (calls[k], statistics.median(v)) for k, v in per_op.items()}
    return Profile(own, statistics.median(totals), floor, ops)


PHASES = ("parse", "render", "plan", "graph", "gpu", "readout", "answer")


@dataclass(frozen=True)
class Phases:
    scenario: str
    input_tokens: int  # of the timed requests, median
    first: Mapping[str, float]  # ms by phase, the scenario's first request
    median: Mapping[str, float]  # ms by phase, median of the rest
    cpu: Mapping[str, float]  # the main thread's CPU ms by phase (Python and MLX's graph building), median
    tokenize_ms: float  # encoding the prompts' text again (decoded from their tokens), timed apart, median

    def table(self) -> str:
        total = sum(self.median.values())
        first = sum(self.first.values())
        cpu = sum(self.cpu.values())
        lines = [
            f"{self.scenario}: {self.input_tokens} input tokens; {total:.1f} ms per request, {cpu:.1f} ms of it the "
            f"main thread's CPU time (the first request: {first:.1f} ms)",
            f"{'phase':<8}  {'ms':>8}  share  {'cpu ms':>8}  {'first':>8}",
        ]
        for name in PHASES:
            ms = self.median[name]
            lines.append(f"{name:<8}  {ms:8.2f}  {ms / total:5.1%}  {self.cpu[name]:8.2f}  {self.first[name]:8.2f}")
            if name == "render":
                lines.append(f"  (tokenizing the prompts' text, timed apart: {self.tokenize_ms:.2f} ms)")
        return "\n".join(lines)


def _phased(decider: Decider, body: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], int, float]:
    """`body` answered by the decider's own path, timed step by step through runtime.PHASE: the wall and the main
    thread's CPU ms by phase, the input tokens, and the ms tokenizing the prompts' text again takes."""
    wall, cpu = dict.fromkeys(PHASES, 0.0), dict.fromkeys(PHASES, 0.0)
    last = [time.perf_counter(), time.thread_time()]

    def mark(name: str) -> None:
        now = [time.perf_counter(), time.thread_time()]
        wall[name] += (now[0] - last[0]) * 1000
        cpu[name] += (now[1] - last[1]) * 1000
        last[:] = now

    token = PHASE.set(mark)
    try:
        request = parse_request(body)
        mark("parse")
        with decider.uncached():
            result = decider.answer(request)
    finally:
        PHASE.reset(token)
    return wall, cpu, int(result.response["usage"].get("input_tokens", 0)), _tokenize_ms(decider, request)


def _tokenize_ms(decider: Decider, request: Request) -> float:
    """Encoding the request's prompts' text again (as `Decider.inspect` decodes it), in ms; 0 when the format's
    tokenizer is not ippon's own (a plugin's may be anything)."""
    tok = decider.tokenizer
    if tok is None:
        return 0.0
    texts = [p["text"] for p in decider.inspect(request)["prompts"]]
    start = time.perf_counter()
    tok.backend.encode_batch(texts, add_special_tokens=False)
    return (time.perf_counter() - start) * 1000


def phases(decider: Decider, bodies: Sequence[Mapping[str, Any]], scenario: str) -> Phases:
    """Each of `bodies` answered phase by phase, in order; the first reported apart from the rest. Under "fresh",
    kept prefixes are cleared before each, so every request is read whole even when the same one repeats."""
    if len(bodies) < 2:
        raise InvalidRequest("phases needs at least two requests: the first, and the rest")
    timed, cpus, tokens, tokenizing = [], [], [], []
    for body in bodies:
        if scenario == "fresh" and decider.prefixes is not None:
            decider.prefixes.clear()
        ms, cpu, read, tokenize = _phased(decider, body)
        timed.append(ms)
        cpus.append(cpu)
        tokens.append(read)
        tokenizing.append(tokenize)
    rest, rest_cpu = timed[1:], cpus[1:]
    return Phases(
        scenario,
        int(statistics.median(tokens[1:])),
        timed[0],
        {name: statistics.median(m[name] for m in rest) for name in PHASES},
        {name: statistics.median(m[name] for m in rest_cpu) for name in PHASES},
        statistics.median(tokenizing[1:]),
    )
