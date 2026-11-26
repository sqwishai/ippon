"""The HTTP server on the toy model: answers equal the decider's own, errors stay with their request, and requests
that arrive while the model is busy are answered together."""

from __future__ import annotations

import functools
import gzip
import http.client
import importlib
import io
import json
import queue
import socket
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, suppress
from pathlib import Path
from typing import Any

import pytest

from ippon import Decider, Loaded, daemon, mcp, serve
from ippon.adapter import Rendering
from ippon.cli import main
from ippon.daemon import Client
from ippon.device import max_body
from ippon.errors import DecisionError, InsufficientMemory, InvalidRequest, ModelNotFound, Overloaded
from ippon.pressure import Level, Monitor, Reading
from ippon.schema import Result, parse_request
from ippon.serve import Batcher, Model, Models, _hold, answered_line, make_server
from ippon.served import ServedModel
from ippon.sglang import from_sglang
from test_decider import REQUEST
from toy import ToyBackbone, ToyFormat
from toy import decider as toy_decider


def toy(**options: Any) -> Decider:
    return toy_decider(**({"name": "toy", "prefix_budget": 0} | options))


@pytest.fixture
def running() -> Iterator[tuple[str, Batcher, Decider]]:
    model = toy()
    address, batcher, stop = serving(model, log=True)
    try:
        yield address, batcher, model
    finally:
        stop()
        model.close()


@pytest.fixture
def url(running: tuple[str, Batcher, Decider]) -> str:
    return running[0]


def call(
    url: str, path: str, body: Any = None, encoding: str | None = None, headers: dict[str, str] | None = None
) -> tuple[int, Any]:
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    headers = {"Content-Type": "application/json", **(headers or {})}
    if encoding is not None and data is not None:
        zstd = importlib.import_module("compression.zstd" if sys.version_info >= (3, 14) else "backports.zstd")
        data, headers["Content-Encoding"] = (gzip if encoding == "gzip" else zstd).compress(data), encoding
    status, _, response = send(url, "GET" if data is None else "POST", path, headers, data)
    return status, response


def logged(capsys: pytest.CaptureFixture[str], text: str) -> str:
    """What the server has logged once `text` is in it: a handler writes its line after the response is sent, so
    the client can hold its answer before the line exists."""
    seen, until = "", time.monotonic() + 10
    while text not in seen and time.monotonic() < until:
        seen += capsys.readouterr().err  # each read takes what was written since the last
        time.sleep(0.01)
    assert text in seen
    return seen


def test_the_server_answers_as_the_decider_does(url: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Both routes give the decider's own body, with how it was answered only when `?diagnostics=1` asks; with
    `log`, a line per answer and per error for a person watching, never the state's text."""
    status, body = call(url, "/v1/systemone", REQUEST)
    assert status == 200
    assert body == toy().systemone(REQUEST)
    line = logged(capsys, " tok")
    assert " ms" in line
    assert " tok" in line
    assert "kept" not in line  # a first reading
    assert REQUEST["state"] not in line
    assert all(f"{qid} " in line for qid in body["answers"])
    assert call(url, "/v1/systemone", {"state": "s", "questions": {}})[0] == 400
    assert " 400 /v1/systemone: " in logged(capsys, " 400 ")
    assert "diagnostics" not in body  # the model's server's body, as it is
    assert call(url, "/v1/decisions", REQUEST) == (200, body)  # Perplexity's Decisions API path, the same schema
    assert call(url, "/v1/systemone", REQUEST, headers={"Authorization": "Bearer x"}) == (200, body)  # no key: ignored
    status, asked = call(url, "/v1/systemone?diagnostics=1", REQUEST)
    assert status == 200
    diagnostics = {"precision": "bf16", "truncated": False, "fallback": None, "kept_tokens": 0}
    assert asked == body | {"diagnostics": diagnostics}
    assert call(url, "/health") == (200, {"status": "ok", "waiting": 0})
    status, models = call(url, "/v1/models")
    assert models["data"][0]["id"] == "toy"
    assert models["data"][0]["media"] == []  # a text model's; Pi's extension offers images only where one is listed


def test_states_sent_to_a_server_are_read_together_and_answered_in_line_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`decide --states --server` writes every line's answer in line order, a bad line's error its own. It asks one
    line at a time, so each answer takes the model's own time and other callers aren't held behind a queue; --batch
    keeps that many in flight, and the server reads those that arrive together in one batch."""
    model = toy()
    url, batcher, stop = serving(model, window_ms=200)  # a batch gathers what arrives within 200 ms of its first
    states = [f"Ticket {n}: I was charged twice." for n in range(12)]
    path = tmp_path / "states.jsonl"
    path.write_text("\n".join([*map(json.dumps, states[:5]), "{oops", *map(json.dumps, states[5:])]))
    asked = json.dumps({"questions": REQUEST["questions"]})
    try:
        assert main(["decide", "toy", asked, "--states", str(path), "--server", url]) == 1  # the bad line failed
        by_line = max(batcher.batches)
        batcher.batches.clear()
        assert main(["decide", "toy", asked, "--states", str(path), "--server", url, "--batch", "8"]) == 1
        together = max(batcher.batches)
        batcher.batches.clear()
        # the MCP tool takes the states in one call too, and says what they share once
        tool = mcp.Session(io.BytesIO(), url=url).call
        many = tool("decide", {"model": "toy", "questions": REQUEST["questions"], "states": states})
        by_tool = max(batcher.batches)
        both = tool("decide", {"model": "toy", **REQUEST, "states": states})
    finally:
        stop()
        model.close()
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert out[13:] == out[:13]  # the same answers, however many were in flight
    out = out[:13]
    assert [o["line"] for o in out] == list(range(1, 14))
    assert out[5]["error"]["type"] == "invalid_request"
    expected = [toy().systemone({"state": s, "questions": REQUEST["questions"]})["answers"] for s in states]
    assert [o["answers"] for o in out if "error" not in o] == expected
    assert by_line == by_tool == 1
    assert together > 1
    assert [r["answers"] for r in many["structuredContent"]["results"]] == expected
    assert many["structuredContent"]["usage"]["input_tokens"] == sum(
        o["usage"]["input_tokens"] for o in out[:5] + out[6:]
    )
    assert both["isError"]
    assert "not both" in both["content"][0]["text"]


def test_the_decide_api_answers_over_http_as_in_process(running: tuple[str, Batcher, Decider]) -> None:
    """Sqwish's decide API at /v1/decide gives what `Decider.sqwish` gives in process, timed by the same keys."""
    url, _, model = running
    body = {"context": "Charged twice.", "decisions": [{"id": "refund", "question": "Refund?", "kind": "binary"}]}
    status, answer = call(url, "/v1/decide", body | {"temperature": 2.0})
    here = model.sqwish(body | {"temperature": 2.0})
    assert status == 200
    assert answer["decisions"] == here["decisions"]
    assert answer["timing_ms"].keys() == here["timing_ms"].keys() == {"render", "score"}


def test_an_sglang_request_gets_the_answers_of_the_same_system_one_request(url: str) -> None:
    """At /v1/decisions SGLang's request asks what its System One equivalent asks, and gets those answers in SGLang's
    shape, its temperature dividing the labels' log-probabilities (zstd and gzip bodies too); what ippon can't
    honour is a 400 naming it."""
    state = {"ticket": "Order #48213 was charged twice.", "tier": "gold"}
    sglang = {
        "input": state,
        "questions": [
            {
                "id": "team",
                "type": "choice",
                "question": "Which team?",
                "options": [
                    {"name": "billing", "description": "money"},
                    {"name": "sales"},
                    {"name": "support", "description": {"scope": "anything else"}},
                ],
            },
            {"id": "mood", "type": "score", "question": "How upset?", "levels": ["calm", "annoyed", "furious"]},
            {"id": "refund", "type": "yes_no", "question": "Is a refund asked for?", "yes": "a refund is named"},
        ],
    }
    one = {
        "state": state,
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "money", "sales": None, "support": {"scope": "anything else"}},
            },
            "mood": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
            "refund": {
                "type": "noul",
                "instructions": "Is a refund asked for?",
                "criteria": {"true": "a refund is named"},
            },
        },
    }
    assert parse_request(from_sglang(sglang)[0]) == parse_request(one)  # the same questions, descriptions and all
    status, expected = call(url, "/v1/decisions", one, encoding="zstd")
    assert (status, expected) == (200, toy().systemone(one))  # System One at this path, as before
    status, answered = call(url, "/v1/decisions", sglang)
    assert status == 200
    want, got = expected["answers"], answered["answers"]
    assert got["team"] == {"type": "choice", "choice": want["team"]["choice"], "label_mass": 1.0} | {
        "probabilities": want["team"]["probabilities"]
    }
    assert got["mood"] == {"type": "score", "score": want["mood"]["score"], "label_mass": 1.0} | {
        "probabilities": want["mood"]["probabilities"]
    }
    assert got["refund"]["probabilities"] == pytest.approx(
        {"yes": want["refund"]["noul"], "no": 1 - want["refund"]["noul"]}
    )
    assert answered["object"] == "decisions"
    assert (answered["model"], answered["prompt_format_version"]) == ("default", 1)
    assert answered["usage"] == {"prompt_tokens": expected["usage"]["input_tokens"], "completion_tokens": 0}
    for other in ("mine", "gpt-4o"):  # a name the server doesn't serve gets no answer labelled with it
        status, refused = call(url, "/v1/decisions", {**sglang, "model": other})
        assert (status, refused["error"]["type"]) == (404, "model_not_found")
    status, hot = call(url, "/v1/decisions", {**sglang, "temperature": 2}, encoding="gzip")
    assert status == 200
    root = {label: p**0.5 for label, p in want["team"]["probabilities"].items()}
    assert hot["answers"]["team"]["probabilities"] == pytest.approx(
        {k: v / sum(root.values()) for k, v in root.items()}
    )
    refused = {
        "images": [{"url": "https://example.com/receipt.png"}],  # never fetched
        "return_prompt_token_ids": True,
        "prompt_format_version": 2,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    for field, value in refused.items():
        status, body = call(url, "/v1/decisions", {**sglang, field: value})
        assert (status, body["error"]["type"]) == (400, "invalid_request"), field


def test_errors_are_the_requests_own(url: str) -> None:
    status, body = call(url, "/v1/systemone", {"state": "s", "questions": {}})
    assert status == 400
    assert body["error"]["type"] == "invalid_request"
    assert call(url, "/v1/systemone", b"{not json")[0] == 400
    assert call(url, "/v1/systemone", [1, 2])[0] == 400
    assert call(url, "/elsewhere")[0] == 404
    for method, path, allow in (("GET", "/v1/decide", "POST, OPTIONS"), ("POST", "/health", "GET, OPTIONS")):
        status, headers, body = send(url, method, path, {"Host": "localhost"}, {} if method == "POST" else None)
        assert (status, headers["Allow"], body["error"]["type"]) == (405, allow, "invalid_request"), path
        assert ".md" not in body["error"]["message"]  # no file of ippon's repository: a user has only the command
    status, headers, body = send(url, "PUT", "/v1/decide", {"Host": "localhost"})  # a method with no handler: JSON
    assert (status, headers["Allow"], body["error"]["type"]) == (405, "POST, OPTIONS", "invalid_request")
    for hostile in ('{"state": "\\ud800", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}',
                    '{"state": "s", "questions": {"\\udc00": {"type": "noul", "instructions": "Is it?"}}}',
                    '{"state": "s", "state": "t", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}',
                    '{"input": "s", "questions": [{"id": "q", "type": "boolean", "question": "Is it?"}], '
                    '"temperature": 1%s}' % ("0" * 400)):  # fmt: skip
        assert call(url, "/v1/decisions", hostile.encode())[0] == 400, hostile[:40]  # the request's fault, not a 500
    host, port = url.removeprefix("http://").rsplit(":", 1)
    connection = http.client.HTTPConnection(host, int(port), timeout=30)  # a body sent in chunks, as some clients do
    data = json.dumps(REQUEST).encode()
    connection.request("POST", "/v1/systemone", iter([data[:7], data[7:]]), {"Host": "localhost"}, encode_chunked=True)
    assert connection.getresponse().status == 200
    for head in (b"Content-Length: -5\r\n\r\n", b"Transfer-Encoding: chunked\r\n\r\n-1\r\n"):  # never read to EOF
        with socket.create_connection((host, int(port)), timeout=30) as raw:
            raw.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n" + head)
            assert raw.recv(64).split(b"\r\n")[0].endswith(b" 400 Bad Request"), head
    assert (
        call(url, "/v1/decisions", {"input": "s"})[1]["error"]["message"]
        == "'questions' must list one question or more"
    )
    blanks: tuple[object, ...] = ("", "  ", {}, [], None)
    for empty in blanks:  # answered from the questions alone, as the creators' code answers it
        assert call(url, "/v1/systemone", {**REQUEST, "state": empty})[0] == 200, empty
    # a text-only model reads a media tag as the state's own text: a command or a page can't switch a check off
    assert call(url, "/v1/systemone", {**REQUEST, "state": "rm -rf ~/.ssh  # see <image:1>"})[0] == 200
    assert call(url, "/v1/systemone", REQUEST)[0] == 200  # still serving


class Gated(Decider):
    """A decider whose first batch waits for a gate, so that requests pile up behind it."""

    def __init__(self, prefix_budget: int = 0) -> None:
        backbone = ToyBackbone()
        super().__init__(Loaded(backbone, ToyFormat(backbone)), prefix_budget=prefix_budget)
        self.armed, self.started, self.gate = False, threading.Event(), threading.Event()
        self.running = False  # inside a pass

    def answer_rendered(self, renderings: Sequence[Rendering]) -> list[Result]:
        self.running = True
        try:
            if self.armed:  # not for the warm-up
                self.started.set()
                self.gate.wait(30)
            return super().answer_rendered(renderings)
        finally:
            self.running = False


@pytest.mark.parametrize(
    ("names", "batches"), [("aaaa", [1, 4]), ("babab", [1, 3, 2])], ids=["single-model", "per-model-oldest-first"]
)
def test_waiting_requests_are_grouped_by_model_without_losing_their_own_errors(names: str, batches: list[int]) -> None:
    gated = Gated()
    models = Models([Model("a", lambda: gated, decider=gated), Model("b", toy)])
    with closing(models), closing(Batcher(models)) as batcher:
        gated.armed = True
        first = batcher.submit({**REQUEST, "model": "a"})
        assert gated.started.wait(30)
        waiting = [batcher.submit({**REQUEST, "model": name}) for name in names]
        bad = batcher.submit({"state": "s", "questions": {}})
        gated.gate.set()
        answer = toy().systemone(REQUEST)["answers"]
        assert first.result(30).response["answers"] == answer
        for name, future in zip(names, waiting, strict=True):
            response = future.result(30).response
            assert (response["model"], response["answers"]) == (name, answer)
        with pytest.raises(InvalidRequest):
            bad.result(30)
    assert list(batcher.batches) == batches


def test_a_window_gathers_the_requests_that_arrive_within_it() -> None:
    """With no window the first request runs alone; with one, a request arriving within it joins the first."""
    import time

    for window, expected in ((0.0, [1, 1]), (300.0, [2])):
        batcher = Batcher(toy(), window_ms=window)
        first = batcher.submit(REQUEST)
        if window == 0.0:
            first.result()
        time.sleep(0.02)
        second = batcher.submit(REQUEST)
        assert first.result() == second.result()
        assert list(batcher.batches) == expected
        batcher.close()


def test_the_concurrent_scenario_times_every_request_and_its_throughput() -> None:
    from ippon import bench

    spec = bench.Spec((10,), (1,), "concurrent", n=8, warmup=1, clients=(1, 4))
    cells = bench.run(toy(), spec)
    assert [c.clients for c in cells] == [1, 4]
    for c in cells:
        assert len(c.milliseconds) == 8
        assert c.throughput > 0
        assert sum(c.batches) == 8
    assert max(cells[0].batches) == 1  # one client: one request at a time
    (capped,) = bench.run(toy(), bench.Spec((10,), (1,), "concurrent", n=8, warmup=1, clients=(4,)), max_batch=2)
    assert max(capped.batches) <= 2  # serve's --max-batch, as bench --max-batch sets it


def test_a_failure_outside_any_request_fails_its_batch_only(monkeypatch: pytest.MonkeyPatch) -> None:
    with closing(Batcher(toy())) as batcher:
        real, calls = batcher._answer, []

        def once(decider: Decider, jobs: list[Any]) -> None:
            calls.append(len(jobs))
            if len(calls) == 1:
                raise RuntimeError("outside any request")
            real(decider, jobs)

        monkeypatch.setattr(batcher, "_answer", once)
        with pytest.raises(RuntimeError, match="outside any request"):
            batcher.submit(REQUEST).result(30)
        assert batcher.alive
        assert batcher.submit(REQUEST).result(30).response["answers"] == toy().systemone(REQUEST)["answers"]


def test_over_http_a_slow_request_times_out_a_full_queue_is_503_and_health_follows_the_model_thread() -> None:
    """A request past its deadline gets a 504 and leaves the queue unread; a full queue and a closed batcher refuse at
    once (503, Retry-After); /health is 200 while the model thread runs and 503 once it has stopped."""
    decider = Gated()
    url, batcher, stop = serving(decider, max_queue=1, timeout_s=0.5)
    decider.armed = True
    blocked = batcher.submit(REQUEST)  # takes the model thread
    assert decider.started.wait(30)
    status, body = call(url, "/v1/systemone", REQUEST)  # waits in the queue past its deadline
    assert (status, body["error"]["type"]) == (504, "timeout")
    assert batcher.waiting == 0  # dropped from the queue on its timeout
    batcher.submit(REQUEST)  # fills the queue
    status, headers, body = send_http(url, REQUEST)
    assert (status, headers["Retry-After"], body["error"]["type"]) == (503, "1", "overloaded")
    status, health = call(url, "/health")
    assert status == 200
    assert health["status"] == "ok"
    decider.gate.set()
    blocked.result(30)
    batcher.close()
    assert list(batcher.batches) == [1, 1]  # the blocked request and the one that filled the queue: never the late one
    assert call(url, "/health")[0] == 503
    status, body = call(url, "/v1/systemone", REQUEST)
    assert (status, body["error"]["type"]) == (503, "unavailable")
    stop()


def test_each_error_keeps_its_http_status_and_type() -> None:
    """Every DecisionError goes over HTTP as its own kind, with its status; the GPU running out of memory is
    `insufficient_memory`, and any other exception `internal`."""
    from ippon.errors import (
        ContextTooLong,
        DecisionError,
        InvalidRequest,
        Overloaded,
        Unavailable,
        UnsupportedModel,
        as_decision_error,
    )

    cases = {
        Overloaded("x"): (503, "overloaded"),
        Unavailable("x"): (503, "unavailable"),
        InsufficientMemory("x"): (503, "insufficient_memory"),
        RuntimeError("[metal::malloc] Resource limit (499000) exceeded."): (503, "insufficient_memory"),
        ContextTooLong("x"): (400, "context_too_long"),
        InvalidRequest("x"): (400, "invalid_request"),
        UnsupportedModel("x"): (400, "unsupported_model"),
        DecisionError("x"): (400, "decision_error"),
        RuntimeError("x"): (500, "internal"),
    }
    for error, (status, kind) in cases.items():
        failed = as_decision_error(error)
        assert (failed.http_status, failed.to_json()["error"]["type"]) == (status, kind), error


class Named(Decider):
    """A toy decider that records the batches it answers, under its name."""

    def __init__(self, name: str, log: list[tuple[str, int]]) -> None:
        backbone = ToyBackbone()
        super().__init__(Loaded(backbone, ToyFormat(backbone)), name=name, prefix_budget=0)
        self.log = log

    def answer_rendered(self, renderings: Sequence[Rendering]) -> list[Result]:
        self.log.append((str(self.name), len(renderings)))
        return super().answer_rendered(renderings)


def model_pool(
    names: str, needed: int = 0, budget: int | None = None
) -> tuple[Models, list[tuple[str, int]], dict[str, list[Decider]]]:
    """Named models, their answered batches and each instance created, in load order."""
    log: list[tuple[str, int]] = []
    made: dict[str, list[Decider]] = {}

    def load(name: str) -> Decider:
        decider = Named(name, log)
        made.setdefault(name, []).append(decider)
        return decider

    return Models([Model(n, functools.partial(load, n), needed=needed) for n in names], budget), log, made


def test_model_routing_lists_lazy_loads_and_keeps_each_models_own_answers() -> None:
    models, log, made = model_pool("ab")
    with closing(models):
        url, _, stop = serving(models)
        try:
            status, refused = call(url, "/v1/systemone", {**REQUEST, "model": "c"})
            assert status == 404
            assert "'c'" in refused["error"]["message"]
            assert not made
            expected = toy().systemone(REQUEST)
            assert call(url, "/v1/systemone", {**REQUEST, "model": "b"}) == (200, expected | {"model": "b"})
            status, listed = call(url, "/v1/models")
            assert status == 200
            assert [(m["id"], m["loaded"]) for m in listed["data"]] == [("a", False), ("b", True)]
            assert [m["name"] for m in listed["models"]] == ["a", "b"]
            assert call(url, "/v1/systemone", REQUEST) == (200, expected | {"model": "a"})
            assert list(made) == ["b", "a"]
            assert [len(instances) for instances in made.values()] == [1, 1]
            assert [name for name, _ in log] == ["b", "a"]
        finally:
            stop()
    assert all(m.decider is None for m in models.models)


def test_models_past_the_budget_unload_the_least_recently_used_first() -> None:
    models, _, made = model_pool("abc", needed=10, budget=25)  # two fit at once
    with closing(models), closing(Batcher(models)) as batcher:
        for name in "abac":  # c needs room: b is the least recently used
            batcher.submit({**REQUEST, "model": name}).result(30)
        assert [m.name for m in models.models if m.decider is not None] == ["a", "c"]
        assert made["b"][0].closed
        assert not made["a"][0].closed
        batcher.submit({**REQUEST, "model": "b"}).result(30)  # back on its next request, a newer one unloading
        assert len(made["b"]) == 2
        assert [m.name for m in models.models if m.decider is not None] == ["b", "c"]


def test_a_request_held_for_another_model_stays_counted_and_cancellable_and_a_bad_one_loads_nothing() -> None:
    a, b, loads = Gated(), Gated(), list[str]()

    def load_c() -> Decider:
        loads.append("c")
        return toy()

    models = Models([Model("a", lambda: a, decider=a), Model("b", lambda: b, decider=b), Model("c", load_c)])
    with closing(models), closing(Batcher(models, max_queue=2)) as batcher:
        a.armed = b.armed = True
        first = batcher.submit({**REQUEST, "model": "a"})
        assert a.started.wait(30)
        to_b, held = batcher.submit({**REQUEST, "model": "b"}), batcher.submit({**REQUEST, "model": "a"})
        a.gate.set()  # b's request starts the next batch; a's, taken while it gathers, is held for a's turn
        assert b.started.wait(30)
        assert batcher.waiting == 1  # held, it still counts against max_queue
        batcher.submit({**REQUEST, "model": "a"})  # one more fits; a third would not
        with pytest.raises(Overloaded):
            batcher.submit({**REQUEST, "model": "a"})
        assert held.cancel()  # not started: a caller that gives up can still drop it
        b.gate.set()
        first.result(30), to_b.result(30)
        with pytest.raises(InvalidRequest):
            batcher.submit({"model": "c", "state": "s"}).result(30)  # no questions: refused before c loads
        assert loads == []


def test_a_model_too_big_alone_is_refused_at_start_and_kept_prefixes_share_the_room() -> None:
    with pytest.raises(InsufficientMemory, match="big needs about"):
        Models([Model("small", toy, needed=10), Model("big", toy, needed=61)], 60)
    models = Models([Model(n, functools.partial(toy, name=n, prefix_budget=100), needed=10) for n in "ab"], budget=60)

    def kept() -> list[int]:  # each model's kept prefixes' budget, 0 while unloaded
        return [p.budget if m.decider and (p := m.decider.prefixes) is not None else 0 for m in models.models]

    models.ready(models.models[0])
    assert kept() == [50, 0]  # alone: the room its weights leave
    models.ready(models.models[1])
    assert kept() == [20, 20]  # 20 + 20 of weights + 2 x 20 kept = the budget
    models.unload(models.models[1])
    assert kept() == [50, 0]
    models.close()


def serving(target: Any, **options: Any) -> tuple[str, Batcher, Callable[[], None]]:
    """A server for `target` on a free port, serving on a thread of its own: its URL, its batcher, and what stops
    it."""
    server, batcher = make_server(target, port=0, **({"monitor": False} | options))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]

    def stop() -> None:
        server.shutdown()
        server.server_close()
        batcher.close()

    return f"http://{host!s}:{port}", batcher, stop


def send(url: str, method: str, path: str, headers: Mapping[str, str], body: Any = None) -> tuple[int, Any, Any]:
    """A request with exactly these headers (Host too, when given): its status, headers and JSON body."""
    host, port = url.removeprefix("http://").rsplit(":", 1)
    connection = http.client.HTTPConnection(host, int(port), timeout=30)
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    connection.request(method, path, data, dict(headers))
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, response.headers, json.loads(data) if data else None


def test_clients_that_connect_while_the_server_is_busy_are_held_not_reset() -> None:
    """Connections arriving while the server's thread can't accept (a long prompt being read) wait for it; with
    socketserver's backlog of 5, macOS reset every one past the fifth."""
    model = toy()
    server, batcher = make_server(model, port=0, monitor=False)
    host, port = server.server_address[:2]
    connections = [http.client.HTTPConnection(str(host), port, timeout=30) for _ in range(32)]
    sent = []
    try:
        for connection in connections:  # nothing accepts yet
            try:
                connection.request("POST", "/v1/systemone", json.dumps(REQUEST).encode(), {"Host": "localhost"})
                sent.append(True)
            except OSError:  # reset
                sent.append(False)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        statuses = [
            connection.getresponse().status if ok else None for connection, ok in zip(connections, sent, strict=True)
        ]
        assert statuses == [200] * 32
    finally:
        for connection in connections:
            connection.close()
        server.shutdown()
        server.server_close()
        batcher.close()
        model.close()


def wait_until(condition: Callable[[], bool], seconds: float = 10.0) -> bool:
    until = time.monotonic() + seconds
    while not condition() and time.monotonic() < until:
        time.sleep(0.01)
    return condition()


def test_a_local_server_refuses_other_hosts_and_origins_and_asks_for_its_key() -> None:
    """On 127.0.0.1 a request for another Host (a web page that rebound its domain to this Mac) is refused, as is
    one from a page whose Origin isn't allowed; an allowed origin gets CORS headers and its preflight answered. With
    a key, every request needs it, sent as TypeSafe's SDK sends one. Nothing refused reaches the model."""
    url, batcher, stop = serving(toy(), api_key="s3cret", allow_origins=["http://app.test"])
    port = url.rsplit(":", 1)[1]
    key = {"Authorization": "Bearer s3cret"}
    for host in ("localhost", f"localhost:{port}", f"127.0.0.1:{port}", "[::1]:8000"):
        assert send(url, "GET", "/health", {"Host": host, **key})[0] == 200, host
    for host in ("evil.test", f"evil.test:{port}", f"localhost.evil.test:{port}", "127.0.0.1.evil.test"):
        status, _, body = send(url, "POST", "/v1/systemone", {"Host": host, **key}, REQUEST)
        assert (status, body["error"]["type"]) == (403, "forbidden"), host
    status, headers, body = send(url, "POST", "/v1/systemone", {"Origin": "http://evil.test", **key}, REQUEST)
    assert (status, body["error"]["type"], headers["Access-Control-Allow-Origin"]) == (403, "forbidden", None)
    status, headers, _ = send(
        url, "OPTIONS", "/v1/systemone", {"Origin": "http://evil.test", "Access-Control-Request-Method": "POST"}
    )
    assert status == 403
    asked = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization, content-type"}
    status, headers, _ = send(url, "OPTIONS", "/v1/systemone", {"Origin": "http://app.test", **asked})
    assert status == 204  # a preflight carries no key
    assert headers["Access-Control-Allow-Origin"] == "http://app.test"
    assert "authorization" in headers["Access-Control-Allow-Headers"].lower()
    for authorization in (None, "Bearer wrong", "s3cret", "Basic s3cret"):
        sent = {} if authorization is None else {"Authorization": authorization}
        status, headers, body = send(url, "POST", "/v1/systemone", sent, REQUEST)
        assert (status, body["error"]["type"], headers["WWW-Authenticate"]) == (401, "unauthorized", "Bearer")
    status, headers, body = send(url, "POST", "/v1/systemone", {"Origin": "http://app.test", **key}, REQUEST)
    assert (status, headers["Access-Control-Allow-Origin"]) == (200, "http://app.test")
    assert body == toy().systemone(REQUEST)
    stop()
    assert list(batcher.batches) == [1]  # the one request let through


def test_the_heartbeat_beats_on_the_model_thread_between_passes_and_rests_after_its_window() -> None:
    """One heartbeat for every model, sent by the thread that runs them and so never inside a pass; it beats for its
    window after the last request (the start counts as one), then rests until the next."""
    from ippon.residency import Heartbeat

    decider = Gated()
    beats: list[tuple[str, bool]] = []
    heartbeat = Heartbeat(0.01, 0.3, lambda: beats.append((threading.current_thread().name, decider.running)))
    with closing(Batcher(decider, heartbeat=heartbeat)) as batcher:
        assert wait_until(lambda: heartbeat.beats > 0)
        time.sleep(0.5)  # past the window
        rested = heartbeat.beats
        decider.armed = True
        first = batcher.submit(REQUEST)
        assert decider.started.wait(30)
        time.sleep(0.2)  # inside the pass: no beat
        assert heartbeat.beats == rested
        decider.gate.set()
        first.result(30)
        assert wait_until(lambda: heartbeat.beats > rested)  # a request: it beats again
    assert beats
    assert all(name == "ippon-model" and not running for name, running in beats)


def test_an_idle_model_unloads_and_a_load_out_of_memory_gives_back_what_it_held() -> None:
    """With keep-alive a model that answers nothing for that long unloads, and loads again on its next request; a
    load that runs out of GPU memory is a 503 to retry, and what it made is freed before the error is answered."""
    import argparse

    from ippon.cli import _keep_alive, _policy

    never = float("inf")
    assert [_keep_alive(t) for t in ("10m", "90", "1.5h", "0", "-1", "forever")] == [600, 90, 5400, 0, never, never]
    unset = {"policy": "balanced", "keep_awake": None, "keep_awake_window": None, "keep_alive": None}

    def policy(**given: Any) -> tuple[float | None, float]:
        return _policy(argparse.Namespace(**(unset | {"no_keep_awake": False} | given)))

    assert policy(policy="battery", keep_awake=2.0) == (300, 300)  # a flag given overrides the preset
    assert policy(policy="latency", keep_awake_window=30.0, keep_alive=60.0) == (30, 60)
    assert policy(no_keep_awake=True) == (None, never)
    models, _, made = model_pool("a")
    models.ready(models.models[0])
    batcher = Batcher(models, keep_alive=0.1)
    batcher.submit(REQUEST).result(30)
    assert wait_until(lambda: made["a"][0].closed)
    assert models.models[0].decider is None
    assert batcher.submit(REQUEST).result(30).response == toy(name="a").systemone(REQUEST)
    assert len(made["a"]) == 2  # loaded again on its next request
    batcher.close()
    models.close()

    class Weights:
        pass

    held: list[weakref.ref[Weights]] = []

    def too_big() -> Decider:
        weights = Weights()
        held.append(weakref.ref(weights))
        raise RuntimeError("[metal::malloc] Attempting to allocate 90000000000 bytes which is greater than the maximum")

    url, _, stop = serving(Models([Model("big", too_big)]))
    status, headers, body = send(url, "POST", "/v1/systemone", {"Host": "localhost"}, REQUEST)
    assert (status, body["error"]["type"], headers["Retry-After"]) == (503, "insufficient_memory", "10")
    assert held[0]() is None  # freed, though the batch's requests still hold the error
    stop()


def test_a_request_whose_caller_hangs_up_while_it_waits_is_never_read() -> None:
    decider = Gated()
    server, batcher = make_server(decider, port=0, monitor=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    decider.armed = True
    blocked = batcher.submit(REQUEST)  # takes the model thread
    assert decider.started.wait(30)
    data = json.dumps(REQUEST).encode()
    with socket.create_connection((str(host), port)) as client:
        client.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: %d\r\n\r\n" % len(data))
        client.sendall(data)
        assert wait_until(lambda: batcher.waiting == 1)
    assert wait_until(lambda: batcher.waiting == 0)  # noticed the hang-up and dropped it
    decider.gate.set()
    blocked.result(30)
    batcher.close()
    assert list(batcher.batches) == [1]  # the blocked request only: the dropped one was never read
    server.shutdown()
    server.server_close()


def scripted(memory: int = 16 << 30) -> tuple[Callable[..., int], Monitor]:
    """A monitor of a `memory`-byte Mac whose readings the test sends, each read once (`report(available, kernel)`,
    which returns how many were sent): the first, plenty, already sent for its start."""
    feed: queue.Queue[Reading] = queue.Queue()
    sent = [0]

    def read() -> Reading:
        try:
            return feed.get(timeout=0.02)
        except queue.Empty:
            raise OSError("no reading sent yet") from None

    def report(available: int, kernel: int = 1) -> int:
        feed.put(Reading(0, memory, available, 0, kernel))
        sent[0] += 1
        return sent[0]

    report(8 << 30)
    return report, Monitor(read, busy=0.01, idle=0.01)


def test_short_of_memory_the_model_thread_gives_back_prefixes_then_cache_then_idle_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a 16 GiB Mac (soft under 1.28 GiB available, hard under 0.5, plenty again above 1.92): while memory is
    short the model thread cuts the kept prefixes' shares and empties MLX's cache, and on the next reading unloads
    the least recently used idle model, never the most recently used; very short, it unloads every other idle model
    and a request for an unloaded model gets a 503 while the loaded one goes on answering; plenty again, the shares
    come back. /v1/status shows each step."""
    import mlx.core as mx

    gib, cleared = 1 << 30, list[str]()
    real_clear = mx.clear_cache

    def clear() -> None:
        cleared.append(threading.current_thread().name)
        real_clear()

    monkeypatch.setattr(mx, "clear_cache", clear)
    made: list[Gated] = []

    def gated() -> Decider:
        made.append(Gated(prefix_budget=100))
        return made[-1]

    report, monitor = scripted()
    models = Models(
        [Model("a", gated, needed=10), Model("b", functools.partial(toy, prefix_budget=100), needed=10)], budget=60
    )
    models.ready(models.models[0])  # as `serve` loads its first model
    url, batcher, stop = serving(models, monitor=monitor)

    def settle(sent: int) -> None:  # every reading sent is read, and acted on by the model thread
        assert wait_until(lambda: batcher._seen == monitor.readings == sent)

    def steps() -> list[tuple[str, str, str | None]]:
        return [(e["pressure"], e["step"], e.get("model")) for e in batcher.given_back]

    def memory() -> list[tuple[str, bool, int]]:
        listed = call(url, "/v1/status")[1]["models"]
        return [(m["name"], m["loaded"], m["prefix_budget"]) for m in listed]

    answer = toy().systemone(REQUEST)["answers"]
    for name in "ba":  # b loads; a answers last, the most recently used
        assert batcher.submit({**REQUEST, "model": name}).result(30).response["answers"] == answer
    status, before = call(url, "/v1/status")
    assert (status, before["memory"]["pressure"], before["given_back"]) == (200, "normal", [])
    assert memory() == [("a", True, 20), ("b", True, 20)]  # 60 - 2 x 10 of weights, shared
    settle(report(gib))  # short
    assert steps() == [("soft", "prefixes", None), ("soft", "cache", None)]
    assert memory() == [("a", True, 5), ("b", True, 5)]  # a quarter each, both still loaded
    settle(report(gib))  # still short on the next reading: the least recently used idle model goes
    assert steps()[2:] == [("soft", "unload", "b")]
    settle(
        report(gib + gib // 2)
    )  # above soft's line, below plenty's: still short, and a, the most recently used, stays
    assert len(steps()) == 3
    assert memory() == [("a", True, 12), ("b", False, 0)]
    settle(report(4 * gib))  # plenty: the shares come back
    assert steps()[3:] == [("normal", "restore", None)]
    assert memory() == [("a", True, 50), ("b", False, 0)]
    assert batcher.submit({**REQUEST, "model": "b"}).result(30).response["answers"] == answer  # b loads again
    made[-1].armed = True
    running = batcher.submit({**REQUEST, "model": "a"})
    assert made[-1].started.wait(30)
    waiting = batcher.submit({**REQUEST, "model": "a"})  # names a while a's pass runs
    critical = report(4 * gib, kernel=4)  # macOS says critical
    assert wait_until(lambda: monitor.level is Level.HARD)
    made[-1].gate.set()
    assert (
        running.result(30).response["answers"] == waiting.result(30).response["answers"] == answer
    )  # a stayed for its waiting request
    settle(critical)
    assert steps()[4:] == [("hard", "prefixes", None), ("hard", "cache", None), ("hard", "unload", "b")]
    status, headers, body = send_http(url, {**REQUEST, "model": "b"})  # loading b would take more: refused
    assert (status, body["error"]["type"], headers["Retry-After"]) == (503, "overloaded", "5")
    assert send_http(url, {**REQUEST, "model": "a"})[2]["answers"] == answer  # a, loaded, goes on answering
    settle(report(4 * gib, kernel=4))  # still critical, nothing waits: a, the most recently used, stays
    after = call(url, "/v1/status")[1]
    assert after["memory"]["pressure"] == "hard"
    assert after["memory"]["macos"] == "critical"
    assert memory() == [("a", True, 0), ("b", False, 0)]
    assert set(cleared) == {"ippon-model"}  # never the monitor's thread, nor a handler's
    settle(report(4 * gib))
    assert batcher.submit({**REQUEST, "model": "b"}).result(30).response["answers"] == answer  # loads again
    stop()
    models.close()


def send_http(url: str, body: Any) -> tuple[int, Any, Any]:
    return send(url, "POST", "/v1/systemone", {"Host": "localhost"}, body)


def test_pressure_eviction_cannot_race_with_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    report, monitor = scripted()
    models = Models([Model("a", toy), Model("b", toy)])
    batcher = Batcher(models, monitor=monitor)
    entering, release, accepted = threading.Event(), threading.Event(), threading.Event()
    unload = models.unload
    replies: list[Any] = []

    def delayed(model: Model) -> None:
        if model.name == "b" and not entering.is_set():
            entering.set()
            assert release.wait(5)
        unload(model)

    def submit() -> None:
        replies.append(batcher.submit({**REQUEST, "model": "b"}))
        accepted.set()

    submitting = threading.Thread(target=submit)
    try:
        for name in "ba":
            batcher.submit({**REQUEST, "model": name}).result(30)
        monkeypatch.setattr(models, "unload", delayed)
        first = report(1 << 30)
        assert wait_until(lambda: batcher._seen == first)
        report(1 << 30)
        assert entering.wait(5)
        submitting.start()
        assert not accepted.wait(0.1)  # eviction must finish before the new request can enter the queue
    finally:
        release.set()
        if submitting.ident is not None:
            submitting.join(5)
        batcher.close()
    assert len(replies) == 1
    assert replies[0].result().response["answers"] == toy().systemone(REQUEST)["answers"]


def test_sigterm_stops_the_server_as_ctrl_c_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """SIGTERM to a server serving on the main thread (as `ippon serve` runs it) ends serve_forever as Ctrl-C
    does; closing then answers the request the model is reading, refuses one still waiting past the grace (503), stops
    the model and memory threads, hangs up on a client keeping its connection between requests (an agent's) so no
    handler thread outlives the server (one writing its log line as Python exits aborts it), and closes the socket."""
    import http.client
    import os
    import signal

    from ippon import serve

    monkeypatch.setattr(serve, "CLOSE_GRACE_S", 0.2)
    models = Models([Model("toy", Gated)])
    decider = models.ready(models.models[0])
    assert isinstance(decider, Gated)
    _, monitor = scripted()
    server, batcher = make_server(models, port=0, monitor=monitor)
    host, port = server.server_address[:2]
    url = f"http://{host!s}:{port}"
    idle = http.client.HTTPConnection(str(host), port, timeout=30)  # kept open between requests, as agents do
    decider.armed = True
    answers: dict[str, tuple[int, Any]] = {}
    clients = [
        threading.Thread(target=lambda n=n: answers.update({n: call(url, "/v1/systemone", REQUEST)})) for n in "rw"
    ]

    def terminate() -> None:
        idle.request("GET", "/health")
        assert idle.getresponse().read()
        clients[0].start()
        assert decider.started.wait(30)
        clients[1].start()
        assert wait_until(lambda: batcher.waiting == 1)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.6)  # past the grace
        decider.gate.set()

    threading.Thread(target=terminate, daemon=True).start()
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        server.serve_forever()
    assert signal.getsignal(signal.SIGTERM) is before
    server.server_close()  # what `serve` does on Ctrl-C
    for handler in [t for t in threading.enumerate() if "process_request_thread" in t.name]:
        handler.join(1)  # none is left reading a connection when the process exits
        assert not handler.is_alive()
    for client in clients:
        client.join(30)
    assert (answers["r"][0], answers["r"][1]["answers"]) == (200, toy().systemone(REQUEST)["answers"])
    assert (answers["w"][0], answers["w"][1]["error"]["type"]) == (503, "unavailable")
    assert idle.sock is not None
    idle.sock.settimeout(1)
    assert idle.sock.recv(1) == b""  # the server hung up on it
    idle.close()
    assert not batcher.alive
    stopped = monitor.readings
    time.sleep(0.05)
    assert monitor.readings == stopped  # its memory thread reads no more
    assert models.models[0].decider is None
    assert decider.closed
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection((str(host), port), timeout=5)


def test_pressure_restores_a_preloaded_deciders_prefix_budget() -> None:
    decider = Gated(prefix_budget=100)
    report, monitor = scripted()
    with closing(Batcher(decider, monitor=monitor)) as batcher:
        soft = report(1 << 30)
        assert wait_until(lambda: batcher._seen == soft)
        assert decider.prefixes is not None
        assert decider.prefixes.budget == 25
        restored = report(4 << 30)
        assert wait_until(lambda: batcher._seen == restored)
        assert decider.prefixes.budget == 100
    assert not decider.closed  # a preloaded decider still belongs to its caller
    decider.close()


def test_a_port_in_use_keeps_the_bind_error_and_starts_no_monitor() -> None:
    readings: list[bool] = []

    def read() -> Reading:
        readings.append(True)
        return Reading(0, 16 << 30, 8 << 30, 0, 1)

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(OSError, match="Address already in use"):
            make_server(toy(), port=port, monitor=Monitor(read))
        with pytest.raises(DecisionError, match=f"port {port} on 127.0.0.1 is in use"):
            _hold("127.0.0.1", port)  # what `ippon serve` holds while its first model loads
        for _ in range(3):  # connections this end closes first: they linger in TIME_WAIT once it stops
            with socket.create_connection(("127.0.0.1", port)) as client:
                listener.accept()[0].close()
                client.recv(1)
    assert readings == []
    held = _hold("127.0.0.1", port)  # a server just stopped leaves no port in use: restarting it at once works
    assert held is not None
    held.close()


def test_shutdown_cannot_leave_a_concurrently_submitted_request_unanswered(monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import Future

    from ippon.errors import Unavailable

    batcher = Batcher(toy())
    entering, release, closed = threading.Event(), threading.Event(), threading.Event()
    put = batcher._queue.put
    replies: list[Future[Result]] = []

    def delayed(job: Any) -> None:
        entering.set()
        assert release.wait(5)
        put(job)

    def close() -> None:
        batcher.close(grace=0)
        closed.set()

    monkeypatch.setattr(batcher._queue, "put", delayed)
    submitting = threading.Thread(target=lambda: replies.append(batcher.submit(REQUEST)))
    closing = threading.Thread(target=close)
    submitting.start()
    try:
        assert entering.wait(5)
        closing.start()
        assert not closed.wait(0.1)  # close must wait for the accepted job to reach the queue
    finally:
        release.set()
        submitting.join(5)
        if closing.ident is not None:
            closing.join(5)
        batcher.close()
    assert len(replies) == 1
    assert replies[0].done()
    with suppress(Unavailable):  # either answered or explicitly refused, never stranded
        assert replies[0].result().response["answers"] == toy().systemone(REQUEST)["answers"]


def test_a_body_past_the_servers_limit_is_413_and_the_default_follows_the_macs_memory() -> None:
    """The limit counts the body as sent and as decoded (a small gzip of a large body too), and names the option."""
    assert [max_body(gb << 30) >> 20 for gb in (4, 8, 16, 32, 48, 128)] == [128, 256, 512, 1024, 1536, 1536]
    assert max_body(1 << 30) == 64 << 20  # never below 64 MiB
    url, _, stop = serving(toy(), max_body=4096)
    try:
        status, body = call(url, "/v1/systemone", {**REQUEST, "state": "x" * 5000})
        assert status == 413
        assert "--max-body-mb" in json.dumps(body)
        assert call(url, "/v1/systemone", {**REQUEST, "state": "x" * 5000}, encoding="gzip")[0] == 413
        client = Client(url)
        with pytest.raises(InvalidRequest, match="--max-body-mb"):
            client.decide({**REQUEST, "state": "x" * (8 << 20)})
        assert client.decide(REQUEST)["answers"] == toy().systemone(REQUEST)["answers"]
        assert call(url, "/v1/systemone", REQUEST)[0] == 200
    finally:
        stop()


def test_a_request_names_a_served_model_by_any_of_its_names_or_gets_none() -> None:
    """A server of Kettei Dot answers its short name and Hub id, and a request that names no model, but never
    kettei-core or gpt-4o in Dot's place. A name that pins a revision gets that revision's weights or none: a server
    of org/model@AAAA is no server of org/model@BBBB, for its requests and for a client looking for one (`match`)."""
    dot = Model("kettei-dot", lambda: toy(), loads="Sqwish/Kettei-Dot")
    served = Models([dot])
    assert served.named({}) is dot
    assert served.named({"model": "kettei-dot"}) is dot
    assert served.named({"model": "Sqwish/Kettei-Dot"}) is dot
    for other in ("kettei-core", "gpt-4o"):
        with pytest.raises(ModelNotFound):
            served.named({"model": other})
    first, second = "org/model@" + "a" * 40, "org/model@" + "b" * 40
    pinned = Models([Model("pinned", lambda: toy(), loads=first)])
    for same in ("org/model", first, first.upper().replace("ORG/MODEL", "org/model")):
        assert pinned.named({"model": same}).name == "pinned"
        assert daemon.match([ServedModel("pinned", first)], same).id == "pinned"
    for other in (second, "org/model@main"):
        with pytest.raises(ModelNotFound):
            pinned.named({"model": other})
        with pytest.raises(ModelNotFound):
            daemon.match([ServedModel("pinned", first)], other)


def test_a_name_asks_for_one_precision_and_gets_only_that_build() -> None:
    """kettei-core:q8 is Kettei Core at 8 bits and any other name of it the build as published: a server of one never
    answers for the other, and a client finds that out from the server's model list before it asks (`match`)."""
    core = Model("kettei-core", lambda: toy(), loads="Sqwish/Kettei-Core", weights="q8")
    served = Models([core])
    assert served.named({"model": "kettei-core"}) is core  # its own name, as the command check sends it
    assert served.named({"model": "kettei-core:q8"}) is core
    for other in ("Sqwish/Kettei-Core", "kettei-core:q6"):
        with pytest.raises(ModelNotFound):
            served.named({"model": other})
    listed = [ServedModel("kettei-core", "Sqwish/Kettei-Core")]
    assert daemon.match(listed, "kettei-core").id == "kettei-core"
    assert daemon.match(listed, "Sqwish/Kettei-Core").id == "kettei-core"
    for asked, precision in (("kettei-core:q8", "auto"), ("kettei-core", "q8")):
        with pytest.raises(ModelNotFound, match=r"kettei-core:q8 --detach"):
            daemon.match(listed, asked, precision)
    # Unloaded vision models still advertise their supported media so Pi registers them with image input.
    unloaded = Model("d3-mini", lambda: toy(), loads="vllm-sr/d3-mini", media=("images", "videos"))
    assert unloaded.listed().to_json()["media"] == ["images", "videos"]
    assert ServedModel.from_json(unloaded.listed().to_json()).media == ("images", "videos")


def test_json_the_decoder_cannot_take_is_a_bad_request_not_a_dropped_connection(url: str) -> None:
    """Nesting deeper than Python's recursion, or than the renderers' MAX_DEPTH, and an integer past Python's 4,300
    digits each get a 400, and the server answers the next request as before."""
    for body in (b"[" * 100_000 + b"]" * 100_000, b"[" * 101 + b"]" * 101, b'{"state": ' + b"9" * 5000 + b"}"):
        status, refused = call(url, "/v1/systemone", body)
        assert (status, refused["error"]["type"]) == (400, "invalid_request")
    assert call(url, "/v1/systemone", REQUEST)[0] == 200


def test_server_caps_multimodal_batch_size() -> None:
    """A burst of distinct image requests runs in small multimodal batches (<= MAX_MEDIA_BATCH) so early callers do
    not wait behind an entire burst."""
    import base64
    import io
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from ippon.policy import MAX_MEDIA_BATCH

    class VisionToy(ToyFormat):
        @property
        def media(self) -> frozenset[str]:
            return frozenset({"images"})

    backbone = ToyBackbone()
    decider = Decider(Loaded(backbone, VisionToy(backbone)), prefix_budget=0)
    address, batcher, stop = serving(decider, max_batch=32, window_ms=50)
    try:
        bodies = []
        for i in range(12):
            buf = io.BytesIO()
            Image.new("RGB", (2, 2), (i * 20, 0, 0)).save(buf, format="PNG")
            bodies.append(
                {
                    "state": f"photo {i}",
                    "images": [base64.b64encode(buf.getvalue()).decode()],
                    "questions": REQUEST["questions"],
                }
            )
        with ThreadPoolExecutor(max_workers=12) as pool:
            futs = [pool.submit(call, address, "/v1/systemone", b) for b in bodies]
            results = [f.result() for f in futs]
        assert all(status == 200 for status, _ in results)
        assert batcher.batches
        assert max(batcher.batches) <= MAX_MEDIA_BATCH
    finally:
        stop()
        decider.close()


def test_a_body_that_expands_past_any_real_json_is_refused(url: str) -> None:
    """A compressed body may expand MAX_EXPANSION times its size: 20 MB of one letter, which zstd holds in a few KB,
    is refused before it is decoded to the body limit."""
    bomb = b'{"state": "' + b"a" * 20_000_000 + b'", "questions": {}}'
    status, refused = call(url, "/v1/systemone", bomb, encoding="zstd")
    assert status == 413
    assert "expands more than" in refused["error"]["message"]
    assert call(url, "/v1/systemone", REQUEST, encoding="gzip")[0] == 200  # real JSON, compressed, as before


def test_a_server_off_loopback_needs_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """--host 0.0.0.0 reaches the network, where the Host check, which stops DNS rebinding on loopback, does not
    apply: serve refuses it without an API key, before anything starts."""
    monkeypatch.delenv("IPPON_API_KEY", raising=False)
    assert main(["serve", "kettei-core", "--host", "0.0.0.0", "--detach"]) == InvalidRequest.exit_code
    assert main(["serve", "--host", "0.0.0.0", "--detach"]) == InvalidRequest.exit_code  # setup's model, unnamed
    daemon.require_key_off_loopback("0.0.0.0", "a-key")
    daemon.require_key_off_loopback("127.0.0.1", None)


def test_status_counts_the_clients_attached_and_the_answers_a_second(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    host, port = url.removeprefix("http://").rsplit(":", 1)
    clients = [http.client.HTTPConnection(host, int(port), timeout=30) for _ in range(3)]
    try:
        for i, client in enumerate(clients):  # each keeps its connection open, as ippon's own clients do
            # the first gives its name, with a terminal escape that the log and status never show
            named = {"X-Ippon-Client": "history-reader\x1b[2J"} if i == 0 else {}
            for _ in range(2):
                client.request(
                    "POST", "/v1/systemone", json.dumps(REQUEST), {"Content-Type": "application/json"} | named
                )
                assert client.getresponse().read()
        traffic = call(url, "/v1/status")[1]["traffic"]
        assert traffic["connections"] == 3  # the three, not the one asking
        assert traffic["answered"] == 6
        assert traffic["questions"] == 18  # 3 questions a request
        assert traffic["answers_per_second"] > 0
        assert traffic["tokens_per_second"] > 0
        assert traffic["by_client"] == {"history-reader2J": 6}  # questions, as the bars beside the rate count them
        assert traffic["tokens"] > 0
        monkeypatch.setenv("IPPON_CLIENT", "repeat-fixes")  # ippon's own client sends the name it is given
        Client(url).call("/v1/systemone", REQUEST)
        assert call(url, "/v1/status")[1]["traffic"]["by_client"]["repeat-fixes"] == 3
    finally:
        for client in clients:
            client.close()


def test_status_keeps_a_finished_jobs_pace_until_its_answers_age_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job of 3 questions a request, one request a second for 10 s, each taking the model 60 ms: its pace and its
    20 ms per decision while it runs, the same while its answers are still in the status window after it ends (not
    that pace diluted by the idle seconds since), none once they have left it."""
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    traffic = serve._Traffic()
    for second in range(11):
        clock[0] = float(second)
        traffic.answered("job", tokens=100, questions=3, ms=60.0)
    assert traffic.now()["questions_per_second"] == 3.3  # 33 questions over the 10 s from the first answer to the last
    assert traffic.now()["ms_per_decision"] == 20.0
    clock[0] = 35.0  # the job ended 25 s ago: answers from 5 s on are still in the 30 s window
    idle = traffic.now()
    assert idle["questions_per_second"] == 3.6  # 18 questions from 5 s to 10 s: the job's own pace, not 18 over 30 s
    assert idle["questions"] == 33
    assert idle["ms_per_decision"] == 20.0
    clock[0] = 41.0
    assert traffic.now()["questions_per_second"] == 0
    assert (traffic.now()["ms_per_decision"], traffic.now()["last_answer_s"]) == (20.0, 31.0)  # quiet: its last speed
    unwatched = serve._Traffic()  # a server nobody asks the status of: what it keeps ages out as it answers too
    for second in range(11):
        clock[0] = float(second)
        unwatched.answered("job", tokens=100, questions=3, ms=60.0)
    clock[0] = 100.0
    unwatched.answered("late", tokens=100, questions=3, ms=600.0)  # one slow answer long after
    clock[0] = 200.0
    assert unwatched.now()["ms_per_decision"] == 200.0  # its pace, never a median over every answer it ever sent
    for client in ("job", "job", "batch"):  # three requests in progress from two clients: two clients asking
        traffic.asking(+1, client)
    assert (traffic.now()["in_progress"], traffic.now()["clients_asking"]) == (3, 2)


def test_the_feed_shows_every_answer_whole_at_any_width() -> None:
    """Ten questions about one commit outgrow a terminal's line: the feed goes on under it instead of cutting them."""
    questions = {f"{qid}{i}": q for i in range(4) for qid, q in REQUEST["questions"].items()}
    response = toy().systemone({**REQUEST, "questions": questions})
    for width in (80, 120, 200):
        lines = answered_line(response, 420.0, "agent-1", width=width).split("\n")
        assert len(lines) > 1
        assert max(map(len, lines)) <= width
        assert all(f"   {qid} " in "\n".join(lines) for qid in questions)
        assert "…" not in "\n".join(lines)
    # the model's time per decision, never the queue's or the HTTP exchange's, which a long wait says apart
    n = len(questions)
    line = answered_line(response, 4300.0 + 10 * n, "agent-1", waited=4300.0)
    assert "   10 ms" in line
    assert "4.3 s queued" in line
    line = answered_line(response, 8.0 + 52 * n, "agent-1", waited=8.0)  # even when the wait is too short to say
    assert "   52 ms" in line
    assert "queued" not in line


def test_server_memory_budget_applies_mlx_limits_and_aborts_over_budget_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When enforce_memory=True, the batcher sets MLX's process-wide memory and cache limits to the server budget,
    restores them on shutdown, and aborts a forward pass with InsufficientMemory (503) if active memory exceeds it."""
    import mlx.core as mx

    original_limit = mx.get_memory_limit()
    target = 2 << 30  # 2 GiB
    registry = Models(
        [Model("a", functools.partial(toy, name="a", prefix_budget=0), needed=10)], target, enforce_memory=True
    )
    batcher = Batcher(registry)
    try:
        # Submitting one normal request synchronizes with _run() so set_memory_limit has definitely landed.
        result = batcher.submit(REQUEST).result(timeout=10)
        assert result.answers["refund"].p_yes is not None
        assert mx.get_memory_limit() == target

        # Simulate active GPU memory crossing the server budget during a forward pass: STOP must fail it with 503.
        monkeypatch.setattr(mx, "get_active_memory", lambda: target + (1 << 20))
        with pytest.raises(InsufficientMemory, match="memory budget"):
            batcher.submit(REQUEST).result(timeout=10)
    finally:
        batcher.close()
    assert mx.get_memory_limit() == original_limit
