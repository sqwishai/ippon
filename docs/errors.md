# Errors and exit codes

Each ippon failure has a type. Find the exit code or the HTTP status in the tables below, then do the action that the
table gives. CLI results go to stdout and diagnostics go to stderr. `prompt` returns JSON, and `decide` returns JSON when it is piped or given `--json`. `decide --states` returns one JSON line for each input line, and this
includes failures of individual rows. Commands that have `--json` also report typed failures as
`{"error":{"type":"invalid_request","message":"...","exit_code":3}}`.
Errors from the argument parser stay command-line usage errors, with exit code 2.

## CLI exit codes

<!-- exit codes: src/ippon/errors.py (tests/test_cli.py checks this table against it) -->
| Code | Kind | Meaning |
|---:|---|---|
| 0 | | completed successfully |
| 1 | `decision_error` | another reported failure, or a verification, diagnostic or lint check that found a problem. With `--server`, a server's `unauthorized`, `forbidden`, `not_found` or `internal` also exits 1 |
| 2 | | incorrect command-line arguments |
| 3 | `invalid_request` | malformed request or an unsupported request feature |
| 4 | `unsupported_model` | no adapter supports the model, variant or supplied files |
| 5 | `context_too_long` | the rendered input exceeds the model's or configured limit |
| 6 | `insufficient_memory` | the requested model or precision does not fit, and the message identifies alternatives where known |
| 7 | `unsupported_profile` | unsupported chip profile or operation switch |
| 8 | `overloaded` | the server queue is full or a model that isn't loaded is asked for under hard memory pressure. With `--server`, a server's `timeout` also exits 8 |
| 9 | `unavailable` | the server is closing, its model thread stopped, or the selected server is unavailable |
| 10 | `model_not_found` | the server is not configured for the requested model |
| 11 | `license_not_accepted` | accept the model licence before downloading new weights |
| 12 | `insufficient_disk` | free enough space on the model cache disk before retrying |
| 13 | `download_failed` | a download stopped or arrived incomplete, so repeat the command to resume it |
| 14 | `runtime_unavailable` | a required dependency, native library or coding agent is missing or cannot load, so install or repair what the message names |

To catch library errors in Python, catch `ippon.errors.DecisionError` or one of its subclasses. Each class gives `kind`
and `exit_code`. Malformed questions, invalid load limits and unknown strategy or truncation values raise
`InvalidRequest`. ippon examines load options before it opens or downloads a model. To prevent `InvalidRequest`, obey
these rules.

- Give `questions` a mapping of names to `YesNo`, `Choice`, `Score` or System One question objects.
  For example, `{"team": Choice("Which team?", ["billing", "sales"])}`.
- Give `Choice` a list or tuple of labels, or a mapping of labels to descriptions. Give `Score` an ordered list
  or tuple of levels, lowest first. A plain string is not a list of options or levels.
- Set `chunk` and `window` to positive integers. Set `prefix_budget` and `answer_cache` to non-negative integers,
  where 0 keeps none. Set `keep_awake` to a finite, non-negative number of seconds, or `None` in Python
  to disable it.
- In `decide_many`, a malformed request names its position, such as `requests[1]` for the second request.
  ippon refuses unknown fields, so a typo cannot drop part of a request without an error.
- Save request files as UTF-8 JSON and state or evaluation files as UTF-8 JSON lines. File errors name the path
  and what to fix. Use `ippon lint request.json` to examine a request before you load a model.

A damaged checkpoint config, shard index, tensor header or tokenizer raises `UnsupportedModel`. If you get this
error, replace the file that it names with a complete copy from its source. ippon does not change checkpoint files.

To resume an interrupted download, repeat the command, and ippon keeps the saved bytes. If a partial file is larger
than its pinned size or fails its content hash, ippon removes that partial file and asks you to repeat the command. If
the disk becomes full during a transfer, free space before you retry. Usable partial files stay available for the
retry.

If a package that ippon needs is not installed, or MLX does not load, ippon raises `RuntimeUnavailable`. This error is
also an `ImportError`, so `except ImportError` around `from ippon import Decider` still catches it. The message tells
how to repair the problem, and commands with `--json` still return a typed error body. `ippon doctor --json` also
examines whether the packaged native library loads.

If ippon raises `InsufficientMemory`, close other apps or make the request smaller, then retry. This error comes from
an allocator failure during model loading or inference. A disconnected or truncated server response raises
`Unavailable`, so clients can retry and do not need to interpret a Python networking exception. If the server refuses
a large upload before the client finishes the upload, the client keeps the error and guidance of the server.

## HTTP errors

Find the HTTP status in this table and do its action. The body of a failed request is
`{"error":{"type":"...","message":"..."}}`.

| HTTP status | Type | Action |
|---:|---|---|
| 400 | `invalid_request`, `context_too_long` | correct the body, supported features or input length |
| 400 | `unsupported_model`, `license_not_accepted`, `insufficient_disk`, `download_failed`, `unsupported_profile`, `runtime_unavailable`, `decision_error` | the server can't load or run the model, so fix what the message names and start it again |
| 401 | `unauthorized` | send the configured `Authorization: Bearer KEY` |
| 403 | `forbidden` | use a permitted Host and Origin |
| 404 | `model_not_found`, `not_found` | select a configured model or supported route |
| 405 | `invalid_request` | send POST to `/v1/decide`, `/v1/decisions` and `/v1/systemone`, and GET to `/health`, `/v1/models` and `/v1/status` |
| 413 | `invalid_request` | reduce the body, or raise the server's limit (`ippon serve --max-body-mb`, by default a 32nd of the Mac's memory, 64 MiB to 1.5 GiB), which counts the body compressed and decoded |
| 415 | `invalid_request` | use a supported `Content-Encoding` |
| 500 | `internal` | inspect the server diagnostics and report the failure |
| 503 | `overloaded`, `unavailable`, `insufficient_memory` | follow `Retry-After` before retrying |
| 504 | `timeout` | the request exceeded `--timeout-s` (no deadline by default) |

Every 503 includes `Retry-After`. The value is normally 1 second, 5 seconds for hard memory pressure, or 10 seconds
for insufficient GPU memory. `/health` reports a server that is unavailable as a status body, not as an error body.
If a request times out or disconnects before inference starts, the server removes it from the queue. Work that
already runs continues to the end, but the server discards its answer. A retry cannot stop that work.

The HTTP type of every library error is its `kind`. When a GPU has no more memory, the server reports
`insufficient_memory`, and it reports all other failures as `internal`. An answer whose probabilities are not finite
raises `NonFiniteAnswer`, an `internal` error. A model loaded with `precision="auto"` in fp16 answers that request
again in bf16, so the caller sees the error only at a precision that was asked for. See
[the server contract](architecture.md#http-server) for authentication, memory policies and shutdown behavior.

## MCP errors

A failed MCP tool call sets `isError: true` and gives `structuredContent.error`, with a type, a message and an exit
code. Malformed protocol messages use JSON-RPC errors. If a model is not downloaded, the error asks for an explicit
`ippon pull`, because automatic MCP startup does not download models. See [the agent guide](agents.md).
