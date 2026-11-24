# Using ippon from an agent

Use ippon to answer typed questions about local data with a decision model. Run `ippon recommend --json` and take its
`start_with`, which is the build to start with on this Mac. Then download that build explicitly with
`ippon pull MODEL`. A question can ask for a yes/no probability, a classification among named options, or a score on
an ordered scale. The model returns probabilities and does not generate text. `ippon pull MODEL --dry-run --json`
reports a build from the Hub without a download. Offline, `ippon check MODEL --json` reports a model from its complete
recorded download.

## Skills

Copy or link the folder that `ippon skills --path` prints into a skills folder that your agent reads, such as
`~/.claude/skills` for Claude Code. These skills are guides that ippon ships for agents. `ippon skills` lists them.
`ippon skills ippon` prints the core guide. It tells when a decision model fits, how to write a request, how to ask
and how to act on probabilities.

## One command for a session

`ippon claude`, `ippon codex`, `ippon opencode` and `ippon pi` start a coding agent with the decision model ready for one session.
Each command starts the shared server first, or uses again the server that already runs. It does not change the
agent's own settings.

```sh
ippon claude
ippon codex --model kettei-dot --keep-alive 2h
```

The command uses the model that `ippon setup` chose, unless `--model` names another. If setup did not run, it uses
Kettei Core at 8 bits (`kettei-core:q8`). The command takes the options of `serve` for the server it starts, such as
`--keep-alive` and `--prefix-budget`. Arguments after `--` go to the agent unchanged. The agent finds the server
through `IPPON_URL` and `IPPON_MODEL` in its environment. An `ippon` command that the agent runs in the session runs
the same installation of ippon as the command that started it, even when another one is first on your `PATH`.

| Agent | How ippon adds itself for the session | What the session gets |
|---|---|---|
| Claude Code | `--mcp-config` and `--plugin-dir` in a temporary folder, deleted when it exits | the MCP tools and ippon's skill |
| Codex | `-c` overrides of `mcp_servers.ippon` | the MCP tools, with ippon's guidance as the server's instructions |
| OpenCode | `OPENCODE_CONFIG_CONTENT` | the MCP tools, with ippon's guidance as the server's instructions |
| Pi | `-e` with ippon's extension and `--skill`, in a temporary folder deleted when it exits | the MCP tools, ippon's skill, and the model as a Pi classifier |

In Pi, the model is also a classifier, named `ippon/MODEL`, so codemode scripts (`models.classify`) and extensions
(`ctx.modelRegistry.classify`) can ask it. A model that reads images takes the images a script or extension passes
with its questions.

To have the model in every session, select that option in `ippon setup`. Setup adds the MCP server and skill of ippon
to the agent's own settings, next to your settings. For Pi, setup writes the MCP server into `~/.pi/agent/mcp.json`
and links ippon's extension into `~/.pi/agent/extensions`. When Pi starts, the extension makes the models of the
ippon server that runs at that time into Pi classifiers. `ippon remove claude` (or `codex`, `opencode`, `pi`) removes only these additions.

Ctrl-C goes to the agent, and the agent decides what it means. When the agent exits, the server continues to run for
other clients. `ippon stop` stops the server. If the agent is not installed, the command fails with
`runtime_unavailable` and tells you how to install the agent.

## MCP setup

To use ippon from an MCP host, install the command where your MCP host can find it and download a model. For example,
use [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv tool install ippon
ippon pull kettei-core
```

An agent's shell has no terminal, so ippon never asks there about a model's licence. Show the licence to the user and
ask, then pull with `--accept-license`. A model under Apache-2.0 or MIT needs no acceptance.

For Claude Code, add the local stdio server.

```sh
claude mcp add --transport stdio ippon -- ippon mcp kettei-core:q8
```

Claude Code also accepts a project `.mcp.json` ([examples/claude-code.mcp.json](../examples/claude-code.mcp.json)).
Cursor uses `.cursor/mcp.json` for a project or `~/.cursor/mcp.json` for all projects. This configuration works with
these hosts.

```json
{
  "mcpServers": {
    "ippon": {
      "type": "stdio",
      "command": "ippon",
      "args": ["mcp", "kettei-core:q8"]
    }
  }
}
```

If the desktop host cannot find `ippon`, replace `command` with the absolute executable path from `command -v ippon`.
These are standard local-server configurations. The [Claude Code MCP guide](https://code.claude.com/docs/en/mcp) and
the [Cursor MCP guide](https://prod.cursor.com/docs/mcp) give the host-specific configuration.

For Hermes, add ippon to `mcp_servers` in `~/.hermes/config.yaml`, next to your current servers.

```yaml
mcp_servers:
  ippon:
    command: ippon
    args: ["mcp", "kettei-core:q8"]
```

For another MCP host, configure the stdio command `ippon mcp kettei-core:q8` in the configuration format of that host.
You do not install an additional MCP dependency, because ippon implements the protocol itself. The implementation
supports the tools-only stdio subset of MCP versions `2024-11-05`, `2025-03-26`, `2025-06-18` and `2025-11-25`. This
subset covers initialize, initialized notification, ping, tools/list and tools/call. The implementation sends one JSON
message per line, refuses batches, and returns JSON text (plus structured tool content from `2025-06-18` on). It has no
MCP resources, prompts or remote transport.

## One model server shared by clients

Each MCP host starts a small stdio process, and these processes send decisions to one shared HTTP server. The stdio
processes do not load their own model weights. The first decision can start the server if the configured model and
its dependencies are downloaded. A server that MCP starts has Hub access offline, so a missing dependency fails and
does not start a download. A missing model gives an error that asks for an explicit `pull`.

To start and stop the server yourself, run these commands.

```sh
ippon serve kettei-core:q8 --detach --json
ippon ps --json
ippon decide kettei-core:q8 request.json --server --json
ippon decide d3-mini request.json --image photo.jpg --video clip.mp4 --json
ippon stop --json
```

For multimodal models, pass local files on the command line with `--image PATH`, `--video PATH`, or `--audio PATH`
(each flag repeats), or pass local file paths directly in `images`, `videos`, or `audio` when calling the MCP `decide`
or `lint` tools. Over HTTP, send base64 or `data:` URIs.

The detached command returns the PID, the actual URL and the log path. The default port is 8000. If another program
uses port 8000, ippon serves on a free port and says so. ippon's own commands and MCP server find that server,
and other clients need its URL. Records and logs are in the `servers/` directory of the ippon cache.

`stop` stops the only server running. If more than 1 server runs, `stop MODEL` stops the server of that model, as
`ippon status` names it, and `stop --all` stops every server. `stop` alone then stops nothing and lists the servers,
because another server may run for the `check` command or for an app. A server is shared, so `stop` says how many
clients were connected to the server it stopped, and `--json` gives the number as `connections`. If other work may
use the server, run `ippon status` first.

Automatic clients coordinate the start and use the oldest server that serves the requested model at the requested
precision. ippon never uses a different model or precision in its place. When no running server serves that build,
the client starts one server for that build alone, beside the others, and every later client of that build uses it.
To keep everything in one process, start all the builds you need together, for example
`ippon serve kettei-core kettei-core:q8 d1-omni-2b --detach`.

A server that starts automatically unloads models after 30 minutes without requests. A server that you start
explicitly uses the policy and the `--keep-alive` settings of `serve`. When you close an MCP client, the shared server
does not stop. `GET /v1/status` reports memory, pressure and recent releases. Under hard memory pressure, requests
for a model that is not loaded receive 503 with `Retry-After`. Retry later, and do not start another copy of the
server.

To use a local endpoint that already runs, use this configuration.

```json
{
  "mcpServers": {
    "ippon": {
      "type": "stdio",
      "command": "ippon",
      "args": ["mcp", "--url", "http://127.0.0.1:8000"]
    }
  }
}
```

To see which client asks what, send `X-Ippon-Client: NAME` with each request. The server's log and `ippon status` then
count the questions answered for that name and never show a state. ippon's own clients send the `IPPON_CLIENT` variable, and `ippon
claude`, `ippon codex`, `ippon opencode` and `ippon pi` set it to the agent's name.

If the server requires a key, put `IPPON_API_KEY` in the environment of the MCP process. Do not put the key in
committed configuration. The client sends the key as a Bearer token. This client accepts local HTTP endpoints and
refuses redirects. Its stdout is only for MCP, and diagnostics go to stderr.

## Tools and questions

The MCP server gives 4 tools. `decide` asks questions, and the other tools help you select a build and check a
request.

| Tool | Input | Result |
|---|---|---|
| `decide` | optional `model`, `state` (or `states`, a list, to ask many at once), named `questions`, optional supported media | System One response with `answers`, `usage` and provenance `diagnostics`. With `states`, `results` in order, each with its `answers` or its `error` |
| `models` | `{}` | known builds, download availability and the shared server's loaded models |
| `recommend` | optional `task`: `text`, `routing`, `classification`, `moderation`, `extraction`, `images`, `audio` or `videos` | builds ordered by published support and fit, not a task-accuracy ranking |
| `lint` | a decision request | `{ "ok": boolean, "problems": [...] }` |

Each question has an `instructions` string and one of these types.

- `noul` is yes/no, and the `noul` of the answer is P(yes). Optional criteria describe `true` and `false`.
- `choice` takes named options in `criteria`, as a list or a mapping of labels to descriptions. The answer contains
  `choice`, `confidence` and the `probabilities` of every label.
- `score` takes 2 or more ordered levels in `criteria` (as many as the model takes), from lowest to highest. The
  answer contains the `score` that the model's creator gives (for d1, the median level), its `confidence`, the
  `legend` and the `probabilities` of every level. In Python, `answer.expected_level` is their mean.

For example, use these arguments with `decide` or save them as `request.json`.

```json
{
  "state": "Order #48213 was charged twice. Please reverse the second charge.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {"billing": "Charges and refunds", "shipping": "Delivery issues", "other": "Anything else"}
    }
  }
}
```

Ask all questions about one state together. To ask the same questions about many states, give `states` in one
`decide` call, or run `ippon decide MODEL request.json --states states.jsonl --server`. One call or one run asks every
state, so the agent does not make a round trip for each state.

Run `lint` when you write a new request. It finds vague or duplicate instructions, overlapping options and missing
catch-all choices. Examine `truncated` in responses. If ippon must not cut inputs, start the server with
`--truncation error`. The selected build must support the media you send, and ippon does not fetch image URLs.
`diagnostics` identifies the actual build and execution path.

## HTTP and Python

Send requests to the URL that `serve --detach` returns (8000 below is an example).

```sh
curl -sS 'http://127.0.0.1:8000/v1/systemone?diagnostics=1' \
  -H 'Content-Type: application/json' --data-binary @request.json
curl -sS http://127.0.0.1:8000/v1/status
```

Python can use the same HTTP pool with the standard library.

```python
import json
from urllib.request import Request, urlopen

with open("request.json") as f:
    body = json.load(f)
request = Request("http://127.0.0.1:8000/v1/systemone?diagnostics=1",
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urlopen(request, timeout=120) as response:
    result = json.load(response)
print(result["answers"])
```

If several clients must share one model pool, use HTTP. The direct Python `Decider.load` API also works, but it loads
weights into that Python process.

HTTP clients use one connection again for each calling thread, so repeated requests do not use a new local port each
time. ippon closes a failed connection, and the next call opens a new connection. ippon never sends a request again
automatically. When you no longer need the client, call `Client.close()` to release the connection of the current
thread.

## Errors and retrying

ippon reports each failure with a type, and the [error table](errors.md) gives the action for each type. Use the table
to separate invalid inputs, unsupported builds, memory limits and server failures that you can retry. In particular,
follow HTTP `Retry-After`, and do not start another model copy.

CLI commands with `--json` return typed failures as
`{"error":{"type":"invalid_request","message":"...","exit_code":3}}` and exit nonzero.
`decide --states` returns one JSON line for each input line, and this includes failures of individual rows.
MCP tool failures set `isError: true` and carry the same typed error in `structuredContent`. Malformed protocol
requests use JSON-RPC errors. HTTP errors use `{"error":{"type":"...","message":"..."}}`.

See [architecture.md](architecture.md#http-server) for the full CLI and HTTP contract, including authentication and
request deadlines. See [errors.md](errors.md) for the exit-code and HTTP status tables. ippon never changes precision
or turns on experimental kernels by itself, under memory pressure or at other times, and leaves both decisions to you.
