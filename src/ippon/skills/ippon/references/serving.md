# Serving

Agents, scripts and MCP hosts send requests to a running server instead of loading weights themselves. Every client
that asks for the same model at the same precision shares one server. A model or precision that no running server
has starts its own server beside the others.

```sh
ippon serve kettei-core:q8 --detach --json   # its PID, URL and log; 8000, or a free port when 8000 is taken
ippon status --json                    # servers, loaded models, memory
ippon decide kettei-core:q8 request.json --server --json
ippon stop kettei-core:q8 --json       # that model's server; alone, the only one; --all for every one
```

- To serve every model you need from one server, run `ippon serve kettei-core d1-omni-2b --detach`. If a server is
  asked for a model it does not serve, it answers `model_not_found` and never substitutes another.
- The HTTP endpoints are `POST /v1/decide` (the Sqwish Labs SDK's format), `POST /v1/systemone` (TypeSafe's System One format, where `?diagnostics=1` names the build
  that answered), `POST /v1/decisions` (Perplexity's and SGLang's formats), `GET /v1/models`, `GET /v1/status` and
  `GET /health`.
- For MCP, `ippon mcp MODEL` gives the tools `decide`, `models`, `recommend` and `lint`. It starts the shared server
  on the first decision and never downloads anything.
- Send concurrent requests instead of queueing them, because the server answers requests that arrive together in
  shared passes.
- If the server has `IPPON_API_KEY` set, send the key as `Authorization: Bearer ...`.
- If you send `X-Ippon-Client: NAME`, the server's log and `ippon status` count that client's answers by name, and
  they never show its state. ippon's own clients send `IPPON_CLIENT`, which `ippon claude` and the other launchers
  set to the agent's name.
- `?diagnostics=1` adds `kept_tokens`, the number of input tokens that came from an earlier reading.

## Errors

Every failure is JSON with a `type` and a `message`. The command line's `--json` and the MCP tools add an `exit_code`.

| Code | Type | What to do |
|---:|---|---|
| 3 | `invalid_request` | fix the request, and `ippon lint` says what to fix |
| 4 | `unsupported_model` | pick another build with `ippon recommend` |
| 5 | `context_too_long` | send less state |
| 6 | `insufficient_memory` | ask the user, then use a smaller build that the message names |
| 7 | `unsupported_profile` | leave out the chip profile or operation switch the message names |
| 8 | `overloaded` | retry after `Retry-After`, and never start a second server |
| 9 | `unavailable` | the server is closing or stopped: `ippon status` says which, then start it again |
| 10 | `model_not_found` | restart the server with that model too |
| 11 | `license_not_accepted` | ask the user to accept the licence |
| 12 | `insufficient_disk` | ask the user to free space |
| 13 | `download_failed` | repeat the pull, which resumes |
| 14 | `runtime_unavailable` | install or repair what the message names (a library, or the coding agent) |
