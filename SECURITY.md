# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/sqwishai/ippon/security/advisories/new), not in a public
issue. Include the version (`ippon --version`), what an attacker controls, and the steps or files that
reproduce it.

We acknowledge reports within 3 working days and keep you informed while we fix them. We credit you in the advisory
and the release notes, unless you would rather not be named. There is no bounty programme.

## Supported versions

Fixes go into the latest release and `main`. Before 1.0, we do not patch older releases, so upgrade to the latest
release.

## Scope

These parts of ippon are in scope.

- Model files. Loading a model directory, a GGUF file or a build that Ollama pulled must not run code or read
  outside the files it names. ippon reads weights from safetensors and GGUF only, and never calls `torch.load`. It
  never downloads `.bin`, `.pt`, `.pth` or `.pkl` files (`downloads.NEVER`). The one exception is a small head that a
  creator publishes only as a pickle and that an adapter names exactly (Kev's `head.pt`). `ptfile.py` reads it
  through a restricted `pickle.Unpickler`. The pickle may name `collections.OrderedDict`,
  `torch._utils._rebuild_tensor_v2` and torch's float, half and double storages, and ippon's own code answers each.
  The reader refuses any other name before it runs, and it reads the tensors from the file's raw storage records.
  ippon's own reader (`gguf/reader.py`) parses GGUF files and refuses unknown tensor types and tensors that run past
  the end of the file. Only the builds pinned in `gguf/builds.py` load, and a download must match its sha256 and a
  local file its header digest. An Ollama build's blobs must match the SHA-256 and length that its manifest gives
  (`ollama.py`), and ippon writes nothing to Ollama's folder. No code from a model repository is imported or run,
  because formats and layers are part of ippon.
- Requests. ippon validates a request body before the model reads it (`schema.parse_request`). It decodes images,
  audio and video only as the formats it names (`media/decode.py`), at most 1 GiB each over HTTP or MCP. It never
  fetches URLs or reads paths from a request.
- The server. `ippon serve` listens on 127.0.0.1 by default. On loopback, it checks the Host against local
  hostnames and rejects browser Origins unless you allow them explicitly. `--api-key KEY` or `IPPON_API_KEY`
  enables Bearer authentication for all requests except OPTIONS preflights. `--max-body-mb` caps compressed and
  decoded request bodies, each on its own, and by default the cap is 1/32 of the Mac's memory, between 64 MiB and
  1.5 GiB. The queue is bounded. Requests have no default deadline, and `--timeout-s` sets one.
- Shared clients. The CLI and MCP HTTP client accepts local endpoints, refuses redirects, and never writes a
  configured API key into server records. Automatic MCP startup requires downloaded models and uses offline Hub
  access. Server records and logs live under the ippon cache, and kept prefixes stay in the server's memory.

These are out of scope.

- Network isolation and encrypted transport around a server that you expose on purpose. `--host 0.0.0.0` makes the
  server reachable from the network. Authentication is optional, and HTTP itself is not encrypted. For that
  deployment, configure the API key and the network and transport controls it needs. The Host and Origin checks
  protect the local endpoint, but they do not replace authentication for clients that are not browsers.
- What a model answers. A decision model can be wrong, and its input can steer it. ippon promises that the model
  answers as its creator's code does ([docs/accuracy.md](docs/accuracy.md)).
- Vulnerabilities in MLX, Pillow, tokenizers or other dependencies. Report those to their projects. We update
  soon after a fix is released.
