# Installation and serving checks

These reports record reliability checks and make no benchmark speed claim. See
[the release procedure](../../docs/releasing.md) for the automated lifecycle and sustained-load commands.

## Installation matrix

[installation.m5-pro.json](installation.m5-pro.json) records fresh wheel installations on 6 November 2026, on an
Apple M5 Pro with 48 GB and macOS 27.0.1. Every environment used the existing lockfile and MLX 0.32.4.

| Python | NumPy | Installation and repair |
|---|---|---|
| 3.11.16 | 2.4.6 | passed |
| 3.12.14 | 2.5.4 | passed |
| 3.13.16 | 2.5.4 | passed |
| 3.14.8 | 2.5.4 | passed |

Each check installed the built wheel into a new environment and verified its import location. It exercised the
errors for damaged checkpoints, the actual removal and restoration of MLX and NumPy, missing and damaged native
libraries, and the repaired CLI startup and doctor output. Both the console command and `python -m ippon` reported
typed missing-dependency errors. The report records the SHA-256 of the wheel and the installed dependency versions.

This report covers only installation and dependency repair, so its `serving` and `full_suite` fields are false.

```sh
uv build
uv run python scripts/check_package.py --wheel dist/*.whl --out /tmp/ippon-installation-checks --install-only
```

## Serving reports

[mcp.m5-pro.json](mcp.m5-pro.json) records two MCP clients that share an HTTP model, and a clean shutdown.
[pressure.m5-pro.json](pressure.m5-pro.json) records memory-pressure injection and recovery in a separate process.
`scripts/smoke_serving.py` produces those reports. The pressure report uses injected readings.

`scripts/serving_lifecycle.py` checks concurrent requests against a probability reference and samples health and
memory. Then it exercises forced termination, stale-record cleanup, automatic restart and clean shutdown. Its JSON
report records the duration, client count, answered and rejected requests, probability error and memory samples.
