# Kernel audit evidence

The audit of 29 October 2026 traced all 35 Metal sources as of `0c721ba` through their Python launchers to their
production callers. It kept 31 and removed four. `int8_header.metal`, `int8_matmul.metal` and `int8_quantize.metal`
went with the W8A8 precision. `gemm_simd.metal`, the standalone SIMD matmul, went because only its own tests selected
it (`choices.attach` gives M1 to M4 no tiles, and `Ops.mm` calls a custom matmul only with a tile).

The tree holds 27 Metal sources, and [kernels.md](../../docs/kernels.md#every-metal-source) lists each one and what
runs it. The 6 attention sources that the audit kept, `attention_simd.metal` among them, are not in the tree.
`gemm_simd.metal` and `gemm_glu_simd.metal` are in it, and `scripts/measure_kernels.py` and `tests/test_gemm.py`
select them.

`before-inventory.json` pins the baseline source. `after-inventory.json` hashes the final production and test
sources from before the audit-only documentation was added. The coverage JSON is gzip-compressed without other
changes. The raw outputs record the results of the tools, and Vulture's nonzero exit comes from its one retained
callback-parameter lead.

Run these commands from the repository root to reproduce the audit. The analysis tools are isolated from the project
dependencies.

```sh
uvx vulture src tests scripts benchmarks/kit --min-confidence 80
uv run ruff check src tests scripts benchmarks/kit --select F401,F841 --output-format json
uv run --with pytest-cov pytest -q --cov=src/ippon --cov-report=json:coverage-after.json
```

No inference benchmark or model verification ran. All 1,035 raw result files hashed before the cutover stay
byte-for-byte unchanged, and only the generated `SUMMARY.md` and `EVIDENCE.json` views were regenerated.

## Evidence and limits

The unused-code check used Vulture 2.16 (`--min-confidence 80`), Ruff 0.17.1 (`--select F401,F841`) and full pytest
coverage with coverage.py 7.17.0, on the M5 Pro with Python 3.12.15 and MLX 0.32.4. The files here include
inventories and hashes, tool output, versions, both complete test logs and the compressed coverage JSON. The scope is
first-party Python in `src`, `tests`, `scripts` and `benchmarks/kit`, plus manual Metal tracing. Historical
reference programs, recorded results, dependencies, caches and build output were excluded. The selected scope has no
Rust.

Ruff reports no unused imports or variables. Before and after, Vulture's sole lead at the 80% threshold is
`daemon._NoRedirect.redirect_request.newurl`. The standard-library redirect callback passes that parameter, so its
intentionally unused value stays. Neither tool establishes dynamic Metal reachability.

The baseline full suite passed **988 tests** with no skips. After the format removal **933 passed**, and after the
removal of the three SIMD-only test cases **930 passed**, with no skips. No benchmark, A/B or model verification ran.
Coverage measures Python launcher statements in the pytest process. It does not measure Metal instructions or
separately launched services, and older physical chips were not tested. The missing lines are validation,
cache-limit and hardware-fallback branches, and they do not prove dead code.

| Launcher module | Before statement coverage | After statement coverage |
|---|---:|---:|
| `_source.py` | 100.0% | 100.0% |
| `attention.py` | 91.2% | 91.2% |
| `conv.py` | 98.1% | 98.1% |
| `delta.py` | 91.7% | 91.7% |
| `gemm.py` | 97.4% | 97.1% |
| `glu.py` | 93.5% | 93.5% |
| `int8.py` | 97.8% | removed |
| `norm.py` | 93.9% | 93.9% |
| `qgemm.py` | 100.0% | 98.2% |
