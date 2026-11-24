# Contributing

Thank you for helping. Every change to ippon comes with the evidence for it, because ippon promises that every
model gives its creator's answers and that every speed claim was measured.

## Setup

Install the dev tools and run the 4 checks with these commands.

```sh
uv sync                                   # dev tools in .venv; requires Python 3.11+ and Rust's cargo
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q
```

The image and video preprocessing runs in a small Rust library, `native/`, which every build of the package
compiles ([rustup.rs](https://rustup.rs) installs cargo). After you change it, run
`uv sync --reinstall-package ippon` to build it again into `src/ippon`. Run `cd native && cargo clippy --release` to
lint it.

Tests run on the CPU by default. Tests marked `metal` or `nax` use the GPU when the Mac has the hardware they need,
and skip otherwise. Do not run the test suite while a benchmark runs. When the repository is public, CI runs the same
4 checks on an Apple-silicon runner. While it is private, run them locally. A pull request must add no skips or
failures compared with the baseline on the same machine.

## Issues and pull requests

- Ask questions in [Discussions](https://github.com/sqwishai/ippon/discussions). Report a bug with the issue
  form. `ippon info --json` and `ippon doctor --json` give the details it asks for. Report a vulnerability
  privately, as [SECURITY.md](SECURITY.md) describes.
- For anything larger than a fix (a new model, a new kernel, a change to the request schema or the server's API),
  open an issue first, so that we agree on the approach before you do the work.
- Keep a pull request to one change. Pull requests are squash-merged, so the title becomes the commit's subject.
  Write the title as a commit subject, in the imperative and under 72 characters.
- AI-assisted contributions are welcome. You are responsible for every line and for the evidence behind it.

## What a change needs

- A bug fix needs a test that fails before the fix.
- A change to what is computed needs `ippon verify MODEL --reference R` on each model it reaches, and each result
  must land within the model's bound ([docs/accuracy.md](docs/accuracy.md)).
- A speedup needs a test that the faster path gives the simple path's outputs. They must be exact where the
  arithmetic is the same, and within a stated bound where it is not. It also needs a fresh-process A/B on a Mac
  with `ippon bench MODEL --ab "A" "B"`, where an arm is extra options and `NAME=value` environment
  (`PYTHONPATH=src` for the working tree against the installed package). Run the A/B with the Mac otherwise idle,
  and record the machine and runtime versions with the measurements. The speedup counts when the verdict is
  `promote` on every model the change reaches, which means 3% faster in a cell and no cell 1% slower
  ([docs/benchmarks.md](docs/benchmarks.md)). If a `mixed` verdict comes from p95 tails alone, run it again with
  more `--rounds`.
- A new model follows [docs/adding-a-model.md](docs/adding-a-model.md). Its accuracy check and benchmark results
  are part of the pull request.
- Anything a caller can see or measure needs a line under `## Unreleased` in [CHANGELOG.md](CHANGELOG.md), with its
  numbers and the machine they came from.

## Conventions

- A model runs as its creator published it, and its weights change only when the caller asks (`--precision q8`).
  Record which checks establish the prompt parity and numerical accuracy of each build.
  [docs/models.md](docs/models.md) marks each build as verified, prompts, past-bound or experimental.
- Keep the library generic. Put what one model needs in its adapter, and describe a model by what its creator
  publishes.
- Validate input once, where it enters (`schema.parse_request`), then trust typed values. Raise an error on wrong
  output instead of returning it.
- Errors are `DecisionError` subclasses, each with a `kind` and an `exit_code` in
  [errors.py](src/ippon/errors.py). [docs/errors.md](docs/errors.md) repeats them, and a test keeps the two equal.
- Keep CLI startup and the check module importable without MLX, so `--help` prints help and `doctor` reports setup
  failures. Model checks still need MLX, and if the runtime is not available they give an error that says what to
  do. Keep MLX imports inside the command functions that need them in `cli.py`, `doctor.py` and `device.py`. Also
  keep them inside the methods of the model sources that read tensors (`model_dir.py`, `gguf/source.py`,
  `ollama.py`), which import MLX only when weights are read. Import `huggingface_hub` inside the functions that ask
  the Hub (`hub.py`, `downloads.py`, `locate.py`), so a command starts without loading it.
- Kernel choices are computed at load (`choices.py`) and never measured on a user's Mac. Raise `choices.RULE`
  whenever they change. If a kernel or MLX changes, refit them with `scripts/measure_kernels.py`
  ([docs/kernels.md](docs/kernels.md)).
- Edit generated text at its source and regenerate it. The models table in docs/models.md comes from `verified.py`
  (`uv run python scripts/models_table.py docs/models.md`). The benchmark tables, the README headline and the charts
  come from the benchmark kit's results. [benchmarks/README.md](benchmarks/README.md) has the complete generation
  commands.
- No code from model repositories runs in the library or in the setup of the tests. Reference scripts that run a
  creator's code live under `references/`, outside the library, and say that they do.

## Code style

- Use syntax that Python 3.11 parses. Type-annotate everything, so `mypy --strict` passes.
- Docstrings say in plain sentences what a function returns or does and why it exists. Comments explain what the
  code cannot say itself. A comment that records a choice gives the measurement behind it, with the chip, the model
  and the numbers.
- Mark tests that need the GPU with `@pytest.mark.metal`.
- Commit messages say what changed and why, in plain prose. The subject line is in the imperative and under 72
  characters, and a body follows when the change needs one.

## Where things are

[docs/architecture.md](docs/architecture.md) explains how a request runs, and [docs/kernels.md](docs/kernels.md)
explains how kernels are chosen. [docs/bring-your-own.md](docs/bring-your-own.md) covers models without an adapter,
[docs/suite.md](docs/suite.md) covers testing on other Macs, and [docs/releasing.md](docs/releasing.md) covers how a
version is released.

## Licence

Contributions are made under the project's Apache-2.0 licence.
