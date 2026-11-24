- Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, what each kind of change needs, and the code style
- ALWAYS finish with `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q`, and gate on
  pytest's own exit status, never on a pipe (`pytest | tail` exits 0 when tests fail). After a change under `native/`,
  also run `cd native && cargo fmt --check && cargo clippy --release --locked -- -D warnings`, as CI does
- NEVER assume a lint error, type error or test failure is pre-existing, because `main` is clean
- PREFER running the tests a change reaches (`uv run pytest tests/test_d1.py -q`) while working, and the whole suite
  once before finishing
- ALWAYS check whether an existing test already covers a behavior before adding one, and extend that test instead
- PREFER few, strong tests that would catch a real regression, such as a tiny checkpoint answered end to end against
  an independent numpy of the creator's math, a creator's reference file, or a parser fed the input that broke it
- DO NOT add tests of constants, shapes, private layout or mocks of ippon's own code
- ALWAYS keep a model's answers within its bound in [docs/accuracy.md](docs/accuracy.md). A change to what is
  computed runs `ippon verify MODEL --reference R` on every model it reaches
- NEVER trade accuracy for speed, and NEVER change a model's weights or precision unless the caller asks
  (`--precision q8`). ippon runs models as their creators published them, at full precision or 8-bit
- DO NOT take a speedup under 3% that adds code. A tie at the hardware's limit is fine
- ALWAYS time on an otherwise idle Mac, one GPU job at a time, in fresh processes (`ippon bench MODEL --ab "A" "B"`).
  NEVER run the test suite or a build alongside a benchmark
- NEVER compare against a rival runtime at a version other than the one the benchmark kit pins
- ALWAYS keep what one model needs in its adapter, so the library stays generic
- ALWAYS raise a `DecisionError` subclass with its `kind` and `exit_code` in `errors.py`, and keep
  [docs/errors.md](docs/errors.md) equal to it
- ALWAYS write GPU kernels in `.metal` files under `src/ippon/kernels/metal/`, read with `metal_source`. NEVER write
  them as source strings in `.py` files
- ALWAYS raise `choices.RULE` when a kernel choice changes, and refit it with `scripts/measure_kernels.py` when a
  kernel or MLX changes
- PREFER top-level imports. The exceptions are MLX inside the CLI's command functions (`cli.py`, `doctor.py`,
  `device.py`) and the model sources' tensor reads (`model_dir.py`, `gguf/source.py`, `ollama.py`), so `--help`,
  `doctor` and `check` work without it. `huggingface_hub` is also imported inside the functions that ask the Hub
  (`hub.py`, `downloads.py`, `locate.py`), so commands start without it
- PREFER `Literal` types and exhaustive branches over open strings, so a new kind needs explicit handling
- AVOID shortened names. Write `question`, not `q`, in anything longer than a comprehension, and `probabilities`,
  not `probs`
- ALWAYS edit generated text at its source and regenerate it. The models table comes from `verified.py`, and the
  benchmark tables, README headline and charts come from the benchmark kit
  ([benchmarks/README.md](benchmarks/README.md))
- ALWAYS call the people who publish a model its creators in prose. Data keys, file names and engine names that
  say `maker` keep their spelling so recorded results still read
- ALWAYS write user-facing docs in short, active sentences, with commas or a dash where a semicolon would go
- PREFER comments that explain the current behavior and the reason for it. A comment that records a choice carries
  its measurement (chip, model, numbers)
- DO NOT leak a conversation, prompt or iteration history into code comments, commit messages or other prose.
  Write for a reader who has not seen it
- DO NOT describe the internals of Sqwish's own models beyond what their model cards publish
- NEVER update every dependency at once. Change one with `uv lock --upgrade-package NAME`
- ALWAYS write commit messages in plain prose, with an imperative subject under 72 characters, then why, and with
  no tool or co-author trailers
- PREFER Python 3.11 syntax. 3.11 and every newer version are supported
