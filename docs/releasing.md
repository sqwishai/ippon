# Releasing

A release is a tag, and [.github/workflows/release.yml](../.github/workflows/release.yml) does the rest. This page is
for maintainers. It tells how a version of ippon is numbered, prepared, tagged and published, and how its release
notes are written.

## Versions

Versions follow [PEP 440](https://peps.python.org/pep-0440/), as `MAJOR.MINOR.PATCH`.

- Before 1.0, the minor number moves for anything a caller has to act on, such as a change to the request schema,
  a command's options or the Python API, a Python version dropped, or a cache that no longer loads. The patch
  number moves for fixes and for additions that change nothing a caller already relies on.
- A pre-release is `rcN` on the version it leads to, so `0.2.0rc1`, then `0.2.0rc2`, then `0.2.0`. pip and uv skip
  pre-releases unless asked (`pip install --pre ippon`, `uv pip install --prerelease allow ippon`, or
  the exact version, `ippon==0.2.0rc1`), and the workflow marks its GitHub release as a pre-release.
- Between releases, `main` keeps the last released version, and changes collect under `## Unreleased` until the
  next one.

Integrations that wrap ippon (such as the Sqwish Labs SDK's Mac backend) rely on 5 stable calls, so every ippon release
must keep them working as they do: `Decider.load(model, precision=..., accept_license=...)`, `Decider.sqwish(body)`,
`Decider.close()`, `Decider.name` and `DecisionError.kind`. If a release has to change one of them, coordinate the
change with downstream callers.

The version is written in 2 places, and both change together. One is `version` in `pyproject.toml`, which is the
package's version and the one that the release workflow checks the tag against. The other is `__version__` in
`src/ippon/__init__.py`, which `ippon --version`, `info` and every bench and suite result report.

## Before tagging

1. Make sure that `main` passes CI on every supported Python, including installing and repairing the built wheel.
2. On an Apple-silicon Mac, from a clean checkout of `main`, run
   `uv sync --locked && uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -q`,
   so the tests that need a real GPU (`metal`, and `nax` on an M5) run too.
3. Make sure that every model whose computation changed since the last release passes
   `ippon verify MODEL --reference R` within its bound ([accuracy.md](accuracy.md)). Make sure that
   `uv run python scripts/models_table.py docs/models.md` leaves no change.
4. Make sure that any speed the README states was measured on the code being released, as installed
   (`benchmarks/kit/summarize.py --docs --readme`, `benchmarks/kit/chart.py`).
5. Make sure that the package builds, its PyPI page has no relative link or image left, and it installs into a clean
   environment. CI's Package job and the release workflow check the first 2 on every build. Run them by hand only
   before the repository is public.

   ```sh
   uv build
   unzip -p dist/ippon-*.whl '*/METADATA' | grep -nE '\]\([^h#]|(src|srcset|href)="[^h#]'   # prints nothing
   uvx twine check --strict dist/*
   uv run python scripts/check_package.py --wheel dist/*.whl --out /tmp/ippon-release-checks --tests --seconds 60
   ```

`check_package.py` creates separate environments for Python 3.11, 3.12, 3.13, 3.14 and 3.15. It installs the wheel with
the dependencies in `uv.lock`, removes and restores MLX and NumPy, damages and repairs the native library, and
checks checkpoint errors. A tiny classifier answers against independent NumPy probabilities. CLI, HTTP and MCP
clients exercise idle unloading, concurrent requests, large-body rejection, forced termination, automatic restart
and clean shutdown. `--tests` also runs the full suite against each installed wheel. `--python 3.15` selects one
version. Each command's exit status gates the check, and JSON reports and logs remain under `--out` on failure.
Use `--install-only` to finish installation and dependency-repair checks without inference or server workloads.
Its reports say `scope: installation` and `serving: false`. You cannot combine it with `--tests`, and it does not
replace the full lifecycle release gate.

GitHub's macOS runners have no usable Apple GPU, so CI and the release workflow run `--install-only`. The serving
lifecycle, its load and the full installed-wheel suite run on a real Mac with the command above before every tag.

For a sustained check of a downloaded model, run this command with its text probability reference.

```sh
uv run python scripts/serving_lifecycle.py /absolute/path/to/model \
  --reference references/d1/d1-2B.json.gz --seconds 600 --clients 8 --out /tmp/ippon-load.json
```

Run it with the GPU otherwise idle. The check keeps 8 clients active, exercises queue backpressure and malformed
requests, and checks every successful answer against the reference's existing bound. It samples health and memory
every 5 seconds and fails if median footprint grows by more than 256 MiB between the first and last quarters.
The report records counts, probability error, latency observations and memory samples. These observations are
reliability evidence, with no benchmark speed claim. A finite run cannot establish the absence of a slow memory leak.
The check uses an isolated cache and offline model loading, then stops its servers and preserves their logs.

PyPI shows README.md without the repository around it, so the build rewrites its relative links
(`[tool.hatch.metadata.hooks.fancy-pypi-readme]` in `pyproject.toml`). Images point to
`raw.githubusercontent.com/sqwishai/ippon/main/...`, folders to `github.com/sqwishai/ippon/tree/main/...`
and files to `.../blob/main/...`. They point at `main`, so a file the README links to keeps its path, or every
published page that links to it breaks.

## The changelog

[CHANGELOG.md](../CHANGELOG.md) collects a line under `## Unreleased` for each change a caller can see or measure,
as it lands ([CONTRIBUTING.md](../CONTRIBUTING.md)). At a release, update it as follows.

1. `## Unreleased` becomes the version's section, with its date, as in `## 0.2.0 (3 November 2026)`
   (`scripts/release.py 0.2.0` does it). A pre-release (`0.2.0rc1`) leaves `## Unreleased` as it is, because its
   notes are the coming version's.
2. Lines a caller has to act on go first, each saying what to do (a renamed option, a dropped command).
3. Each line keeps its numbers and the machine they came from, and nothing is rounded into an adjective.
4. A new, empty `## Unreleased` goes above it once the release is out.

## Tagging

The release is a pull request like any other change, so CI runs on exactly what is tagged.

1. Run `uv run python scripts/release.py 0.2.0`. It sets the version in `pyproject.toml` and
   `src/ippon/__init__.py`, refreshes `uv.lock` and dates `## Unreleased` as `## 0.2.0 (date)`. Open a pull request
   titled `Release 0.2.0` with those changes and merge it once CI passes.
2. Tag the merge commit on `main` and push the tag.

   ```sh
   git switch main && git pull --ff-only
   git tag -a v0.2.0 -m "ippon 0.2.0"
   git push origin v0.2.0
   ```

3. When the workflow has finished, read over the draft release it opened (its text is the version's changelog
   section, from `scripts/release.py --notes`) and publish it.
4. Add a new, empty `## Unreleased` at the top of `CHANGELOG.md`.

A tag is `v` and the version exactly, for example `v0.2.0` or `v0.2.0rc1`.

## What the release workflow does

A pushed tag that starts with `v` runs 3 jobs, only in the public repository.

| Job | Runs on | Does |
|---|---|---|
| Build | macOS 15 (Apple silicon) | stops unless the tag is `v` + `pyproject.toml`'s version and `CHANGELOG.md` has the version's notes. Marks the release a pre-release when the version has a letter (`rc`, `a`, `b`, `.dev`). Runs `uv sync --locked`, format, lint, types and tests, `uv build`, `twine check --strict` and no relative links in the PyPI description. Writes `SHA256SUMS` and keeps `dist/` and the notes |
| Publish to PyPI | Ubuntu, environment `pypi` | uploads the sdist and wheel through trusted publishing, with GitHub's OIDC token and no stored PyPI token |
| Draft the GitHub release | Ubuntu | opens a draft release for the tag, titled `ippon v0.2.0`, with the version's changelog section as its text and the sdist, wheel and `SHA256SUMS` attached |

Nothing is published unless the build and its tests pass. If the build fails, fix `main`, then move the tag
with `git tag -d v0.2.0 && git push origin :refs/tags/v0.2.0` and tag again. Once PyPI has a version, it never takes
the same file name again, even after a deletion, so a broken release is yanked on PyPI and followed by the next
patch version.

## One-time setup

Set up the PyPI trusted publisher before the first release. The project does not exist on PyPI yet, so add it as a
pending publisher. On pypi.org, go to **Your account > Publishing > Add a new pending publisher > GitHub** and enter
these values.

| Field | Value |
|---|---|
| PyPI project name | `ippon` |
| Owner | `sqwishai` |
| Repository name | `ippon` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The first successful publish creates the project with this publisher. Later changes are under the project's
**Settings > Publishing**. Add at least one other maintainer as an owner of the PyPI project.

Set up the GitHub environment. In the repository's **Settings > Environments**, create `pypi`. Give it required
reviewers, so a maintainer approves each upload, and limit its deployments to tags matching `v*`.

Once the repository is public, set these repository settings.

- Branch protection on `main` requires a pull request and the CI checks to pass, with no force-pushes. The checks
  are `Format, lint, types and tests`, `Tests (Python 3.11)`, `Tests (Python 3.13)`, `Tests (Python 3.14)`,
  `Tests (Python 3.15)` and `Package`.
- A tag ruleset lets only maintainers create, move or delete tags matching `v*`, since a pushed tag publishes.
- Private vulnerability reporting is on, as [SECURITY.md](../SECURITY.md) promises. Discussions are on, with Q&A
  and Ideas categories, where the issue forms and docs/suite.md send questions and suite results.

The workflow's jobs run only once the repository is public (macOS runners are free there). The README's images on PyPI load from `raw.githubusercontent.com`, which serves a public repository only.

## Release notes

The notes are the version's changelog section, written for someone who decides whether to upgrade. They use the
changelog's voice, in plain sentences with every claim next to its measurement. The notes have these parts, in order.

1. An opening of 1 or 2 sentences on what the release does for a caller.
2. Before you upgrade, which says what a caller has to do (a dropped Python, a changed option), or nothing.
3. Measured changes as tables, each under a line that names the machine and versions. That line gives chip and
   memory, macOS, MLX, the 2 ippon versions compared, and how it was timed (for example, a fresh-process `bench --ab` with its
   rounds and requests). A speed table gives the model and build, the state size and question count, both medians
   and the percentage speed increase. An accuracy table gives the model, build and precision, the largest difference from the creator's
   fp32, the bound and the top answers that changed. Figures from another machine or a model card never go in these
   tables.
4. Everything else from the changelog section, as it is written there.
5. Install, which gives `pip install ippon==0.2.0` (with `--pre` noted for a pre-release), and credit for
   contributors and for anyone who reported a vulnerability fixed in it ([SECURITY.md](../SECURITY.md)).
