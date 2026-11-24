## Summary

<!-- Say what this changes and why. Link the issue it closes. -->

## Evidence

<!--
Give the evidence that CONTRIBUTING.md asks for this kind of change.
- A bug fix needs the test that failed before it.
- A change to what is computed needs `ippon verify` on each model it reaches, and the distance it reports.
- A speedup needs `ippon bench MODEL --ab ...` on each model it reaches, with the verdict, the Mac (chip, memory,
  macOS), the runtime versions and clean timing evidence.
-->

## Checklist

- [ ] `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q` pass.
- [ ] CHANGELOG.md has a line under Unreleased, if a caller can see or measure the change.
- [ ] I understand every line of this change and can explain it in review.
