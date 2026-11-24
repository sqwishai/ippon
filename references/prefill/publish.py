"""Curate the archived experiment; file conversion only, never model execution."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path
from typing import Any

from audit import PREFIX_SOURCE, digest, numerical, portable, read, sha

LABELS = {"d1": "d1-3B", "d3": "d3-lite", "clef": "Clef-Flash MLX 8-bit"}


def q95(values: list[float]) -> float:
    ordered = sorted(values)
    at = 0.95 * (len(ordered) - 1)
    low = int(at)
    return ordered[low] if low + 1 == len(ordered) else ordered[low] + (ordered[low + 1] - ordered[low]) * (at - low)


def rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for original in result["cells"]:

        def matches(cell: dict[str, Any], original: dict[str, Any] = original) -> bool:
            return all(cell[k] == original[k] for k in ("scenario", "words", "questions")) and (
                original["scenario"] != "concurrent" or cell["clients"] == original["clients"]
            )

        selected = [[c for rep in result[a]["reports"] for c in rep["cells"] if matches(c)] for a in ("a", "b")]
        samples = [[n for c in side for n in c["ms_raw"]] for side in selected]
        medians = [statistics.median(side) for side in samples]
        ratio = medians[1] / medians[0]
        padding = [i["ordinary_padding"]["padding_fraction"] for c in selected[0] for i in c.get("iterations", [])]
        output.append(
            {
                "scenario": original["scenario"],
                "words": original["words"],
                "questions": original["questions"],
                "clients": original.get("clients", selected[0][0].get("clients", 1)),
                "samples_per_arm": [len(s) for s in samples],
                "a_median_ms": medians[0],
                "b_median_ms": medians[1],
                "raw_median_ratio": ratio,
                "median_time_change_percent": 100 * (ratio - 1),
                "p95_time_change_percent": 100 * (q95(samples[1]) / q95(samples[0]) - 1),
                "observed_median_over_one_percent": ratio > 1.01,
                "library": original,
                "actual_padding_min": min(padding) if padding and all(p is not None for p in padding) else None,
                "actual_padding_max": max(padding) if padding and all(p is not None for p in padding) else None,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.absolute()
    base = Path(__file__).absolute().parent
    datasets = []
    mapping = {}

    def dataset(relative: str, target: str) -> Any:
        original = source / relative
        raw = read(original)
        public = portable(raw, source)
        assert numerical(raw) == numerical(public)
        encoded = (json.dumps(public, separators=(",", ":"), allow_nan=False) + "\n").encode()
        output = base / target
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(gzip.compress(encoded, mtime=0) if target.endswith(".gz") else encoded)
        mapping[relative] = target
        datasets.append(
            {
                "source": relative,
                "source_sha256": sha(original.read_bytes()),
                "published": target,
                "published_sha256": sha(output.read_bytes()),
                "numeric_preservation": digest(public),
            }
        )
        return public

    results = {}
    for path in sorted((source / "runs").glob("*/result.json")):
        name = path.parent.name
        results[name] = dataset(str(path.relative_to(source)), f"data/runs/{name}.json.gz")
    for model in LABELS:
        for kind in ("fresh", "growing"):
            name = f"auto-{model}-{kind}-final"
            dataset(f"runs/{name}/manifest.json", f"data/manifests/{name}.json.gz")
    threshold = dataset("threshold-evidence.json", "data/threshold-evidence.json.gz")
    correctness = dataset("validation/auto-policy/correctness-summary.json", "data/correctness-summary.json.gz")
    for name in (
        "qualification",
        "d1-3B-maker",
        "d1-3B-invariance",
        "d3-lite-maker",
        "d3-lite-invariance",
        "clef-maker",
        "clef-invariance",
        "d1-capture-grown",
        "d3-capture-grown",
        "clef-capture-grown",
        "d1-withheld",
        "d3-withheld",
        "clef-withheld",
        "clef-capture-reference-requests",
        "clef-capture-fp32",
    ):
        dataset(f"validation/auto-policy/{name}.json", f"data/correctness/{name}.json.gz")
    dataset("archive-final/archive-manifest.json", "data/archive-manifest.json.gz")
    dataset("seed-manifest.json", "data/seed-manifest.json.gz")
    dataset("idle-control-1/summary.json", "data/idle-control.json.gz")

    attempts = []
    for run in sorted((source / "runs").iterdir()):
        if not run.is_dir():
            continue
        for folder in sorted(run.glob("round-*")):
            acceptance = folder / "acceptance.json"
            item = read(acceptance) if acceptance.exists() else None
            monitor = item.get("monitoring", {}) if item else {}
            activity = [a["cpu_percent"] for interval in monitor.get("intervals", []) for a in interval["activity"]]
            sources = []
            for raw in sorted(folder.glob("*.json")):
                if raw.name == "acceptance.json":
                    continue
                sources.append(
                    {
                        "source": str(raw.relative_to(source)),
                        "sha256": sha(raw.read_bytes()),
                        "bytes": raw.stat().st_size,
                    }
                )
            private = folder / "cpu-monitor/processes.private.jsonl"
            attempts.append(
                {
                    "run": run.name,
                    "attempt": folder.name,
                    "status": "interrupted" if item is None else "accepted" if item["accepted"] else "excluded",
                    "contaminated": monitor.get("contaminated"),
                    "monitoring_unknown": monitor.get("monitoring_unknown"),
                    "peak_competing_cpu_percent": max(activity, default=0),
                    "failure": portable(item.get("failure"), source) if item else "interrupted before a complete pair",
                    "acceptance_sha256": sha(acceptance.read_bytes()) if item else None,
                    "raw_local_sources": sources,
                    "private_activity_sha256": sha(private.read_bytes()) if private.exists() else None,
                }
            )
    (base / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")

    final_rows = []
    gates = {}
    for model, label in LABELS.items():
        gates[label] = {}
        for kind in ("fresh", "growing"):
            name = f"auto-{model}-{kind}-final"
            result = results[name]
            assert result["complete"]
            assert result["coverage"]["complete"]
            for arm, mode in (("a", "off"), ("b", "auto")):
                for report in result[arm]["reports"]:
                    for record in [report, *report["cells"]]:
                        assert all(
                            record[field] == mode
                            for field in ("effective_packed", "effective_packed_before", "effective_packed_after")
                        )
            cells = rows(result)
            final_rows += [
                {"model": label, "kind": kind, "dataset": mapping[f"runs/{name}/result.json"], **c} for c in cells
            ]
            gates[label][kind] = {
                "library_verdict": result["verdict"],
                "conservative_gate": result["release_gate"]["status"],
                "cells": len(cells),
                "accepted_units": sum(a["accepted"] for a in result["attempts"]),
                "excluded_attempts": sum(not a["accepted"] for a in result["attempts"]),
                "worst_observed_median_time_change_percent": max(c["median_time_change_percent"] for c in cells),
            }
    assert len(final_rows) == 60
    summary = {
        "decision": "archive; not shipped",
        "overall_gate": "reject",
        "machine": {"chip": "Apple M5 Pro", "memory_gib": 48, "macos": "27.0.1", "mlx": "0.32.4"},
        "final_cells": 60,
        "gates": gates,
        "final_rows": final_rows,
        "final_accepted_units": sum(g[k]["accepted_units"] for g in gates.values() for k in g),
        "final_excluded_attempts": sum(g[k]["excluded_attempts"] for g in gates.values() for k in g),
        "archive_attempts": dict(collections.Counter(a["status"] for a in attempts)),
        "candidate_source_sha256": correctness["source_sha256"],
        "baseline_reference": "8f09ee41b461f5cbdbac01594b24f1c7578b040e",
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Complete final gate",
        "",
        (
            "**Rejected and archived; the candidate was not shipped.** All 60 cells are shown. A is "
            "the unchanged baseline; B is the experimental automatic policy. Recorded effective modes "
            "are A `off` and B `auto` before/after every accepted report and cell. Tracing was "
            "disabled."
        ),
        "",
        (
            "Apple M5 Pro, 48 GiB; macOS 27.0.1; MLX 0.32.4. Each cell has four accepted paired "
            "rounds. Percentages are time changes: `100 x (B/A - 1)`; negative means B is faster. "
            "Median and p95 changes use raw samples. Confidence intervals and cell verdicts retain the "
            "original library calculations."
        ),
        "",
        (
            "The 36 fresh cells cover 1/4/8 clients x 1/3 questions x 100/1,000 words. The 24 growing "
            "rows are four views of each 4/8-client session: all requests, first turn, steady turns "
            "and session wall time. These views reuse the same observations and are not independent "
            "workloads."
        ),
        "",
    ]
    for model, label in LABELS.items():
        for kind in ("fresh", "growing"):
            name = f"auto-{model}-{kind}-final"
            result = results[name]
            lines += [
                f"## {label}: {kind}",
                "",
                (
                    f"Library gate: **{result['verdict']}**; conservative raw-median gate: "
                    f"**{result['release_gate']['status']}**. [Complete samples and "
                    f"reports]({mapping[f'runs/{name}/result.json']})."
                ),
                "",
                (
                    "| cell | words | q | A median ms | B median ms | median time change | median 95% CI | p95 "
                    "time change | library | median >1% slower |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
            for c in rows(result):
                title = f"{c['clients']} clients" if kind == "fresh" else c["scenario"]
                ci = c["library"]["ratio_ci"]
                lines.append(
                    f"| {title} | {c['words']:,} | {c['questions']} | {c['a_median_ms']:.3f} | "
                    f"{c['b_median_ms']:.3f} | {c['median_time_change_percent']:+.3f}% | "
                    f"{100 * (ci[0] - 1):+.2f}% to {100 * (ci[1] - 1):+.2f}% | "
                    f"{c['p95_time_change_percent']:+.3f}% | {c['library']['verdict']} | "
                    f"{'yes' if c['observed_median_over_one_percent'] else 'no'} |"
                )
            lines += [""]
    (base / "FINAL.md").write_text("\n".join(lines) + "\n")

    lines = [
        "# Threshold sweep",
        "",
        (
            "Diagnostic forced packing versus padded capture: 96 complete cells, two accepted pairs "
            "and eight timed group calls per arm/cell. This earlier candidate is identified by its own "
            "source hashes; it is not the final automatic-policy gate."
        ),
        "",
        (
            "All percentages are time changes; negative means forced packing is faster. Actual padding "
            "ranges cover all accepted iterations after ordinary sorted/budgeted grouping. A dash "
            "means no ordinary padded group exists."
        ),
        "",
        (
            "A conservative 20% actual-padding cut selected 39 cells (10 d1, 17 d3, 12 Clef), all "
            "faster in this diagnostic comparison. Later correctness replays confirmed the intended "
            "routes around this cut; the final fresh gate still failed. d1's 16 three-question branch cells have "
            "undefined padded-group waste and cannot be selected from this scalar alone."
        ),
        "",
        (
            "The grouping pitfall is visible in d3 B8/q3/1,000 words: nominal 50%/75% mixtures split "
            "into two ordinary budget groups, leaving only about 0.2 to 0.6% padding and causing "
            "+6.08%/+7.40% time increases. A global longest-prompt calculation would select the wrong "
            "path."
        ),
        "",
        (
            "The original d1 stage's three contaminated pairs, and bulk Clef's three contaminated "
            "pairs plus interrupted fourth, remain in [the exclusion ledger](attempts.json). They "
            "contribute no clean samples. Each Clef cell's initial exclusion also remains recorded. "
            "The early indexing hypothesis was an inference from generic activity categories, not a "
            "reason to waive contamination."
        ),
        "",
    ]
    for label in LABELS.values():
        lines += [
            f"## {label}",
            "",
            ("| B | q | words | target padding | actual padding | A ms | B ms | time change | p95 change | library |"),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for c in [r for r in threshold["rows"] if r["model"] == label]:
            pad = (
                "—"
                if c["actual_padding_min"] is None
                else f"{100 * c['actual_padding_min']:.2f}% to {100 * c['actual_padding_max']:.2f}%"
            )
            link = mapping[c["source"]["path"]]
            lines.append(
                f"| [{c['clients']}]({link}) | {c['questions']} | {c['words']:,} | "
                f"{100 * c['requested_padding_fraction']:.0f}% | {pad} | {c['a_median_ms']:.3f} | "
                f"{c['b_median_ms']:.3f} | {c['median_time_change_percent']:+.3f}% | "
                f"{c['p95_time_change_percent']:+.3f}% | {c['library_comparison_unchanged']['verdict']} |"
            )
        lines += [""]
    (base / "THRESHOLDS.md").write_text("\n".join(lines) + "\n")

    lines = [
        "# Controls and withheld boundary timings",
        "",
        (
            "These are supporting diagnostics. They do not replace the failed complete fresh gate. "
            "Percentages are time changes; negative means B is faster."
        ),
        "",
        "## Identical-arm control",
        "",
        (
            "The cumulative d1 A/A control retained eight clean pairs. It still recorded +1.664% at "
            "B4/q1/1,000 words and +1.332% at B1/q3/1,000 words; the library verdict was reject. These "
            "clean shifts were retained. They were not used to relax the final 1% bound."
        ),
        "",
        "| clients | q | words | A ms | B ms | time change | p95 change | library |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    control = "control-d1-fresh-aa-more"
    for c in rows(results[control]):
        lines.append(
            f"| {c['clients']} | {c['questions']} | {c['words']:,} | {c['a_median_ms']:.3f} | "
            f"{c['b_median_ms']:.3f} | {c['median_time_change_percent']:+.3f}% | "
            f"{c['p95_time_change_percent']:+.3f}% | {c['library']['verdict']} |"
        )
    lines += [
        "",
        (
            f"[Complete cumulative A/A samples]({mapping[f'runs/{control}/result.json']}). The separate "
            f"idle sampler control recorded 29 valid intervals and 3.97% of one CPU core in the "
            f"observer; instrumentation cost is not assumed to be zero."
        ),
        "",
        "## Untraced withheld cases",
        "",
        (
            "All 14 withheld cases ran with tracing disabled: capture (A) versus auto (B), B3/B7 near "
            "18%/22% actual padding, plus B3/q3 for noncausal d3. The library aggregate verdict was "
            "promote for each build, but the below-boundary d3 and Clef B7 cells still had raw median "
            "increases above 1%. These diagnostic gains did not qualify the final gate."
        ),
        "",
    ]
    for model, label in LABELS.items():
        name = f"auto-{model}-withheld-timed"
        lines += [
            f"### {label}",
            "",
            f"[All samples]({mapping[f'runs/{name}/result.json']}).",
            "",
            "| case | actual padding | A ms | B ms | time change | p95 change | library |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for c in rows(results[name]):
            lines.append(
                f"| {c['scenario']} | {100 * c['actual_padding_min']:.2f}% to "
                f"{100 * c['actual_padding_max']:.2f}% | {c['a_median_ms']:.3f} | {c['b_median_ms']:.3f} | "
                f"{c['median_time_change_percent']:+.3f}% | {c['p95_time_change_percent']:+.3f}% | "
                f"{c['library']['verdict']} |"
            )
        lines += [""]
    (base / "CONTROLS.md").write_text("\n".join(lines) + "\n")

    manifest = {
        "schema": 1,
        "decision": "archive; not shipped",
        "datasets": datasets,
        "numeric_preservation": (
            "All measurement numbers and booleans retain identical JSON paths "
            "and values; only nonmeasurement environment metadata is omitted. Strings may be made "
            "portable or anonymous."
        ),
        "normalization": {
            "prefix_source": PREFIX_SOURCE,
            "prefix_source_sha256": sha((source / PREFIX_SOURCE).read_bytes()),
            "relocation_safe": True,
        },
        "text_files": [
            {"path": p.name, "sha256": sha(p.read_bytes())}
            for p in sorted(base.iterdir())
            if p.is_file() and (p.suffix in (".md", ".py") or p.name in ("summary.json", "attempts.json", "AUDIT.json"))
        ],
    }
    (base / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "datasets": len(datasets),
                "final_cells": len(final_rows),
                "final_accepted_units": summary["final_accepted_units"],
                "final_excluded_attempts": summary["final_excluded_attempts"],
                "all_attempts": summary["archive_attempts"],
            }
        )
    )


if __name__ == "__main__":
    main()
