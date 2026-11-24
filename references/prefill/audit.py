"""Verify archived evidence without loading a model or running a benchmark."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
from functools import cache
from pathlib import Path
from typing import Any

OMIT = {"power"}
PREFIX_SOURCE = "runs/auto-d1-fresh-final/manifest.private.json"


@cache
def recorded_prefixes(source: Path) -> tuple[tuple[str, str], ...]:
    """Read immutable recorded roots, so moving the retained archive does not change normalization."""
    metadata = read(source / PREFIX_SOURCE)
    suffix = "/build/experiments/prefill-2026-10-25/revival"
    roots = {value.split(suffix, 1)[0] for value in metadata["sources"] if suffix in value}
    if not roots:
        raise ValueError("recorded source metadata has no experiment root")
    pairs = []
    for root in sorted(roots):
        for prefix in (root, re.sub(r"^/(?:Users|home)/[^/]+", "~", root)):
            pairs.extend(((prefix + suffix, "<experiment>"), (prefix, "<repo>")))
    return tuple(pairs)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> Any:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def portable(value: Any, source: Path | None = None, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: portable(v, source, k) for k, v in value.items() if k not in OMIT}
    if isinstance(value, list):
        return [portable(v, source, key) for v in value]
    if not isinstance(value, str):
        return value
    if source is not None:
        for prefix, replacement in recorded_prefixes(source):
            value = re.sub(re.escape(prefix) + r"(?=/|$|[\s\"'`<>),;:\]])", replacement, value)
    value = value.replace("<repo>/build/experiments/prefill-2026-10-25/revival", "<experiment>")
    value = re.sub(r"(?:~|/(?:Users|home)/[^/\s\"'`<>]+)/models/", "<models>/", value)
    value = re.sub(
        r"(?:~|/(?:Users|home)/[^/\s\"'`<>]+)/[^\s\"'`<>),;:\]]+", lambda m: "<local>/" + m[0].rsplit("/", 1)[-1], value
    )
    value = re.sub(
        r"/(?:private/)?(?:var/folders|tmp|var/tmp)/[^\s\"'`<>),;:\]]+",
        lambda m: "<tmp>/" + m[0].rsplit("/", 1)[-1],
        value,
    )
    if key in ("busy", "activity") and (match := re.fullmatch(r".+ \((\d+)%\)", value)):
        value = f"process ({match[1]}%)"
    return value


def numerical(value: Any, pointer: str = "") -> list[tuple[str, str, Any]]:
    if isinstance(value, dict):
        return [leaf for k in sorted(value) if k not in OMIT for leaf in numerical(value[k], pointer + "/" + k)]
    if isinstance(value, list):
        return [leaf for i, item in enumerate(value) for leaf in numerical(item, pointer + f"/{i}")]
    if isinstance(value, (int, float, bool)):
        return [(pointer, type(value).__name__, value)]
    return []


def digest(value: Any) -> dict[str, Any]:
    leaves = numerical(value)
    return {
        "numeric_and_boolean_leaves": len(leaves),
        "numeric_sha256": sha(json.dumps(leaves, separators=(",", ":"), allow_nan=False).encode()),
    }


def check_final_rows(base: Path, summary: dict[str, Any]) -> None:
    """Recompute displayed final values directly from the preserved raw samples."""
    datasets = {row["dataset"] for row in summary["final_rows"]}
    assert len(datasets) == 6
    accepted = excluded = 0
    for name in sorted(datasets):
        data = read(base / name)
        published = [row for row in summary["final_rows"] if row["dataset"] == name]
        assert data["coverage"]["complete"]
        assert len(published) == len(data["cells"])
        accepted += sum(attempt["accepted"] for attempt in data["attempts"])
        excluded += sum(not attempt["accepted"] for attempt in data["attempts"])
        for arm, mode in (("a", "off"), ("b", "auto")):
            for report in data[arm]["reports"]:
                for record in [report, *report["cells"]]:
                    for field in ("effective_packed", "effective_packed_before", "effective_packed_after"):
                        assert record[field] == mode
        for row, original in zip(published, data["cells"], strict=True):
            assert row["library"] == original
            samples = []
            for arm in ("a", "b"):
                values = []
                for report in data[arm]["reports"]:
                    for cell in report["cells"]:
                        if any(cell[key] != original[key] for key in ("scenario", "words", "questions")):
                            continue
                        if original["scenario"] == "concurrent" and cell["clients"] != original["clients"]:
                            continue
                        values.extend(cell["ms_raw"])
                samples.append(values)
            a, b = (statistics.median(values) for values in samples)
            assert row["a_median_ms"] == a
            assert row["b_median_ms"] == b
            assert row["samples_per_arm"] == [len(values) for values in samples]
            assert row["raw_median_ratio"] == b / a
            assert row["median_time_change_percent"] == 100 * (b / a - 1)
            assert row["observed_median_over_one_percent"] == (b / a > 1.01)
            tails = []
            for values in samples:
                ordered = sorted(values)
                index = 0.95 * (len(values) - 1)
                lower = int(index)
                tails.append(
                    ordered[lower] + (ordered[min(lower + 1, len(values) - 1)] - ordered[lower]) * (index - lower)
                )
            assert row["p95_time_change_percent"] == 100 * (tails[1] / tails[0] - 1)
    assert accepted == summary["final_accepted_units"] == 156
    assert excluded == summary["final_excluded_attempts"] == 9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="optional retained local experiment archive for direct source comparison"
    )
    args = parser.parse_args()
    base = Path(__file__).absolute().parent
    manifest = read(base / "MANIFEST.json")
    leaves = 0
    for item in manifest["datasets"]:
        path = base / item["published"]
        assert sha(path.read_bytes()) == item["published_sha256"], item["published"]
        data = read(path)
        assert digest(data) == item["numeric_preservation"], item["published"]
        text = json.dumps(data)
        assert not re.search(r"/(?:Users|home)/|~/|/(?:private/)?var/folders/", text), item["published"]
        if args.source:
            original = args.source / item["source"]
            assert sha(original.read_bytes()) == item["source_sha256"], item["source"]
            raw = read(original)
            assert numerical(raw) == numerical(data), item["published"]
            assert portable(raw, args.source) == data, item["published"]
        leaves += item["numeric_preservation"]["numeric_and_boolean_leaves"]
    for item in manifest["text_files"]:
        assert sha((base / item["path"]).read_bytes()) == item["sha256"], item["path"]
    summary = read(base / "summary.json")
    check_final_rows(base, summary)
    assert summary["final_cells"] == 60
    assert all(summary["gates"][model]["fresh"]["library_verdict"] == "reject" for model in summary["gates"])
    assert all(summary["gates"][model]["growing"]["library_verdict"] == "promote" for model in summary["gates"])
    if args.source:
        assert sha((args.source / PREFIX_SOURCE).read_bytes()) == manifest["normalization"]["prefix_source_sha256"]
        for attempt in read(base / "attempts.json"):
            folder = args.source / "runs" / attempt["run"] / attempt["attempt"]
            if attempt["acceptance_sha256"] is not None:
                assert sha((folder / "acceptance.json").read_bytes()) == attempt["acceptance_sha256"]
            for item in attempt["raw_local_sources"]:
                assert sha((args.source / item["source"]).read_bytes()) == item["sha256"]
            if attempt["private_activity_sha256"]:
                assert (
                    sha((folder / "cpu-monitor/processes.private.jsonl").read_bytes())
                    == attempt["private_activity_sha256"]
                )
    print(
        json.dumps(
            {
                "passed": True,
                "datasets": len(manifest["datasets"]),
                "numeric_and_boolean_leaves": leaves,
                "direct_source_comparison": args.source is not None,
                "final_cells": 60,
                "overall_gate": "reject",
            }
        )
    )


if __name__ == "__main__":
    main()
