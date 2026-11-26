"""Draw the README's per-model charts from summarize.py's selected paired measurements: one SVG per model and
theme, a block of horizontal bars per workload, ippon above each rival on the same build.

    python3 -I benchmarks/kit/chart.py --readme-round --recorded
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize as s

THEMES = {  # one blue and one orange per surface; neutral text
    "light": {"short": "#2a78d6", "long": "#eb6834", "text": "#1f2328", "muted": "#59636e"},
    "dark": {"short": "#3987e5", "long": "#d95926", "text": "#e6edf3", "muted": "#9198a1"},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


CHART_WIDTH, MARGIN = 830, 16
LABEL_W = 214  # the workload's name and what it sends, left of its rows
NAME_RIGHT, BAR_X, BAR_MAX = 350, 360, 190  # runtime names end at NAME_RIGHT, bars start at BAR_X
VERDICT_RIGHT = CHART_WIDTH - MARGIN  # how ippon compares, right-aligned in one column
ROW_H, BAR_H, CAPTION_H, PANEL_GAP, LINE = 22, 14, 18, 22, 14
SMALL = 11


def text_width(text: str, size: float) -> float:
    """An estimate of a line's width in the system sans font: rendered by WebKit (qlmanage), the charts' widest
    labels at 11 px, such as "ippon 57% faster" in semibold, take about 0.50 of the font size per character."""
    return len(text) * size * 0.52


def wrap(text: str, width: float, size: float = SMALL) -> list[str]:
    """Lines of at most `width`, broken at spaces; a no-break space keeps its words on one line."""
    lines: list[str] = []
    for word in text.split(" "):
        if lines and text_width(f"{lines[-1]} {word}", size) <= width:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return lines


def amount(value: float, unit: str) -> str:
    return f"{s.rate(value)} req/s" if unit == "req/s" else f"{value:,.0f} ms" if value >= 10 else f"{value:.1f} ms"


def verdict(rival: dict, unit: str) -> str:
    """How ippon compares with one rival, said the way the README's first chart says it."""
    rate = unit == "req/s"
    return ("ippon: " if rate else "ippon ") + s.compared(rival["ratio"], rate)


def model_svg(model: dict, theme: dict[str, str]) -> str:
    """One model's chart: a block of horizontal bars per workload, each with its own scale, ippon's bar above every
    rival's on the same build, the value after every bar and how ippon compares in one column on the right."""
    parts: list[str] = []

    def text(
        tx: float,
        ty: float,
        value: str,
        size: float = SMALL,
        weight: int = 400,
        muted: bool = False,
        anchor: str = "start",
    ) -> None:
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="{size:g}" font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{theme["muted" if muted else "text"]}">{escape(value)}</text>'
        )

    subtitle = (
        "Timed on ippon alone"
        if model["alone"]
        else "ippon in orange, the other runtime in blue, both on the same build"
    )
    text(MARGIN, 20, subtitle, size=12, muted=True)  # the README's fold line above the chart names the model
    y = 40.0
    for panel in model["panels"]:
        unit = panel["unit"]
        how = "requests per second, higher is better" if unit == "req/s" else "ms, shorter is better"
        header = [(panel["title"], 13, 600, False)] + [
            (line, SMALL, 400, True) for t in (panel["detail"], how) for line in wrap(t, LABEL_W - 10)
        ]
        for i, (line, size, weight, muted) in enumerate(header):
            text(MARGIN, y + 11 + i * LINE, line, size=size, weight=weight, muted=muted)
        top = max(max([c["ippon"], *(r["value"] for r in c["rivals"])]) for c in panel["clusters"])
        captioned = len(panel["clusters"]) > 1  # several builds in one workload: name each cluster's build
        ry = y
        for cluster in panel["clusters"]:
            if captioned and cluster["build"]:
                text(BAR_X, ry + 12, cluster["build"], muted=True)
                ry += CAPTION_H
            rows = [("ippon", cluster["ippon"], None)] + [
                (s.bare(s.short_runtime(r)), r["value"], r) for r in cluster["rivals"]
            ]
            for name, value, rival in rows:
                ours = rival is None
                width = max(BAR_MAX * value / top, 1.5)
                text(NAME_RIGHT, ry + BAR_H - 2, name, weight=600 if ours else 400, anchor="end")
                parts.append(
                    f'<rect x="{BAR_X}" y="{ry + 1:.1f}" width="{width:.1f}" height="{BAR_H}" rx="3" '
                    f'fill="{theme["long" if ours else "short"]}"><title>{value:,.6f} {unit}</title></rect>'
                )
                text(BAR_X + width + 6, ry + BAR_H - 2, amount(value, unit))
                if rival is not None and rival.get("ratio") is not None:
                    text(VERDICT_RIGHT, ry + BAR_H - 2, verdict(rival, unit), size=12, weight=600, anchor="end")
                ry += ROW_H
        y = max(ry, y + 4 + LINE * len(header)) + PANEL_GAP
    machines = sorted(
        {f"{r['machine']['chip']}, {r['machine']['memory_bytes'] / 2**30:g} GB" for r in model["records"]}
    )
    timed = "the median over ippon's requests" if model["alone"] else "the median over the same requests"
    for line in wrap(
        f"{' and '.join(machines)} · each value is {timed} · method and versions in docs/benchmarks.md",
        CHART_WIDTH - 2 * MARGIN,
    ):
        text(MARGIN, y, line, muted=True)
        y += 15
    height = round(y + 4)
    background = "#ffffff" if theme["text"] == THEMES["light"]["text"] else "#0d1117"
    description = (
        "Orange bars show the model's measured times on ippon."
        if model["alone"]
        else "Orange bars are ippon and blue bars another runtime on the same build."
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_WIDTH}" height="{height}" '
            f'viewBox="0 0 {CHART_WIDTH} {height}" font-family="{FONT}" role="img" '
            f'aria-label="{escape(model["name"])}">',
            f"<title>{escape(model['name'])}</title>",
            f"<desc>{description} Every workload has its own linear scale from zero.</desc>",
            f'<rect width="{CHART_WIDTH}" height="{height}" fill="{background}"/>',
            *parts,
            "</svg>",
            "",
        ]
    )


def hero(report: dict, workload: str | None = None) -> list[tuple[str, float, int, int]]:
    """Each runtime's time against ippon's on the same requests, as the geometric mean of ippon's lead over every model
    and workload it ran, or over one workload's panels (requests per second turned into time): (runtime, ratio, cells,
    models), ippon first at 1."""
    cells: dict[str, list[tuple[str, float]]] = {}
    for model in s.readme_chart_models(report):
        for panel in model["panels"]:
            if workload and panel["key"] != workload:
                continue
            for cluster in panel["clusters"]:
                for rival in cluster["rivals"]:
                    # the runtime by name alone; its version is in the per-model charts and docs/benchmarks.md
                    name = "the creators' own code" if rival["group"] == "maker" else s.short_runtime(rival).split()[0]
                    cells.setdefault(name, []).append((model["slug"], rival["ratio"]))
    rows = [(name, s.gmean([r for _, r in v]) or 1.0, len(v), len({m for m, _ in v})) for name, v in cells.items()]
    return [("ippon", 1.0, 0, 0), *sorted(rows, key=lambda row: row[1])]


def hero_svg(
    rows: list[tuple[str, float, int, int]],
    theme: dict,
    title: str,
    aria: str = "Time to answer the same requests, and how much faster ippon is than each runtime",
    say: Callable[[float], str] = lambda ratio: "ippon " + s.compared(ratio),
) -> str:
    """One horizontal bar a runtime, its length its measure against ippon's, under a title that says what the bars
    average: the README's first chart (time), and with `aria` and `say` its memory chart."""
    label_w, bar_x, row_h, bar_h, top = 190, 200, 34, 20, 40
    scale = (CHART_WIDTH - bar_x - 130) / max(r for _, r, _, _ in rows)  # room for the longest label
    height = top + len(rows) * row_h + 4
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CHART_WIDTH} {height}" width="{CHART_WIDTH}" '
        f'height="{height}" font-family="{FONT}" role="img" aria-label="{escape(aria)}">',
        f'<text x="{bar_x}" y="18" font-size="15" font-weight="600" fill="{theme["text"]}">{escape(title)}</text>',
    ]
    for i, (name, ratio, _, _) in enumerate(rows):
        y = top + i * row_h
        ours = name == "ippon"
        fill = theme["long" if ours else "short"]
        weight = "600" if ours else "400"
        parts.append(
            f'<text x="{label_w}" y="{y + bar_h - 5}" text-anchor="end" font-size="15" font-weight="{weight}" '
            f'fill="{theme["text"]}">{escape(name)}</text>'
        )
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{ratio * scale:.1f}" height="{bar_h}" rx="3" fill="{fill}"/>')
        value = "" if ours else say(ratio)
        parts.append(
            f'<text x="{bar_x + ratio * scale + 8:.1f}" y="{y + bar_h - 5}" font-size="14" '
            f'font-weight="{weight}" fill="{theme["muted"] if not ours else theme["text"]}">{value}</text>'
        )
    parts.append("</svg>\n")
    return "".join(parts)


def memory(campaign: str = s.MEMORY_CAMPAIGN) -> list[tuple[str, float, int, int]]:
    """Ollama's highest memory against ippon's on the same builds, as the geometric mean over the memory campaign's
    Ollama pairs, each server's highest footprint over every workload of the round: (runtime, ratio, builds, builds),
    ippon first at 1, the hero chart's rows. oMLX runs the same MLX checkpoints as ippon and needs about 10% more, which
    the README says in a sentence rather than a bar."""
    folder = s.ROOT / "benchmarks/results" / campaign

    def peak(engine: str) -> float:
        runs = [json.loads(f.read_text()) for f in folder.glob(f"r*/{engine}.*.json")]
        return max(r["memory"]["server"]["lifetime_max_phys_footprint"] for r in runs if r.get("complete"))

    present = {f.name.rpartition(".")[0].rpartition(".")[0] for f in folder.glob("r*/*.json")}
    ratios: dict[str, list[float]] = {}
    for group, _, other, ours in s.PAIRS:
        if group == "ollama" and other in present and ours in present:
            ratios.setdefault("Ollama", []).append(peak(other) / peak(ours))
    rows = [(name, s.gmean(v) or 1.0, len(v), len(v)) for name, v in ratios.items()]
    return [("ippon", 1.0, 0, 0), *sorted(rows, key=lambda row: row[1])]


def more_memory(ratio: float) -> str:
    """How much less memory ippon needs than another runtime that needs `ratio` times ippon's: "ippon 31% less"."""
    return f"ippon {(1 - 1 / ratio) * 100:.0f}% less"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="*", type=Path, help="result folders, one per round")
    p.add_argument(
        "--recorded",
        action="store_true",
        help="with --readme-round: every README campaign recorded so far, besides any folders given",
    )
    p.add_argument(
        "--out",
        type=Path,
        help="the files' stem: <out>-<model>-light.svg and -dark.svg (default: docs/assets/readme)",
    )
    p.add_argument("--readme-round", action="store_true", help="draw the README's charts, one per model (required)")
    p.add_argument(
        "--partial",
        action="store_true",
        help="with --readme-round: the finished pairs of unfinished attempt-rN folders",
    )
    p.add_argument(
        "--more",
        nargs="+",
        type=Path,
        default=[],
        help="with --readme-round: also draw the other workloads' campaign rounds (attempt-rN: partial)",
    )
    p.add_argument(
        "--older",
        nargs="+",
        type=Path,
        default=[],
        help="with --readme-round: older campaign folders whose one-question times chart the pairs not yet measured",
    )
    p.add_argument("--exclude-pair", action="append", default=[], metavar="OTHER_ENGINE=REASON")
    args = p.parse_args()
    if not args.readme_round:
        p.error("only the README's charts are drawn: give --readme-round")
    if args.recorded:
        text, more_rounds = s.recorded_campaigns()
        args.folders = [*text, *args.folders]
        args.more = [*more_rounds, *args.more]
    if not args.folders:
        p.error("give result folders, or --readme-round --recorded")
    try:
        report = s.readme_comparisons(args.folders, s.exclusions(args.exclude_pair), args.partial)
        if args.more:
            report["more"] = s.readme_more(args.more)
    except ValueError as exc:
        p.error(str(exc))
    if args.older:
        report["older"] = s.readme_older(args.older, report)
    stem = args.out or Path(s.README_CHARTS)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():  # the README's first chart
        path = stem.with_name(f"{stem.name}-hero-{name}.svg")
        path.write_text(hero_svg(hero(report), theme, "On average, over every model and workload"))
        print(path)
        path = stem.with_name(f"{stem.name}-memory-{name}.svg")
        path.write_text(
            hero_svg(
                memory(),
                theme,
                "On average, the most memory used while answering",
                "The most memory each runtime used while answering, relative to ippon",
                more_memory,
            )
        )
        print(path)
    for model in s.readme_chart_models(report):  # a chart a model
        for name, theme in THEMES.items():
            path = stem.with_name(f"{stem.name}-{model['slug']}-{name}.svg")
            path.write_text(model_svg(model, theme))
            print(path)


if __name__ == "__main__":
    main()
