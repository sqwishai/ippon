"""`ippon eval DATA.jsonl --model A [--model B ...]`: how models answer the caller's own labelled examples, as
they come (no calibration).

Each line of DATA.jsonl is a labelled request: `{"state": ..., "questions": {...}, "labels": {question_id: expected}}`
(with "id", "images" or "audio" if wanted; "questions" may instead come from one shared file, `--questions`). What is
expected is a noul's true or false (or "yes"/"no"), a choice's label, or a score's level: its index, lowest 0, or its
text. The data is read and checked before any model loads; a line that isn't a request a model can be asked, or a
label that isn't one of its question's options, stops the run naming its line.

The models load one at a time, each closed before the next loads, so two are never resident together. Every row is
answered through the batched path (`batches.answer_all`), a row's own error (a request the model can't be asked) kept
as that row's. For each model and question the report gives the accuracy and macro-F1 of the answers, the confusion
matrix (expected by answered), how many rows were refused and how many answered with input cut to fit the model's
window; then, for each confidence threshold, how many answers have at least that confidence (their top option's
probability), their accuracy, and for a noul the false yeses and noes among them; and where two models' answers
differ. A threshold is a report of what an abstention policy would keep: the probabilities stay the model's own,
unchanged, in the per-row results file.

A noul also gets its flagging (`flagging`): a row is flagged when the model's P(true) is at least a threshold t, as a
judge flags a failing trace, and 0.5 is often the wrong t (a model that gives every failure 0.40 and every pass 0.10
is perfect at 0.30 and catches nothing at 0.50: Husain and Saha, "How to Use Decision Models for Evals", 2026). It
reports precision, recall and F1 at 0.5 and across t, the t of the highest F1 that meets the caller's precision and
recall floors, that choice checked on held-out rows (stratified folds, each t chosen on the other folds, with how many
folds had no t meeting the floors), and how calibrated P(true) is: its Brier score and, by tenths, the mean predicted
against the share that were true. Nothing is fitted to the probabilities: they stay the model's own.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .batches import answer_all
from .errors import DecisionError, InvalidRequest
from .schema import Question, parse_request

THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SWEEP = tuple(round(0.1 * i, 1) for i in range(1, 10))  # a noul's flagging thresholds shown, besides the chosen one
FOLDS = 5  # held-out folds for checking a chosen flagging threshold
BINS = 10  # the calibration table's bins of P(true)
SHOWN = 10  # disagreeing rows printed per question; the results file has every row


@dataclass(frozen=True)
class Row:
    """A labelled request: its line (from 1), its id (the line's "id", else its line), its System One body, its
    questions as parsed, the expected option of each labelled question (an option key: "true"/"false", a label,
    a level's index), and any per-row media error that refuses only this row rather than the whole file."""

    line: int
    id: str
    body: dict[str, Any]
    questions: dict[str, Question]
    labels: dict[str, str]
    group: str | None = None  # rows that belong together (one agent trace's steps): held-out folds keep them together
    error: DecisionError | None = None


def _has_media(body: Mapping[str, Any]) -> bool:
    if any(body.get(k) for k in ("images", "videos", "audio", "files")):
        return True
    state = body.get("state")
    return isinstance(state, list) and any(isinstance(part, dict) and part.get("type") != "text" for part in state)


def load(path: str | Path, questions: Mapping[str, Any] | None = None) -> list[Row]:
    """The labelled rows of a JSONL file, each checked as a request (`parse_request`) with its labels among its
    questions' options; InvalidRequest naming the line otherwise."""
    rows = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InvalidRequest(f"can't read {path} ({error.strerror}): give a JSONL file of labelled requests") from None
    except UnicodeError:
        raise InvalidRequest(f"can't read {path} as UTF-8: save the labelled requests as UTF-8 JSON lines") from None
    for n, text in enumerate(lines, 1):
        if not text.strip():
            continue
        where = f"{path} line {n}"
        try:
            item = json.loads(text)
        except json.JSONDecodeError as error:
            raise InvalidRequest(f"{where} is not JSON: {error.msg}") from None
        if not isinstance(item, dict) or not isinstance(item.get("labels"), dict) or not item["labels"]:
            raise InvalidRequest(
                f'{where}: a row is {{"state": ..., "questions": {{...}}, "labels": {{id: expected}}}}'
            )
        body = {k: v for k, v in item.items() if k not in ("id", "labels", "group")}
        if questions is not None:
            body.setdefault("questions", dict(questions))
        elif "questions" not in body:
            raise InvalidRequest(
                f"{where}: no questions: give each row its own, or one file for all (--questions FILE)"
            )
        row_error: DecisionError | None = None
        try:
            request = parse_request(body)
        except InvalidRequest as error:
            if not _has_media(body):
                raise InvalidRequest(f"{where}: {error}") from None
            try:  # bad media on one row (e.g. an unsupported HEIC photo) refuses that row, not the whole file
                request = parse_request({"state": "", "questions": body.get("questions")})
            except InvalidRequest:
                raise InvalidRequest(f"{where}: {error}") from None
            row_error = error
        asked = {q.id: q for q in request.questions}
        labels = {}
        for qid, expected in item["labels"].items():
            if qid not in asked:
                raise InvalidRequest(f"{where}: a label for {qid!r}, which no question asks ({', '.join(asked)})")
            labels[qid] = option(asked[qid], expected, where)
        group = None if item.get("group") is None else str(item["group"])
        rows.append(Row(n, str(item.get("id", n)), body, asked, labels, group, error=row_error))
    if not rows:
        raise InvalidRequest(f"{path}: no labelled rows")
    return rows


def option(question: Question, expected: Any, where: str = "") -> str:
    """The option key `expected` names for `question`: a noul's true/false (yes/no, 1/0), a choice's label, a
    score's level as its index or its text."""
    q, given = question, expected
    if q.type == "noul":
        text = str(given).strip().lower()
        found = {"true": "true", "yes": "true", "1": "true", "false": "false", "no": "false", "0": "false"}.get(text)
    elif q.type == "choice":
        found = given if isinstance(given, str) and given in q.options else None
    else:
        levels = {str(level): key for key, level in q.criteria}
        found = str(given) if str(given) in q.options and not isinstance(given, bool) else levels.get(str(given))
    if found is None:
        shown = "true or false" if q.type == "noul" else ", ".join(q.options if q.type == "choice" else levels)
        raise InvalidRequest(f"{where}: {given!r} is not an answer to {q.id} ({shown})")
    return found


@dataclass
class Answered:
    """A model's answer to a row: per question, its probabilities by option; or the error that refused the row."""

    row: Row
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    truncated: bool = False
    error: DecisionError | None = None

    def top(self, question: str) -> tuple[str, float]:
        p = self.probabilities[question]
        best = max(p, key=p.__getitem__)
        return best, p[best]

    def to_json(self, model: str) -> dict[str, Any]:
        out: dict[str, Any] = {"line": self.row.line, "id": self.row.id, "model": model}
        if self.error is not None:
            return out | self.error.to_json()
        answers = {}
        for qid, p in self.probabilities.items():
            top, confidence = self.top(qid)
            expected = self.row.labels.get(qid)
            answers[qid] = {"answer": top, "confidence": confidence, "expected": expected}
            answers[qid] |= {"correct": None if expected is None else top == expected, "probabilities": p}
        return out | {"answers": answers, "truncated": self.truncated}


def run(
    models: Sequence[tuple[str, Callable[[], Any]]],
    rows: Sequence[Row],
    batch: int | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> tuple[dict[str, list[Answered]], dict[str, dict[str, Any]]]:
    """Every model's answers to every row, by model name: each loaded (`load()`, a Decider), answered through the
    batched path, then closed before the next loads; and of each model, the build it was loaded from, the seconds
    its load and its answers took. A model that can't be loaded (a DecisionError: too little memory free, a build
    ippon doesn't read) has that error in its `about` and no answers, and the others are still asked."""
    out: dict[str, list[Answered]] = {}
    about: dict[str, dict[str, Any]] = {}
    for name, load in models:
        start = time.perf_counter()
        try:
            decider = load()
        except DecisionError as error:
            about[name] = error.to_json()
            log(f"{name}: not loaded: {error}")
            continue
        loaded = time.perf_counter()
        try:
            answers: list[Answered | None] = [
                Answered(row, error=row.error) if row.error is not None else None for row in rows
            ]
            runnable = [i for i, row in enumerate(rows) if row.error is None]
            for pos, result in answer_all(decider, [rows[i].body for i in runnable], batch):
                row = rows[runnable[pos]]
                if isinstance(result, DecisionError):
                    answers[runnable[pos]] = Answered(row, error=result)
                    continue
                by = {qid: a.by_option for qid, a in result.answers.items()}
                answers[runnable[pos]] = Answered(row, by, bool(result.diagnostics["truncated"]))
            about[name] = {"build": decider.details.get("build")}
        finally:
            decider.close()
        took = time.perf_counter() - loaded
        about[name] |= {"load_s": round(loaded - start, 2), "answer_s": round(took, 2)}
        log(f"{name}: loaded in {loaded - start:.1f} s, {len(rows)} rows in {took:.1f} s ({len(rows) / took:.1f}/s)")
        out[name] = [a for a in answers if a is not None]
    return out, about


def f1(pairs: Sequence[tuple[str, str]]) -> float:
    """The macro-F1 of (expected, answered) pairs: the mean, over every option either names, of its F1 (0 for an
    option never answered right)."""
    scores = []
    for c in {x for pair in pairs for x in pair}:
        tp = sum(e == c and a == c for e, a in pairs)
        fp = sum(e != c and a == c for e, a in pairs)
        fn = sum(e == c and a != c for e, a in pairs)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


Scores = Sequence[tuple[bool, float]]  # a noul's rows: whether each is true, and the model's P(true)


def _counts(scores: Scores, t: float) -> tuple[int, int, int]:
    """True positives, false positives and false negatives when a row is flagged at P(true) >= t."""
    tp = sum(y and p >= t for y, p in scores)
    fp = sum(not y and p >= t for y, p in scores)
    return tp, fp, sum(y for y, _ in scores) - tp


def _rates(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "flagged": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
    }


def choose(scores: Scores, min_precision: float = 0.0, min_recall: float = 0.0) -> float | None:
    """The flagging threshold, among the rows' own P(true), with the highest F1 of those that flag something and
    meet both floors (the higher threshold on a tie); None when none does."""
    best: tuple[float, float] | None = None
    for t in sorted({p for _, p in scores}):
        r = _rates(*_counts(scores, t))
        meets = r["f1"] > 0 and r["precision"] >= min_precision and r["recall"] >= min_recall
        if meets and (best is None or r["f1"] >= best[1]):
            best = (t, r["f1"])
    return None if best is None else best[0]


def copyable(scores: Scores, t: float) -> float:
    """`t` with as few decimals as flag exactly the rows `t` flags: rounded down, and still above every row's P(true)
    under `t`. A threshold of 0.7985 shown as 0.799 left out the row at 0.7985."""
    below = max((p for _, p in scores if p < t), default=0.0)
    for digits in range(1, 17):
        r = float(math.floor(t * 10**digits) / 10**digits)
        if below < r <= t:
            return r
    return t


def held_out(
    scores: Scores,
    min_precision: float,
    min_recall: float,
    folds: int = FOLDS,
    groups: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """`choose` checked on rows it didn't see: the rows in `folds` stratified folds (each class dealt to them in turn,
    in file order), each fold flagged at the threshold chosen on the others, and the flags pooled; how many folds had
    no threshold meeting the floors. With `groups` (one per row), a group's rows stay in one fold, the groups dealt
    in turn from the most true rows down, so no threshold is chosen on rows of the group it is judged on. None with
    fewer than two rows of either class, or two groups."""
    true = [i for i, (y, _) in enumerate(scores) if y]
    false = [i for i, (y, _) in enumerate(scores) if not y]
    if groups is None:
        k = min(folds, len(true), len(false))
        fold = {i: n % k for group in (true, false) for n, i in enumerate(group)} if k >= 2 else {}
    else:
        positives: dict[str, int] = {}
        for g, (y, _) in zip(groups, scores, strict=True):
            positives[g] = positives.get(g, 0) + y
        order = sorted(positives, key=lambda g: -positives[g])  # stable: ties keep their first appearance
        k = min(folds, len(order)) if true and false else 0
        dealt = {g: n % k for n, g in enumerate(order)} if k >= 2 else {}
        fold = {i: dealt[g] for i, g in enumerate(groups)} if dealt else {}
    if k < 2:
        return None
    tp = fp = fn = 0
    chosen: list[float] = []
    found: list[float | None] = []
    for f in range(k):
        t = choose([s for i, s in enumerate(scores) if fold[i] != f], min_precision, min_recall)
        found.append(t)
        if t is None:
            continue
        chosen.append(t)
        a, b, c = _counts([s for i, s in enumerate(scores) if fold[i] == f], t)
        tp, fp, fn = tp + a, fp + b, fn + c
    judged = {f for f, t in enumerate(found) if t is not None}
    rows = sum(fold[i] in judged for i in range(len(scores)))
    return {"folds": k, "without_threshold": k - len(chosen), "thresholds": chosen, "rows": rows} | _rates(tp, fp, fn)


def flagging(
    scores: Scores,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
    folds: int = FOLDS,
    groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """A noul's rows flagged at P(true) >= t: at 0.5, across SWEEP, at the threshold `choose` picks and on held-out
    rows (`held_out`); and P(true)'s calibration: the Brier score and, per tenth of P(true), its rows, their mean
    P(true) and the share of them true."""
    t = choose(scores, min_precision, min_recall)
    bins: dict[int, list[tuple[bool, float]]] = {}
    for y, p in scores:
        bins.setdefault(min(int(p * BINS), BINS - 1), []).append((y, p))
    return {
        "rows": len(scores),
        "true": sum(y for y, _ in scores),
        "floors": {"precision": min_precision, "recall": min_recall},
        "at_half": _rates(*_counts(scores, 0.5)),
        "sweep": [{"threshold": x} | _rates(*_counts(scores, x)) for x in SWEEP],
        "best": None if t is None else {"threshold": t, "use": copyable(scores, t)} | _rates(*_counts(scores, t)),
        "held_out": held_out(scores, min_precision, min_recall, folds, groups),
        "brier": sum((p - y) ** 2 for y, p in scores) / len(scores),
        "calibration": [
            {
                "from": b / BINS,
                "to": (b + 1) / BINS,
                "rows": len(group),
                "predicted": sum(p for _, p in group) / len(group),
                "observed": sum(y for y, _ in group) / len(group),
            }
            for b, group in sorted(bins.items())
        ],
    }


def summary(
    answered: Sequence[Answered], min_precision: float = 0.0, min_recall: float = 0.0, folds: int = FOLDS
) -> dict[str, Any]:
    """Per question asked with a label: rows, accuracy, macro-F1, refusals, truncations, the confusion matrix and the
    threshold table, for one model; for a noul its flagging (`flagging`) too."""
    out: dict[str, Any] = {}
    asked = list(dict.fromkeys(qid for a in answered for qid in a.row.labels))
    for qid in asked:
        labelled = [a for a in answered if qid in a.row.labels]
        done = [a for a in labelled if a.error is None]
        pairs = [(a.row.labels[qid], a.top(qid)[0]) for a in done]
        kinds = {a.row.questions[qid].type for a in labelled}
        seen = {o for pair in pairs for o in pair}
        options = [o for o in dict.fromkeys(o for a in labelled for o in a.row.questions[qid].options) if o in seen]
        confusion = {e: {o: sum(p == (e, o) for p in pairs) for o in options} for e in options}
        first = labelled[0].row.questions[qid]
        levels = {o: str(first.description(o)) for o in options} if first.type == "score" else {}
        thresholds = []
        for t in THRESHOLDS:
            kept = [(e, a) for (e, a), x in zip(pairs, done, strict=True) if x.top(qid)[1] >= t]
            row: dict[str, Any] = {
                "threshold": t,
                "kept": len(kept),
                "accuracy": sum(e == a for e, a in kept) / len(kept) if kept else None,
            }
            if kinds == {"noul"}:
                row |= {"false_yes": sum(a == "true" != e for e, a in kept)}
                row |= {"false_no": sum(a == "false" != e for e, a in kept)}
            thresholds.append(row)
        out[qid] = {
            "rows": len(labelled),
            "answered": len(done),
            "refused": len(labelled) - len(done),
            "truncated": sum(a.truncated for a in done),
            "accuracy": sum(e == a for e, a in pairs) / len(pairs) if pairs else None,
            "macro_f1": f1(pairs) if pairs else None,
            "confusion": confusion,
            "levels": levels,  # a score's level texts, by index
            "thresholds": thresholds,
        }
        if kinds == {"noul"} and done:
            scores = [(a.row.labels[qid] == "true", a.probabilities[qid]["true"]) for a in done]
            groups = [a.row.group or a.row.id for a in done] if any(a.row.group for a in done) else None
            out[qid]["flagging"] = flagging(scores, min_precision, min_recall, folds, groups)
    return out


def disagreements(results: Mapping[str, Sequence[Answered]]) -> dict[str, list[dict[str, Any]]]:
    """Per question, the rows where the models' answers differ: each model's answer and confidence, and what was
    expected."""
    names = list(results)
    out: dict[str, list[dict[str, Any]]] = {}
    if len(names) < 2:
        return out
    for i, row in enumerate(a.row for a in results[names[0]]):
        for qid in row.labels:
            answers = {n: results[n][i] for n in names}
            if any(a.error is not None for a in answers.values()):
                continue
            tops = {n: a.top(qid) for n, a in answers.items()}
            if len({top for top, _ in tops.values()}) > 1:
                out.setdefault(qid, []).append(
                    {"line": row.line, "id": row.id, "expected": row.labels[qid]}
                    | {"answers": {n: {"answer": t, "confidence": c} for n, (t, c) in tops.items()}}
                )
    return out


def report(
    results: Mapping[str, Sequence[Answered]],
    about: Mapping[str, Mapping[str, Any]] | None = None,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
    folds: int = FOLDS,
) -> dict[str, Any]:
    """The whole report as JSON: of each model (`run`'s `about`), and per model and question, a noul's flagging with
    the precision and recall floors and folds given; and where the models disagree."""
    return {
        "about": dict(about or {}),
        "models": {name: summary(a, min_precision, min_recall, folds) for name, a in results.items()},
        "disagreements": disagreements(results),
    }


def text(found: Mapping[str, Any]) -> str:
    """The report for a person to read."""
    lines: list[str] = []
    for name, about in found["about"].items():
        if "error" in about:
            lines += [f"{name}: not loaded: {about['error']['message']}", ""]
    for name, questions in found["models"].items():
        lines.append(f"{name}{_about(found['about'].get(name))}")
        rows = [("  question", "rows", "accuracy", "macro-F1", "refused", "truncated")]
        for qid, s in questions.items():
            rows.append(
                (
                    f"  {qid}",
                    str(s["rows"]),
                    _rate(s["accuracy"]),
                    _rate(s["macro_f1"]),
                    str(s["refused"]),
                    str(s["truncated"]),
                )
            )
        lines += _table(rows)
        for qid, s in questions.items():
            options = list(s["confusion"])
            shown = {o: f"{o} {s['levels'][o]}" if o in s["levels"] else o for o in options}  # a score's level text
            lines.append(f"  {qid}: expected (rows) by answered (columns)")
            lines += _table(
                [
                    ("    ", *(shown[o] for o in options)),
                    *((f"    {shown[e]}", *(str(s["confusion"][e][o]) for o in options)) for e in options),
                ]
            )
            noul = "false_yes" in s["thresholds"][0]
            head = ("    confidence at least", "kept", "accuracy", *(("false yes", "false no") if noul else ()))
            table = [head]
            for t in s["thresholds"]:
                kept = f"{t['kept']} ({t['kept'] / s['answered']:.0%})" if s["answered"] else "0"
                more = (str(t["false_yes"]), str(t["false_no"])) if noul else ()
                table.append((f"    {t['threshold']:.2f}", kept, _rate(t["accuracy"]), *more))
            lines += _table(table)
            if "flagging" in s:
                lines += _flagging(qid, s["flagging"])
        lines.append("")
    if found["disagreements"]:
        lines.append("Where the models' answers differ:")
        for qid, rows_ in found["disagreements"].items():
            lines.append(f"  {qid}: {len(rows_)} rows")
            for d in rows_[:SHOWN]:
                said = "; ".join(f"{n} {a['answer']} {a['confidence']:.2f}" for n, a in d["answers"].items())
                lines.append(f"    line {d['line']} ({d['id']}): expected {d['expected']}; {said}")
            if len(rows_) > SHOWN:
                lines.append(f"    and {len(rows_) - SHOWN} more, in the results file")
    lines.append(
        "Uncalibrated: each model's own probabilities, as published. A confidence threshold above is a report of what "
        "an abstention policy would keep, and a flagging threshold of what a judge flagging at it would catch; neither "
        "changes an answer."
    )
    return "\n".join(lines)


def _flagging(qid: str, f: Mapping[str, Any]) -> list[str]:
    """A noul's flagging, for a person to read."""
    lines = [f"  {qid}: flagged when P(true) is at least t ({f['true']} of {f['rows']} rows are true)"]
    rows = [("    t", "flagged", "precision", "recall", "F1")]
    shown = [{"threshold": 0.5} | f["at_half"], *(x for x in f["sweep"] if x["threshold"] != 0.5)]
    for x in sorted(shown, key=lambda x: x["threshold"]):
        rows.append(
            (f"    {x['threshold']:.2f}", str(x["flagged"]), _rate(x["precision"]), _rate(x["recall"]), _rate(x["f1"]))
        )
    lines += _table(rows)
    floors = f["floors"]
    asked = (
        f"precision at least {floors['precision']:.2f} and recall at least {floors['recall']:.2f}"
        if floors["precision"] or floors["recall"]
        else "the highest F1"
    )
    b = f["best"]
    lines.append(
        f"    no threshold meets {asked}"
        if b is None
        else f"    chosen for {asked}: t = {b['use']:g} (flag P(true) >= t), flagging {b['flagged']}, precision "
        f"{_rate(b['precision'])}, recall {_rate(b['recall'])}, F1 {_rate(b['f1'])}"
    )
    h = f["held_out"]
    if h is not None:
        missing = (
            f"; on {h['rows']} of {f['rows']} rows, since {h['without_threshold']} of {h['folds']} folds had no "
            "threshold meeting it"
            if h["without_threshold"]
            else ""
        )
        lines.append(
            f"    held out ({h['folds']} folds, each t chosen on the others): precision {_rate(h['precision'])}, "
            f"recall {_rate(h['recall'])}, F1 {_rate(h['f1'])}{missing}"
        )
    lines.append(
        f"    calibration: Brier score {f['brier']:.3f} (0 is perfect); by P(true), mean predicted against share true"
    )
    lines += _table(
        [
            ("      P(true)", "rows", "predicted", "true"),
            *(
                (
                    f"      {c['from']:.1f}-{c['to']:.1f}",
                    str(c["rows"]),
                    f"{c['predicted']:.2f}",
                    f"{c['observed']:.2f}",
                )
                for c in f["calibration"]
            ),
        ]
    )
    return lines


def _about(about: Mapping[str, Any] | None) -> str:
    """(repo @ commit), loaded in 2.1 s, 120 rows answered in 3.4 s"""
    if not about:
        return ""
    build = about.get("build") or {}
    where = build.get("repo") or build.get("ollama") or build.get("path") or ""
    at = f" @ {build['revision'][:12]}" if build.get("revision") else ""
    return f" ({where}{at}): loaded in {about['load_s']:.1f} s, answered in {about['answer_s']:.1f} s"


def _rate(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    return ["  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)).rstrip() for r in rows]
