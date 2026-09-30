"""Score raw benchmark logs without calling any engine: bench/runs/<set>/<engine>.jsonl -> bench/runs/<set>/report.md.

    uv run python bench/score.py bench/runs/questions           # rewrite report.md from the raw logs
    uv run python bench/score.py bench/runs/questions --check   # exit 1 if report.md is not what the logs give

A raw log is one JSON object per line. The first is `{"header": {set, set_sha256, engine, model, reps, warmup,
started, host}}` and the last is `{"footer": {finished}}`, written only when the run completes. Every line between
is one engine call, `{item, call, rep, answers, error, ms, pressure}`, where `call` is `cold`, `warmup`, `timed`
(rep 0..reps-1) or `reversed` (one call per item with choice options reversed).

`--check` compares report.md with the logs; it does not fail on the engine errors a report counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("normal", "warn", "critical")
SCORE_RULES = ("round",)
SAME_IN_EVERY_LOG = ("set", "set_sha256", "reps", "warmup")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Every line must parse: a log cut short by a killed run is an error, not a shorter run."""
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{n}: not a JSON line ({e}); the run was cut short, measure it again") from e
    return out


def check_scoring(item: dict) -> None:
    """An item may name how its score questions are graded; only the default, rounding the score, exists."""
    rule = item.get("scoring", {}).get("score", "round")
    if rule not in SCORE_RULES:
        raise SystemExit(f"item {item['id']}: unknown score rule {rule!r}, want one of {SCORE_RULES}")


def load_dir(d: Path) -> tuple[dict, list[dict], dict[str, tuple[dict, list[dict]]]]:
    """(header fields every log shares, set items, {engine: (header, call records)}). Refuses logs measured on
    another version of the set or with other reps, so one report never mixes them."""
    logs = sorted(d.glob("*.jsonl"))
    if not logs:
        raise SystemExit(f"{d}: no raw logs (*.jsonl)")
    runs, common = {}, None
    for path in logs:
        lines = load_jsonl(path)
        if not lines or "header" not in lines[0]:
            raise SystemExit(f"{path}: first line is not a header")
        if "footer" not in lines[-1]:
            raise SystemExit(f"{path}: no footer, the run was interrupted; measure it again")
        header = lines[0]["header"]
        if header["engine"] != path.stem or header["engine"] in runs:
            raise SystemExit(f"{path}: header says engine {header['engine']!r}; one log per engine, named after it")
        shared = {k: header[k] for k in SAME_IN_EVERY_LOG}
        if common is None:
            common = shared
        elif shared != common:
            raise SystemExit(f"{path}: {shared} differs from {logs[0].name}: {common}; "
                             "one directory holds logs of one set and one reps setting")
        runs[header["engine"]] = (header, lines[1:-1])
    set_path = ROOT / common["set"]
    if (now := sha256(set_path)) != common["set_sha256"]:
        stale = [p.name for p in logs]
        raise SystemExit(f"{', '.join(stale)}: measured on {common['set']} sha256 {common['set_sha256'][:12]}, "
                         f"the file is now {now[:12]}; measure again")
    items = load_jsonl(set_path)
    for it in items:
        check_scoring(it)
    return common, items, runs


def decide(q: dict, answer: dict) -> Any:
    """The discrete decision an answer makes: noul -> bool, choice -> option key, score -> criterion index."""
    kind = q["type"]
    if kind == "noul":
        return answer["noul"] >= 0.5
    if kind == "choice":
        return answer["choice"]
    return math.floor(answer["score"] + 0.5)  # round() sends 0.5 to 0 and 2.5 to 2


def gold(q: dict, expected: Any) -> Any:
    return bool(expected) if q["type"] == "noul" else int(expected) if q["type"] == "score" else expected


def timed(records: list[dict]) -> dict[tuple[str, int], dict]:
    """Answered timed calls by (item, rep)."""
    return {(r["item"], r["rep"]): r["answers"] for r in records if r["call"] == "timed" and not r["error"]}


def accuracy(items: list[dict], records: list[dict]) -> tuple[int, int]:
    """(right, graded) over decisions (item x expected question), each graded once from rep 0. Repeats of the
    same input are not independent samples; a decision whose rep 0 failed is not graded."""
    answered, right, n = timed(records), 0, 0
    for it in items:
        if (a := answered.get((it["id"], 0))) is None:
            continue
        for qid, want in it["expected"].items():
            q = it["questions"][qid]
            n += 1
            right += decide(q, a[qid]) == gold(q, want)
    return right, n


def rep_disagree(items: list[dict], records: list[dict]) -> tuple[int, int]:
    """(changed, compared): graded decisions where some later answered rep decided otherwise than rep 0."""
    answered, changed, n = timed(records), 0, 0
    for it in items:
        if (first := answered.get((it["id"], 0))) is None:
            continue
        later = [a for (item, rep), a in answered.items() if item == it["id"] and rep > 0]
        if not later:
            continue
        for qid in it["expected"]:
            q = it["questions"][qid]
            n += 1
            changed += any(decide(q, a[qid]) != decide(q, first[qid]) for a in later)
    return changed, n


def order_flip(items: list[dict], records: list[dict]) -> tuple[int, int]:
    """(flipped, compared): choice answers of rep 0 that changed when the options were listed in reverse."""
    answered = timed(records)
    flipped = {r["item"]: r["answers"] for r in records if r["call"] == "reversed" and not r["error"]}
    flips = n = 0
    for it in items:
        first, rev = answered.get((it["id"], 0)), flipped.get(it["id"])
        if first is None or rev is None:
            continue
        for qid, q in it["questions"].items():
            if q["type"] == "choice":
                n += 1
                flips += rev[qid]["choice"] != first[qid]["choice"]
    return flips, n


def pct(xs: list[float], p: int) -> float:
    return statistics.quantiles(xs, n=100, method="inclusive")[p - 1] if len(xs) > 1 else xs[0]


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    p = right / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def ratio(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})"


def row(header: dict, reps: int, items: list[dict], records: list[dict]) -> tuple[str, list[str]]:
    """(table row, error lines) for one engine."""
    ok = [r for r in records if not r["error"]]
    cold = next((f"{r['ms']:.1f}" for r in ok if r["call"] == "cold"), "failed")
    all_ms = [r["ms"] for r in ok if r["call"] == "timed"]
    first_ms = [r["ms"] for r in ok if r["call"] == "timed" and r["rep"] == 0]
    p50, p95 = (f"{pct(all_ms, 50):.1f}", f"{pct(all_ms, 95):.1f}") if all_ms else ("failed", "failed")
    first = f"{pct(first_ms, 50):.1f}" if first_ms else "failed"

    right, n = accuracy(items, records)
    if n:
        lo, hi = wilson(right, n)
        acc = f"{right}/{n} ({right / n:.0%}, {lo * 100:.0f}-{hi * 100:.0f})"
    else:
        acc = "failed"
    changed, compared = rep_disagree(items, records)
    disagree = "n/a" if reps == 1 else f"{changed}/{compared}" if compared else "failed"
    flips, flip_n = order_flip(items, records)
    has_choice = any(q["type"] == "choice" for it in items for q in it["questions"].values())
    flip = ratio(flips, flip_n) if flip_n else "failed" if has_choice else "n/a"

    pressure = [r["pressure"] for r in records]
    worst = max(pressure, key=LEVELS.index, default="unsampled") if all(p in LEVELS for p in pressure) \
        else ", ".join(sorted(set(pressure)))
    errors = [r for r in records if r["error"]]
    engine = header["engine"]
    line = (f"| {engine} | {header['model'] or '-'} | {cold} | {first} | {p50} | {p95} | {len(all_ms)} | {acc} "
            f"| {disagree} | {flip} | {len(errors)} | {worst} |")
    return line, [f"- {engine} error: {r['item']} {r['call']}: {r['error']}" for r in errors[:5]]


def report(common: dict, items: list[dict], runs: dict[str, tuple[dict, list[dict]]]) -> str:
    decisions = sum(len(it["expected"]) for it in items)
    lines = [
        "# System One local benchmark",
        "",
        f"Set: `{common['set']}` (sha256 `{common['set_sha256'][:12]}`), {len(items)} items, {decisions} decisions.",
        f"{common['reps']} timed calls per item, sequential, {common['warmup']} warm-up calls excluded. "
        "Generated by `bench/score.py` from the raw logs next to this file.",
        "Accuracy grades each decision (item x question) once, from the first timed call (rep 0), with a 95% Wilson",
        "interval; overlapping intervals are not a difference. `rep-disagree` counts decisions that a later timed call",
        "answered differently.",
        "`first-call p50` covers only each item's first call. Repeats of the same state can be served from an engine's",
        "embedding cache (CLM caches state vectors), so all-call percentiles understate uncached latency.",
        "`order-flip` asks each item once more with every choice's options reversed and counts changed choices.",
        "",
        "| engine | model | cold ms | first-call p50 ms | all-call p50 ms | all-call p95 ms | calls | accuracy "
        "| rep-disagree | order-flip | errors | worst memory pressure |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = [row(header, common["reps"], items, records) for header, records in (runs[e] for e in sorted(runs))]
    return "\n".join(lines + [r for r, _ in rows] + [e for _, errs in rows for e in errs]) + "\n"


def render(d: Path) -> str:
    """Rewrite `<d>/report.md` from the raw logs in `d` and return it."""
    text = report(*load_dir(d))
    (d / "report.md").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path, help="bench/runs/<set>")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if report.md differs from what the logs give (engine errors do not fail it)")
    args = ap.parse_args()
    if args.check:
        path = args.dir / "report.md"
        if not path.exists() or path.read_text() != report(*load_dir(args.dir)):
            print(f"{path} is stale; run bench/score.py {args.dir}", file=sys.stderr)
            return 1
        return 0
    print(render(args.dir), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
