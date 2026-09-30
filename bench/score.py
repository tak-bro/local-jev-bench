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
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("normal", "warn", "critical")
SCORE_RULES = ("round", "argmax")
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


def score_rule(item: dict) -> str:
    """How the item's score questions are graded: `round` (the default) rounds the returned expected score, `argmax`
    takes the most probable level, as Kev's own benchmark does."""
    rule = item.get("scoring", {}).get("score", "round")
    if rule not in SCORE_RULES:
        raise SystemExit(f"item {item['id']}: unknown score rule {rule!r}, want one of {SCORE_RULES}")
    return rule


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
        score_rule(it)
    return common, items, runs


def decide(q: dict, answer: dict, rule: str = "round") -> Any:
    """The discrete decision an answer makes: noul -> bool, choice -> option key, score -> criterion index."""
    kind = q["type"]
    if kind == "noul":
        return answer["noul"] >= 0.5
    if kind == "choice":
        return answer["choice"]
    if rule == "round":
        return math.floor(answer["score"] + 0.5)  # round() sends 0.5 to 0 and 2.5 to 2
    return max(range(len(q["criteria"])), key=score_levels(q, answer).__getitem__)


def score_levels(q: dict, answer: dict) -> list[float]:
    """Probability of each score level. Engines key them all by index ("0", "1", ...) or all by criterion text; the
    form is decided over the whole answer, never per level."""
    probs = answer.get("probabilities") or {}
    by_index = [probs.get(str(i)) for i in range(len(q["criteria"]))]
    by_text = [probs.get(text) for text in q["criteria"]]
    if None not in by_index and None not in by_text and by_index != by_text:
        raise SystemExit(f"the argmax score rule cannot tell index keys from criterion keys, ambiguous: {probs!r} "
                         f"for {q['criteria']}")
    levels = by_index if None not in by_index else by_text
    if None in levels:
        raise SystemExit(f"the argmax score rule needs a probability per level, got {probs!r} for {q['criteria']}")
    return levels


def gold(q: dict, expected: Any) -> Any:
    return bool(expected) if q["type"] == "noul" else int(expected) if q["type"] == "score" else expected


def graded(items: list[dict], records: list[dict]) -> dict[tuple[str, str], tuple[dict, dict, Any, str]]:
    """{(item, question): (question, rep-0 answer, expected, score rule)} for every decision whose rep 0 was answered.
    Repeats of the same input are not independent samples, so each decision is graded once, from rep 0."""
    answered = timed(records)
    return {(it["id"], qid): (it["questions"][qid], a[qid], want, score_rule(it))
            for it in items if (a := answered.get((it["id"], 0))) is not None
            for qid, want in it["expected"].items()}


def is_right(q: dict, answer: dict, expected: Any, rule: str = "round") -> bool:
    return decide(q, answer, rule) == gold(q, expected)


def timed(records: list[dict]) -> dict[tuple[str, int], dict]:
    """Answered timed calls by (item, rep)."""
    return {(r["item"], r["rep"]): r["answers"] for r in records if r["call"] == "timed" and not r["error"]}


def forecast(q: dict, answer: dict) -> list[float] | None:
    """Probability of each option, in the question's order (a noul is [yes, no]); None for a score question, whose
    probabilities engines key differently (criterion text or index)."""
    if q["type"] == "noul":
        return [answer["noul"], 1 - answer["noul"]]
    if q["type"] == "choice":
        return [answer["probabilities"].get(k, 0.0) for k in q["criteria"]]
    return None


def outcome(q: dict, value: Any) -> int:
    """Index of `value` (an expected answer or a decision) among the forecast's options."""
    return (0 if value else 1) if q["type"] == "noul" else list(q["criteria"]).index(value)


def brier(decisions: list[tuple[dict, dict, Any]]) -> tuple[float, int] | None:
    """(mean multi-class Brier score, n) over noul and choice decisions: sum over options of (p - 1[option is
    right])^2, so 0 is perfect and 2 is sure and wrong. None when there are none."""
    scores = [sum((p - (i == outcome(q, gold(q, want)))) ** 2 for i, p in enumerate(f))
              for q, a, want, _ in decisions if (f := forecast(q, a)) is not None]
    return (sum(scores) / len(scores), len(scores)) if scores else None


def ece(points: list[tuple[float, bool]], bins: int = 15) -> float:
    """Expected calibration error (Guo et al. 2017): confidences split into equal-width bins (lo, hi], the gap
    between each bin's accuracy and mean confidence, weighted by the bin's share of the points."""
    groups: dict[int, list[tuple[float, bool]]] = {}
    for conf, right in points:
        groups.setdefault(max(math.ceil(conf * bins) - 1, 0), []).append((conf, right))
    return sum(abs(sum(r for _, r in g) - sum(c for c, _ in g)) for g in groups.values()) / len(points)


def confidence_points(decisions: list[tuple[dict, dict, Any]]) -> list[tuple[float, bool]]:
    """(probability the engine gave its own decision, whether that decision is right) for noul and choice."""
    return [(f[outcome(q, decide(q, a))], is_right(q, a, want))
            for q, a, want, _ in decisions if (f := forecast(q, a)) is not None]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p value from the discordant pairs: b decisions only the first engine got right, c
    only the second. 2 * P(X <= min(b, c)) for X ~ Binomial(b + c, 1/2), capped at 1."""
    n = b + c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


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
            q, rule = it["questions"][qid], score_rule(it)
            n += 1
            changed += any(decide(q, a[qid], rule) != decide(q, first[qid], rule) for a in later)
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

    engine = header["engine"]
    g = graded(items, records)
    items_by_id = {it["id"]: it for it in items}
    for (item, _), answers in timed(records).items():  # every answered timed call, not only rep 0
        it, rule = items_by_id[item], score_rule(items_by_id[item])
        for qid, q in it["questions"].items():
            a = answers[qid]
            if q["type"] == "choice" and a["choice"] not in a["probabilities"]:
                # keyed by option text or index instead: every option would read as probability 0
                raise SystemExit(f"{engine} {item} {qid}: chose {a['choice']!r} but its probabilities are keyed "
                                 f"{sorted(a['probabilities'])}; an engine must key choice probabilities by option key")
            if q["type"] == "score" and rule == "argmax":
                try:
                    score_levels(q, a)
                except SystemExit as e:
                    raise SystemExit(f"{engine} {item} {qid}: {e}") from None
    decisions = list(g.values())
    right, n = sum(is_right(*d) for d in decisions), len(decisions)
    b = brier(decisions)
    calib = confidence_points(decisions)
    has_forecast = any(q["type"] in ("noul", "choice") for it in items for q in it["questions"].values())
    missing = "failed" if has_forecast else "n/a"
    brier_cell = f"{b[0]:.3f} ({b[1]})" if b else missing
    ece_cell = f"{ece(calib):.3f}" if calib else missing
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
    line = (f"| {engine} | {header['model'] or '-'} | {cold} | {first} | {p50} | {p95} | {len(all_ms)} | {acc} "
            f"| {brier_cell} | {ece_cell} | {disagree} | {flip} | {len(errors)} | {worst} |")
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
        "interval. Whether two engines differ is the pairwise McNemar table, not the intervals. `rep-disagree`",
        "counts decisions that a later timed call answered differently.",
        "`first-call p50` covers only each item's first call. Repeats of the same state can be served from an engine's",
        "embedding cache (CLM caches state vectors), so all-call percentiles understate uncached latency.",
        "`order-flip` asks each item once more with every choice's options reversed and counts changed choices.",
        "`Brier` is the mean multi-class Brier score of the graded noul and choice decisions (sum over options, a noul",
        "being yes/no; 0 is perfect, 2 is sure and wrong), with their count. `ECE` is the expected calibration error of",
        "the probability each engine gave its own decision on those, in 15 equal-width bins. Score questions are left",
        "out of both: engines key their probabilities differently.",
        "",
        "| engine | model | cold ms | first-call p50 ms | all-call p50 ms | all-call p95 ms | calls | accuracy "
        "| Brier (n) | ECE | rep-disagree | order-flip | errors | worst memory pressure |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    engines = sorted(runs)
    rows = [row(runs[e][0], common["reps"], items, runs[e][1]) for e in engines]
    lines += [r for r, _ in rows] + [e for _, errs in rows for e in errs]
    if len(engines) > 1:
        lines += pairs(items, {e: runs[e][1] for e in engines})
    for key in ("source", "variant"):
        lines += breakdown(items, {e: runs[e][1] for e in engines}, key)
    return "\n".join(lines) + "\n"


def breakdown(items: list[dict], runs: dict[str, list[dict]], key: str) -> list[str]:
    """Accuracy per value of the items' `meta[key]` (a set that records where each item came from); none without."""
    groups = sorted({it["meta"][key] for it in items if key in it.get("meta", {})})
    if not groups:
        return []
    right = {e: {k: is_right(*v) for k, v in graded(items, records).items()} for e, records in runs.items()}
    lines = ["", f"## By {key}", "", f"| {key} | decisions | " + " | ".join(runs) + " |",
             "|---|---|" + "---|" * len(runs)]
    for g in groups:
        keys = {(it["id"], qid) for it in items if it.get("meta", {}).get(key) == g for qid in it["expected"]}
        cells = []
        for e in runs:
            got = [right[e][k] for k in keys if k in right[e]]
            failed = f", {len(keys) - len(got)} failed" if len(got) < len(keys) else ""
            cells.append(ratio(sum(got), len(got)) + failed if got else "failed")
        lines.append(f"| {g} | {len(keys)} | " + " | ".join(cells) + " |")
    return lines


def pairs(items: list[dict], runs: dict[str, list[dict]]) -> list[str]:
    """McNemar table for every pair of engines, over the decisions both had graded."""
    right = {e: {k: is_right(*v) for k, v in graded(items, records).items()} for e, records in runs.items()}
    lines = [
        "",
        "## Pairwise differences (McNemar exact)",
        "",
        "Over the decisions both engines had graded: `b` = only engine A right, `c` = only engine B right, `p` = two-sided",
        "exact McNemar p value. With many pairs, some p values under 0.05 come by chance.",
        "",
        "| engine A | engine B | both graded | b | c | p |",
        "|---|---|---|---|---|---|",
    ]
    for a, b in itertools.combinations(runs, 2):
        both = right[a].keys() & right[b].keys()
        only_a = sum(right[a][k] and not right[b][k] for k in both)
        only_b = sum(right[b][k] and not right[a][k] for k in both)
        lines.append(f"| {a} | {b} | {len(both)} | {only_a} | {only_b} | {p_cell(mcnemar_exact(only_a, only_b))} |")
    return lines


def p_cell(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


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
