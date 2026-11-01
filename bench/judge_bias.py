"""Judge-bias metrics from the raw logs of the judge sets (bench/make_sets.py judge-*), without calling any engine.

    uv run python bench/judge_bias.py bench/runs/judge-mtbench bench/runs/judge-ko bench/runs/judge-length
    uv run python bench/judge_bias.py bench/runs/judge-* --out bench/runs/judge-bias.md

Every pair is asked in both orders (meta `variant` ab / ba), so each engine's rep-0 answers give, per pair:
- **position flip**: the two orders picked different content, so the pick followed the position. `A both` is the
  share of flipped pairs where the engine picked A both times (a first-position preference).
- **human agreement**: on the human-labelled sets, items whose pick is the human winner, and Cohen's kappa per
  pair, the pair's verdict being the content both orders picked, or `tie` when they disagree.
- **longer pick**: on the pairs whose two orders agree, how often the engine's pick is the longer response, next to
  how often the human winner is on the same pairs.
- **padded pick**: on judge-length, how often the engine picks an answer's padded copy over the answer itself.
- **x pick**: on judge-self (bench/make_judge_self.py), how often the engine picks the answer of model x's family
  over model y's. Self-preference shows as judges of x's family (FAMILY) picking x more often than other judges on
  the same pairs, tested pair by pair with McNemar.

Intervals: Wilson 95% for proportions, a pair bootstrap (2,000 resamples, seed 0) for kappa and the longer-pick gap.
Engines are compared pair by pair with exact McNemar on per-pair human agreement, so the two orders of a pair are
never counted as two samples.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import score

BOOT = 2000
# Base-model family of each judge engine (README engine list), for self-preference on judge-self.
FAMILY = {"clm": "qwen", "anyjev-raw": "qwen", "anyjev-l0": "qwen", "kev-0.8b": "qwen", "kev-4b": "qwen",
          "kev-9b": "qwen", "jeff": "qwen", "jeff-0.8b": "qwen", "decider-2b": "qwen", "decider-4b": "qwen",
          "decision-eos": "qwen", "jevk5": "qwen", "clef-flash": "qwen", "clef": "qwen",
          "winnow": "gemma", "winnow-12b": "gemma", "von": "modernbert"}


def verdicts(items: list[dict], records: list[dict]) -> dict[str, dict]:
    """{pair: {"ab": pick, "ba": pick}} over pairs with both rep-0 answers; a pick is "x" or "y", the content the
    engine chose (x is response A in the ab order). Pairs missing an order are left out."""
    answered = score.timed(records)
    out: dict[str, dict] = {}
    for it in items:
        a = answered.get((it["id"], 0))
        if a is None:
            continue
        variant = it["meta"]["variant"]
        chose_a = a["better"]["choice"] == "A"
        out.setdefault(it["meta"]["pair"], {})[variant] = "x" if chose_a == (variant == "ab") else "y"
    return {p: v for p, v in out.items() if len(v) == 2}


def pair_info(items: list[dict]) -> dict[str, dict]:
    """{pair: {"human": "x"/"y", "longer": "x"/"y"/None}} from the ab item of each pair."""
    info = {}
    for it in items:
        if it["meta"]["variant"] != "ab":
            continue
        la, lb = it["meta"]["lengths"]
        info[it["meta"]["pair"]] = {"human": "x" if it["expected"]["better"] == "A" else "y",
                                    "longer": "x" if la > lb else "y" if lb > la else None}
    return info


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa of (judge, human) labels; None when chance agreement is total."""
    n = len(pairs)
    cats = {c for p in pairs for c in p}
    po = sum(j == h for j, h in pairs) / n
    pe = sum(sum(j == c for j, _ in pairs) / n * sum(h == c for _, h in pairs) / n for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def boot(xs: list, stat, seed: int = 0) -> tuple[float, float] | None:
    """Percentile 95% interval of stat over BOOT resamples of xs; resamples where stat is None are skipped."""
    rng = random.Random(seed)
    vals = sorted(v for _ in range(BOOT) if (v := stat([rng.choice(xs) for _ in xs])) is not None)
    if not vals:
        return None
    return vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]


def prop(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = score.wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}, {lo * 100:.0f}-{hi * 100:.0f})"


def interval(x: float | None, ci: tuple[float, float] | None, fmt: str = "{:.2f}") -> str:
    if x is None:
        return "n/a"
    return fmt.format(x) + (f" ({fmt.format(ci[0])} to {fmt.format(ci[1])})" if ci else "")


def kind(items: list[dict]) -> str:
    """`length`, `self` or `human`, from the sources make_sets.py and make_judge_self.py give the items."""
    for k in ("length", "self"):
        if all(it["meta"]["source"].endswith(f"-{k}") for it in items):
            return k
    return "human"


def engine_row(items: list[dict], records: list[dict]) -> dict[str, str]:
    v, info = verdicts(items, records), pair_info(items)
    flipped = [p for p, d in v.items() if d["ab"] != d["ba"]]
    # A flipped pair picked A both times when the ab pick is x (A in ab) and the ba pick is y (A in ba).
    a_both = sum(v[p]["ab"] == "x" for p in flipped)
    row = {"pairs": str(len(v)), "position flip": prop(len(flipped), len(v)),
           "A both": prop(a_both, len(flipped))}
    picks = [d[o] for d in v.values() for o in ("ab", "ba")]
    if kind(items) == "length":
        # x is the original, y the padded copy, in every pair.
        row["padded pick"] = prop(sum(p == "y" for p in picks), len(picks))
        row["padded in both orders"] = prop(sum(d["ab"] == d["ba"] == "y" for d in v.values()), len(v))
        return row
    if kind(items) == "self":
        row["x pick"] = prop(sum(p == "x" for p in picks), len(picks))
        row["x in both orders"] = prop(sum(d["ab"] == d["ba"] == "x" for d in v.values()), len(v))
        return row
    answered = score.timed(records)
    right = [answered[(it["id"], 0)]["better"]["choice"] == it["expected"]["better"]
             for it in items if (it["id"], 0) in answered]
    row["human agreement"] = prop(sum(right), len(right))
    labelled = [(d["ab"] if d["ab"] == d["ba"] else "tie", info[p]["human"]) for p, d in v.items()]
    row["kappa"] = interval(kappa(labelled) if labelled else None, boot(labelled, kappa) if labelled else None)
    consistent = [(d["ab"], info[p]) for p, d in v.items() if d["ab"] == d["ba"] and info[p]["longer"]]
    judge_long = sum(pick == i["longer"] for pick, i in consistent)
    human_long = sum(i["human"] == i["longer"] for _, i in consistent)
    row["longer pick (judge)"] = prop(judge_long, len(consistent))
    row["longer pick (human)"] = prop(human_long, len(consistent))

    def gap(xs: list) -> float | None:
        return (sum(p == i["longer"] for p, i in xs) - sum(i["human"] == i["longer"] for _, i in xs)) / len(xs) \
            if xs else None

    row["longer gap"] = interval(gap(consistent), boot(consistent, gap) if consistent else None, "{:+.2f}")
    return row


def mcnemar_lines(items: list[dict], runs: dict[str, tuple[dict, list[dict]]]) -> list[str]:
    """Engine by engine, per pair: on a human set, whether both orders pick the human winner; on judge-self, whether
    both orders pick x, between judges of different families only."""
    info = pair_info(items)
    agree = {e: {p: d["ab"] == d["ba"] == info[p]["human"] for p, d in verdicts(items, recs).items()}
             for e, (_, recs) in runs.items()}
    what = "pick x" if kind(items) == "self" else "pick the human winner"
    lines = ["", f"Per pair, both orders {what}; exact McNemar:", "",
             "| engine A | engine B | pairs | only A | only B | p |", "|---|---|---|---|---|---|"]
    for a, b in itertools.combinations(sorted(runs), 2):
        if kind(items) == "self" and FAMILY.get(a) == FAMILY.get(b):
            continue
        both = agree[a].keys() & agree[b].keys()
        oa = sum(agree[a][p] and not agree[b][p] for p in both)
        ob = sum(agree[b][p] and not agree[a][p] for p in both)
        lines.append(f"| {a} | {b} | {len(both)} | {oa} | {ob} | {score.p_cell(score.mcnemar_exact(oa, ob))} |")
    return lines


def section(d: Path) -> list[str]:
    common, items, runs = score.load_dir(d)
    rows = {e: engine_row(items, recs) for e, (_, recs) in sorted(runs.items())}
    cols = list(next(iter(rows.values())))
    lines = [f"## {common['set']}", "", f"sha256 `{common['set_sha256'][:12]}`, {len(items) // 2} pairs, "
             f"{common['reps']} timed call(s) per item; only rep 0 is used.", "",
             "| engine | family | model | measured | host | " + " | ".join(cols) + " |",
             "|---|---|---|---|---|" + "---|" * len(cols)]
    for e, r in rows.items():
        h = runs[e][0]
        lines.append(f"| {e} | {FAMILY.get(e, '-')} | {h['model'] or '-'} | {h['started'][:10]} | {h['host']} | "
                     + " | ".join(r[c] for c in cols) + " |")
    if kind(items) != "length" and len(runs) > 1:
        lines += mcnemar_lines(items, runs)
    return lines + [""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+", help="bench/runs/judge-<set>")
    ap.add_argument("--out", type=Path, help="also write the tables here")
    args = ap.parse_args()
    text = "\n".join(["# Judge bias", "", "Generated by `bench/judge_bias.py` from the raw logs; see its docstring for "
                      "the metrics.", ""] + [line for d in args.dirs for line in section(d)])
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
