"""Confidence-gated cascade routing, simulated from the raw logs without calling any engine.

    uv run python bench/cascade.py --small kev-4b --big winnow-12b clef --out bench/runs/cascade.md --figures reports/figures

A small engine answers every item; when its confidence on the item is low, the item goes to a big engine instead.
Confidence is the probability the small engine gave its own decision (`forecast` at the decided option); an item
with several questions takes the lowest of them, because one request answers them all. An item whose questions
have no forecast (score questions graded by rounding) is never routed. Both engines' answers and first-call
latencies come from their logs on the same set, each measured alone on an idle machine, so the cascade's latency
is the small engine's plus the big engine's on routed items, with both servers assumed resident.

Sweeping the threshold over every distinct confidence routes the 0..N least confident items and traces
accuracy against the share of items sent to the big engine. Two references bound it: `random` routes the same share
at random (its expected accuracy is linear between the two engines), and the end point is the big engine alone.

- **operating point**: the lowest share whose accuracy is within 1 %p of the big engine alone, with the exact
  McNemar p of the cascade against the big engine on the same decisions. It is picked in-sample, so it is checked
  split-half: the threshold picked on even-indexed items is applied to odd ones and back.
- **peak**: the most accurate point of the curve, with McNemar against the big engine alone. Also in-sample; its
  split-half check applies the peak threshold of each half to the other.
- **gain over random**: the area between the cascade curve and random routing over the routed share (0 to 1), in
  %p; ties in confidence are routed together, so the curve is linear across a tie group.
- **efficiency**: that gain over the gain of an oracle router, which sends first the items where only the big
  engine is right. 1 = confidence ranks items as well as hindsight, 0 = no better than random. It does not grow with
  the accuracy gap between the engines, as the gain does, so engines of any strength compare on it.
- **fixable errors**: the share of the small engine's wrong decisions that the big engine gets right, the most
  routing can recover.
- **AUROC**: how well the small engine's confidence ranks its own right decisions above its wrong ones (0.5 = no
  better than chance); ECE is how far its confidence is from its accuracy. The calibration section correlates both
  with efficiency over every small engine routed to one big engine, per set and pooled.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import score

SETS = ("questions", "questions_ko", "questions_banking77", "transfer-v4", "typed-decisions", "nsmc", "klue-ynat",
        "clinc-oos")
RUNS = Path(__file__).with_name("runs")
WITHIN = 0.01


@dataclass
class Item:
    right: list[bool]
    conf: float  # math.inf when never routed
    ms: float


@dataclass
class Point:
    k: int  # items routed
    rate: float
    acc: float
    ms: float
    conf: float  # the highest confidence routed; -inf at k = 0


def view(items: list[dict], records: list[dict]) -> dict[str, Item]:
    """{item: Item} for items whose rep-0 call answered."""
    g = score.graded(items, records)
    ms = {r["item"]: r["ms"] for r in records if r["call"] == "timed" and r["rep"] == 0 and not r["error"]}
    out: dict[str, Item] = {}
    for it in items:
        keys = [(it["id"], qid) for qid in it["expected"]]
        if not all(k in g for k in keys):
            continue
        confs = [f[score.outcome(q, score.decide(q, a, rule))] for q, a, _, rule in (g[k] for k in keys)
                 if (f := score.forecast(q, a, rule)) is not None]
        out[it["id"]] = Item([score.is_right(*g[k]) for k in keys], min(confs) if confs else math.inf, ms[it["id"]])
    return out


def curve(small: dict[str, Item], big: dict[str, Item], ids: list[str]) -> list[Point]:
    """Cascade points routing the k least confident of ids, at every boundary between distinct confidences."""
    order = sorted((i for i in ids if small[i].conf != math.inf), key=lambda i: small[i].conf)
    d = sum(len(small[i].right) for i in ids)
    acc = sum(sum(small[i].right) for i in ids)
    ms = sum(small[i].ms for i in ids)
    n = len(ids)
    points = [Point(0, 0.0, acc / d, ms / n, -math.inf)]
    for j, i in enumerate(order):
        acc += sum(big[i].right) - sum(small[i].right)
        ms += big[i].ms
        if j + 1 == len(order) or small[order[j + 1]].conf != small[i].conf:
            points.append(Point(j + 1, (j + 1) / n, acc / d, ms / n, small[i].conf))
    return points


def accuracy(engine: dict[str, Item], ids: list[str]) -> float:
    return sum(sum(engine[i].right) for i in ids) / sum(len(engine[i].right) for i in ids)


def operating(points: list[Point], big_acc: float) -> Point:
    return next(p for p in points if p.acc >= big_acc - WITHIN - 1e-12)


def routed(small: dict[str, Item], ids: list[str], conf: float) -> set[str]:
    return {i for i in ids if small[i].conf <= conf}


def mcnemar_vs_big(small: dict[str, Item], big: dict[str, Item], ids: list[str], sent: set[str]) -> tuple[int, int, float]:
    """(cascade only right, big only right, p) over the decisions; routed items agree with the big engine."""
    b = c = 0
    for i in ids:
        if i in sent:
            continue
        for s, g in zip(small[i].right, big[i].right):
            b += s and not g
            c += g and not s
    return b, c, score.mcnemar_exact(b, c)


def peak(points: list[Point], big_acc: float) -> Point:
    return max(points, key=lambda p: (p.acc, -p.k))


def split_half(small: dict[str, Item], big: dict[str, Item], ids: list[str],
               pick_point=operating) -> list[tuple[float, float]]:
    """For each half: (share routed, accuracy minus the big engine's) on the other half, at the threshold the first
    half's point (`operating` or `peak`) gives."""
    halves = [ids[0::2], ids[1::2]]
    out = []
    for pick, test in (halves, halves[::-1]):
        conf = pick_point(curve(small, big, pick), accuracy(big, pick)).conf
        sent = routed(small, test, conf)
        acc = sum(sum((big if i in sent else small)[i].right) for i in test) / sum(len(small[i].right) for i in test)
        out.append((len(sent) / len(test), acc - accuracy(big, test)))
    return out


def auroc(points: list[tuple[float, bool]]) -> float | None:
    """P(a right decision's confidence > a wrong one's), ties counted half (Mann-Whitney)."""
    pos = sorted(c for c, r in points if r)
    neg = [c for c, r in points if not r]
    if not pos or not neg:
        return None
    import bisect

    wins = sum(bisect.bisect_left(pos, c) for c in neg)  # positives below each negative
    ties = sum(bisect.bisect_right(pos, c) - bisect.bisect_left(pos, c) for c in neg)
    return 1 - (wins + ties / 2) / (len(pos) * len(neg))


def calibration(items: list[dict], records: list[dict]) -> tuple[float | None, float | None]:
    """(ECE, AUROC) of an engine's confidence in its own graded decisions."""
    pts = score.confidence_points(list(score.graded(items, records).values()))
    return (score.ece(pts) if pts else None), auroc(pts)


def area(points: list[Point]) -> float:
    """Trapezoid area under accuracy over the routed share."""
    return sum((b.rate - a.rate) * (a.acc + b.acc) / 2 for a, b in zip(points, points[1:]))


def oracle(small: dict[str, Item], big: dict[str, Item], ids: list[str]) -> list[Point]:
    """The best curve hindsight allows: items ordered by how many more decisions the big engine gets right."""
    order = sorted(ids, key=lambda i: sum(small[i].right) - sum(big[i].right))
    d, n = sum(len(small[i].right) for i in ids), len(ids)
    acc = sum(sum(small[i].right) for i in ids)
    points = [Point(0, 0.0, acc / d, 0.0, -math.inf)]
    for j, i in enumerate(order):
        acc += sum(big[i].right) - sum(small[i].right)
        points.append(Point(j + 1, (j + 1) / n, acc / d, 0.0, -math.inf))
    return points


@dataclass
class Result:
    set: str
    small: str
    big: str
    n: int
    small_acc: float
    big_acc: float
    small_ms: float
    big_ms: float
    points: list[Point]
    op: Point
    op_test: tuple[int, int, float]
    small_vs_big: tuple[int, int, float]
    peak: Point
    peak_test: tuple[int, int, float]
    peak_halves: list[tuple[float, float]]
    halves: list[tuple[float, float]]
    gain: float
    efficiency: float | None
    fixable: tuple[int, int]
    ece: float | None
    auroc: float | None


def run(set_name: str, small_e: str, big_e: str) -> Result | None:
    _, items, runs = score.load_dir(RUNS / set_name)
    if small_e not in runs or big_e not in runs:
        return None
    small, big = view(items, runs[small_e][1]), view(items, runs[big_e][1])
    ids = sorted(small.keys() & big.keys())
    if not ids:
        return None
    pts = curve(small, big, ids)
    sa, ba = accuracy(small, ids), accuracy(big, ids)
    op, top = operating(pts, ba), peak(pts, ba)
    ece, roc = calibration(items, runs[small_e][1])
    random_area = (sa + ba) / 2
    gain, best = area(pts) - random_area, area(oracle(small, big, ids)) - random_area
    pairs = [(s, b) for i in ids for s, b in zip(small[i].right, big[i].right) if not s]
    return Result(set_name, small_e, big_e, len(ids), sa, ba, sum(small[i].ms for i in ids) / len(ids),
                  sum(big[i].ms for i in ids) / len(ids), pts, op,
                  mcnemar_vs_big(small, big, ids, routed(small, ids, op.conf)), mcnemar_vs_big(small, big, ids, set()),
                  top, mcnemar_vs_big(small, big, ids, routed(small, ids, top.conf)), split_half(small, big, ids, peak),
                  split_half(small, big, ids),
                  gain, gain / best if best >= 0.005 else None, (sum(b for _, b in pairs), len(pairs)), ece, roc)


def spearman(xs: list[float], ys: list[float], perms: int = 10000, seed: int = 0) -> tuple[float, float]:
    """(rho, two-sided permutation p)."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    def rho(a: list[float], b: list[float]) -> float:
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den if den else 0.0

    rx, ry = ranks(xs), ranks(ys)
    r0 = rho(rx, ry)
    rng = random.Random(seed)
    hits = 0
    for _ in range(perms):
        shuffled = ry[:]
        rng.shuffle(shuffled)
        hits += abs(rho(rx, shuffled)) >= abs(r0) - 1e-12
    return r0, (hits + 1) / (perms + 1)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def pp(x: float) -> str:
    return f"{x * 100:+.1f}"


def table(results: list[Result]) -> list[str]:
    lines = ["| set | items | small acc | big acc | small vs big (b / c, p) | operating point: calls | acc (vs big) | McNemar vs big | "
             "mean ms (small / big / cascade) | split-half calls, acc vs big | peak: calls, acc vs big, McNemar | "
             "peak split-half | gain over random | efficiency | fixable errors | "
             "small ECE | small AUROC |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        b, c, p = r.op_test
        halves = "; ".join(f"{pct(rate)}, {pp(gap)}" for rate, gap in r.halves)
        peak_halves = "; ".join(f"{pct(rate)}, {pp(gap)}" for rate, gap in r.peak_halves)
        pb, pc, pq = r.peak_test
        sb, sc, sp = r.small_vs_big
        lines.append(f"| {r.set} | {r.n} | {pct(r.small_acc)} | {pct(r.big_acc)} | {sb} / {sc}, p = {score.p_cell(sp)} | "
                     f"{pct(r.op.rate)} | "
                     f"{pct(r.op.acc)} ({pp(r.op.acc - r.big_acc)}) | {b} / {c}, p = {score.p_cell(p)} | "
                     f"{r.small_ms:.0f} / {r.big_ms:.0f} / {r.op.ms:.0f} | {halves} | "
                     f"{pct(r.peak.rate)}, {pp(r.peak.acc - r.big_acc)}, {pb} / {pc} p = {score.p_cell(pq)} | {peak_halves} | "
                     f"{pp(r.gain)} | "
                     f"{'n/a' if r.efficiency is None else f'{r.efficiency:.2f}'} | {score.ratio(*r.fixable)} | "
                     f"{'n/a' if r.ece is None else f'{r.ece:.3f}'} | {'n/a' if r.auroc is None else f'{r.auroc:.3f}'} |")
    return lines


# --- figures: static SVG, reference palette (dataviz skill), light surface --------------------------------------

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3", "#8a8984"
SERIES = ("#2a78d6", "#eb6834")
FONT = "font-family='-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif'"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def curves_svg(groups: list[list[Result]], title: str) -> str:
    """Small multiples, one panel per set: accuracy against the share sent to the big engine, one line per big
    engine (fixed slot order), dashed random routing per big engine, a dot at each operating point."""
    cols, pw, ph, ml, mt, gap = 4, 230, 170, 44, 34, 26
    sets = [rs[0].set for rs in zip(*groups)]
    rows = math.ceil(len(sets) / cols)
    w, h = cols * (pw + ml) + 20, rows * (ph + mt + gap) + 100
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT}>",
           f"<rect width='{w}' height='{h}' fill='{SURFACE}'/>",
           f"<text x='16' y='24' font-size='15' font-weight='600' fill='{INK}'>{esc(title)}</text>"]
    lx = 16
    for k, rs in enumerate(groups):
        out.append(f"<line x1='{lx}' y1='44' x2='{lx + 22}' y2='44' stroke='{SERIES[k]}' stroke-width='2'/>")
        label = f"cascade to {rs[0].big}"
        out.append(f"<text x='{lx + 28}' y='48' font-size='12' fill='{INK2}'>{esc(label)}</text>")
        lx += 40 + 7 * len(label)
    out.append(f"<line x1='{lx}' y1='44' x2='{lx + 22}' y2='44' stroke='{MUTED}' stroke-width='1.5' "
               f"stroke-dasharray='4 3'/><text x='{lx + 28}' y='48' font-size='12' fill='{INK2}'>random routing</text>")
    for idx, per_set in enumerate(zip(*groups)):
        x0 = 16 + (idx % cols) * (pw + ml) + ml
        y0 = 70 + (idx // cols) * (ph + mt + gap) + mt
        accs = [p.acc for r in per_set for p in r.points]
        lo = math.floor(min(accs) * 20) / 20
        hi = math.ceil(max(accs) * 20) / 20 if max(accs) < 1 else 1.0
        if hi - lo < 0.1:
            lo = max(0.0, hi - 0.1)
        X = lambda r: x0 + r * pw  # noqa: E731
        Y = lambda a: y0 + ph - (a - lo) / (hi - lo) * ph  # noqa: E731
        out.append(f"<text x='{x0}' y='{y0 - 10}' font-size='13' font-weight='600' fill='{INK}'>"
                   f"{esc(per_set[0].set)} <tspan font-weight='400' fill='{INK2}'>({per_set[0].n} items)</tspan></text>")
        step = 0.05 if hi - lo >= 0.15 else 0.025
        for s in range(round((hi - lo) / step) + 1):
            a = lo + s * step
            out.append(f"<line x1='{x0}' y1='{Y(a):.1f}' x2='{x0 + pw}' y2='{Y(a):.1f}' stroke='{GRID}' stroke-width='1'/>")
            out.append(f"<text x='{x0 - 6}' y='{Y(a) + 4:.1f}' font-size='10' fill='{INK2}' text-anchor='end'>"
                       f"{a * 100:.{0 if step == 0.05 else 1}f}%</text>")
        for r in (0, 0.5, 1):
            out.append(f"<text x='{X(r):.1f}' y='{y0 + ph + 14}' font-size='10' fill='{INK2}' text-anchor='middle'>"
                       f"{r * 100:.0f}%</text>")
        out.append(f"<line x1='{x0}' y1='{y0 + ph}' x2='{x0 + pw}' y2='{y0 + ph}' stroke='{INK2}' stroke-width='1'/>")
        for k, r in enumerate(per_set):
            out.append(f"<line x1='{X(0):.1f}' y1='{Y(r.small_acc):.1f}' x2='{X(1):.1f}' y2='{Y(r.big_acc):.1f}' "
                       f"stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='4 3'/>")
            path = " ".join(f"{'M' if j == 0 else 'L'}{X(p.rate):.1f},{Y(p.acc):.1f}" for j, p in enumerate(r.points))
            out.append(f"<path d='{path}' fill='none' stroke='{SERIES[k]}' stroke-width='2' stroke-linejoin='round'/>")
        for k, r in enumerate(per_set):  # dots last, ringed in the surface so they sit over every line
            tip = (f"{r.small} to {r.big}, {r.set}: {pct(r.op.rate)} calls, {pct(r.op.acc)} accuracy "
                   f"(big alone {pct(r.big_acc)}), {r.op.ms:.0f} ms mean")
            out.append(f"<circle cx='{X(r.op.rate):.1f}' cy='{Y(r.op.acc):.1f}' r='4.5' fill='{SERIES[k]}' "
                       f"stroke='{SURFACE}' stroke-width='2'><title>{esc(tip)}</title></circle>")
    out.append(f"<text x='16' y='{h - 12}' font-size='11' fill='{INK2}'>x: share of items sent to the big engine. "
               f"y: accuracy (each panel its own range). Dot: lowest share within 1 %p of the big engine alone.</text>")
    return "\n".join(out + ["</svg>"]) + "\n"


def scatter_svg(points: list[tuple[str, float, float, float, float]], title: str, rho_ece: tuple[float, float],
                rho_roc: tuple[float, float]) -> str:
    """Two panels over (label, ECE, AUROC, efficiency): efficiency against ECE, and against AUROC. One series."""
    pw, ph, ml, mt = 300, 220, 50, 40
    w, h = 2 * (pw + ml) + 60, ph + mt + 110
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT}>",
           f"<rect width='{w}' height='{h}' fill='{SURFACE}'/>",
           f"<text x='16' y='24' font-size='15' font-weight='600' fill='{INK}'>{esc(title)}</text>"]
    gains = [p[3] for p in points]
    glo, ghi = min(-0.25, math.floor(min(gains) * 4) / 4), max(1.0, math.ceil(max(gains) * 4) / 4)
    for panel, (name, xi, rh) in enumerate((("small engine ECE", 1, rho_ece), ("small engine AUROC", 2, rho_roc))):
        xs = [p[xi] for p in points]
        xlo, xhi = (0.0, math.ceil(max(xs) * 10) / 10) if xi == 1 else (math.floor(min(xs) * 10) / 10, 1.0)
        x0, y0 = 16 + ml + panel * (pw + ml + 30), 50 + mt
        X = lambda v: x0 + (v - xlo) / (xhi - xlo) * pw  # noqa: E731
        Y = lambda g: y0 + ph - (g - glo) / (ghi - glo) * ph  # noqa: E731
        out.append(f"<text x='{x0}' y='{y0 - 14}' font-size='13' font-weight='600' fill='{INK}'>{esc(name)} "
                   f"<tspan font-weight='400' fill='{INK2}'>(Spearman {rh[0]:+.2f}, p = {score.p_cell(rh[1])})</tspan></text>")
        for s in range(round((ghi - glo) / 0.25) + 1):
            g = glo + s * 0.25
            out.append(f"<line x1='{x0}' y1='{Y(g):.1f}' x2='{x0 + pw}' y2='{Y(g):.1f}' stroke='{GRID}' stroke-width='1'/>")
            out.append(f"<text x='{x0 - 6}' y='{Y(g) + 4:.1f}' font-size='10' fill='{INK2}' text-anchor='end'>"
                       f"{g:.2f}</text>")
        xstep = 0.1 if xhi - xlo <= 0.7 + 1e-9 else 0.2
        for s in range(round((xhi - xlo) / xstep) + 1):
            v = xlo + s * xstep
            out.append(f"<text x='{X(v):.1f}' y='{y0 + ph + 14}' font-size='10' fill='{INK2}' text-anchor='middle'>"
                       f"{v:.1f}</text>")
        if glo < 0 < ghi:
            out.append(f"<line x1='{x0}' y1='{Y(0):.1f}' x2='{x0 + pw}' y2='{Y(0):.1f}' stroke='{INK2}' stroke-width='1'/>")
        for label, e, roc, g, _ in points:
            v = e if xi == 1 else roc
            out.append(f"<circle cx='{X(v):.1f}' cy='{Y(g):.1f}' r='4' fill='{SERIES[0]}' fill-opacity='0.75' "
                       f"stroke='{SURFACE}' stroke-width='1.5'><title>{esc(label)}: ECE {e:.3f}, AUROC {roc:.3f}, "
                       f"efficiency {g:.2f}</title></circle>")
    out.append(f"<text x='16' y='{h - 30}' font-size='11' fill='{INK2}'>y: routing efficiency, gain over random routing "
               f"as a share of an oracle router's (1 = hindsight, 0 = random). One dot per small engine and set.</text>")
    return "\n".join(out + ["</svg>"]) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="kev-4b")
    ap.add_argument("--big", nargs="+", default=["winnow-12b", "clef"])
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--out", type=Path, help="also write the tables here")
    ap.add_argument("--figures", type=Path, help="write cascade-<small>.svg and cascade-calibration.svg here")
    ap.add_argument("--scatter-big", default="clef",
                    help="the big engine the calibration figure routes every other engine to")
    args = ap.parse_args()
    lines = ["# Cascade routing", "", "Generated by `bench/cascade.py` from the raw logs; see its docstring for the "
             "method.", ""]
    groups = []
    for big in args.big:
        rs = [r for s in args.sets if (r := run(s, args.small, big))]
        groups.append(rs)
        lines += [f"## {args.small} to {big}", ""] + table(rs) + [""]
    pts = []
    for s in args.sets:
        _, _, runs = score.load_dir(RUNS / s)
        for e in sorted(runs):
            if e == args.scatter_big:
                continue
            r = run(s, e, args.scatter_big)
            if r and r.ece is not None and r.auroc is not None and r.efficiency is not None:
                pts.append((f"{e} / {s}", r.ece, r.auroc, r.efficiency, r.fixable[0] / r.fixable[1]))
    rho_e = spearman([p[1] for p in pts], [p[3] for p in pts])
    rho_r = spearman([p[2] for p in pts], [p[3] for p in pts])
    rho_f = spearman([p[4] for p in pts], [p[3] for p in pts])
    lines += [f"## Does calibration predict routing? (every engine to {args.scatter_big}, {len(pts)} engine x set)", "",
              "Engine x set pairs where an oracle router gains less than 0.5 %p over random are left out (no headroom).",
              f"Pooled Spearman of efficiency with the small engine's ECE: {rho_e[0]:+.2f} (permutation p = "
              f"{score.p_cell(rho_e[1])}); with its AUROC: {rho_r[0]:+.2f} (p = {score.p_cell(rho_r[1])}); with the share of "
              f"its errors the big engine fixes: {rho_f[0]:+.2f} (p = {score.p_cell(rho_f[1])}).", "",
              "| set | engines | Spearman with ECE (p) | with AUROC (p) | with fixable errors (p) |", "|---|---|---|---|---|"]
    for s in args.sets:
        sub = [p for p in pts if p[0].endswith(f" / {s}")]
        if len(sub) >= 5:
            cells = [spearman([p[k] for p in sub], [p[3] for p in sub]) for k in (1, 2, 4)]
            lines.append(f"| {s} | {len(sub)} | " + " | ".join(f"{r:+.2f} ({score.p_cell(q)})" for r, q in cells) + " |")
    lines += ["", "| small engine / set | ECE | AUROC | fixable errors | efficiency |", "|---|---|---|---|---|"]
    lines += [f"| {lab} | {e:.3f} | {roc:.3f} | {f:.0%} | {g:.2f} |" for lab, e, roc, g, f in sorted(pts, key=lambda p: -p[3])]
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        (args.figures / f"cascade-{args.small}.svg").write_text(
            curves_svg(groups, f"{args.small} cascade: accuracy against calls to the big engine"))
        (args.figures / "cascade-calibration.svg").write_text(
            scatter_svg(pts, f"Does the small engine's calibration or ranking predict routing efficiency? "
                             f"(big engine: {args.scatter_big})", rho_e, rho_r))
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
