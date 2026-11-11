"""The README's two charts, drawn as plain SVG from the committed `bench/runs/<set>/report.md` tables.

    uv run python bench/charts.py            # rewrite docs/charts/*.svg
    uv run python bench/charts.py --check    # fail if a chart is out of date with the reports

Standard library only, and deterministic, so the SVGs diff cleanly and the offline tests can check them.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "bench" / "runs"
OUT = ROOT / "docs" / "charts"

# (run directory, column label), in the README's table order.
SETS = (
    ("questions", "English 30"),
    ("questions_ko", "Korean 30"),
    ("questions_banking77", "BANKING77-20"),
    ("transfer-v4", "transfer-v4"),
    ("typed-decisions", "typed-decisions"),
    ("nsmc", "NSMC"),
    ("klue-ynat", "KLUE-YNAT"),
    ("clinc-oos", "clinc-oos"),
)

# Engine id (log file stem) -> the name the README uses.
NAMES = {
    "anyjev-l0": "AnyJev L0", "anyjev-raw": "AnyJev raw", "clef-flash": "Clef-flash", "clef": "Clef",
    "clm": "CLM-8B", "coba-1b": "coba-1b", "coba-400m": "coba-400m", "d1-omni": "d1-omni",
    "decider-2b": "Decider-2B", "decider-4b": "Decider-4B", "decision-eos": "Decision-Eos", "drex-1.5": "Drex v1.5",
    "jeff-0.8b": "Jeff-Qwen3.5-0.8B", "jeff": "Jeff-Qwen3.5-2B", "jevk5": "JevK5", "julia-1": "Julia-1",
    "kev-0.8b": "Kev-0.8B", "kev-4b": "Kev-4B", "kev-9b": "Kev-9B", "lev": "lev", "nimble-v3": "nimble-v3",
    "ollaya-td": "Ollaya-td", "ollaya": "Ollaya", "onejev-4b": "OneJev-4B", "opendecider-td": "OpenDecider-td",
    "openjev": "OpenJev", "pplx-decider": "pplx-decider", "startlux-4b": "StartLux-4B", "sura-0.8b": "Sura-0.8B",
    "sura-2b": "Sura-2B", "von": "Von", "winnow-12b": "Winnow-12B", "winnow": "Winnow-E4B",
}

# The README's `†` cells: the engine trained on a train split of data in that set (README § Results footnote).
IN_DISTRIBUTION = {
    ("kev-9b", "questions_banking77"), ("kev-4b", "questions_banking77"), ("kev-0.8b", "questions_banking77"),
    ("jeff", "transfer-v4"), ("ollaya-td", "typed-decisions"),
    ("opendecider-td", "transfer-v4"), ("opendecider-td", "typed-decisions"), ("opendecider-td", "clinc-oos"),
    ("lev", "questions_banking77"), ("lev", "clinc-oos"),
    ("onejev-4b", "questions_banking77"), ("onejev-4b", "clinc-oos"),
    ("von", "questions_banking77"), ("von", "transfer-v4"),
}

# Engines the README notes ran on CPU here, so their latency does not compare with the Metal rows.
CPU = {"decider-2b", "decider-4b", "pplx-decider", "nimble-v3", "ollaya-td"}

FONT = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""
VIRIDIS = ((0x44, 0x01, 0x54), (0x3b, 0x52, 0x8b), (0x21, 0x91, 0x8c), (0x5e, 0xc9, 0x62), (0xfd, 0xe7, 0x25))
LOW, HIGH = 40.0, 95.0  # accuracy (%) mapped to the two ends of the colour scale


@dataclass
class Cell:
    right: int
    n: int
    p50: float

    @property
    def acc(self) -> float:
        return 100 * self.right / self.n


def engine_rows(report: Path) -> dict[str, Cell | None]:
    """{engine: first-call cell, or None when the engine failed the set} from a report's first table."""
    lines = report.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| engine | model | cold ms |"))
    head = [c.strip() for c in lines[start].strip("|").split("|")]
    col_acc, col_p50 = head.index("accuracy"), head.index("first-call p50 ms")
    rows: dict[str, Cell | None] = {}
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        m = re.match(r"(\d+)/(\d+) ", cells[col_acc])
        rows[cells[0]] = Cell(int(m[1]), int(m[2]), float(cells[col_p50])) if m else None
    return rows


def load() -> dict[str, dict[str, Cell | None]]:
    """{engine: {set: cell}}; a set the engine was never measured on is absent."""
    table: dict[str, dict[str, Cell | None]] = {}
    for d, _ in SETS:
        for engine, cell in engine_rows(RUNS / d / "report.md").items():
            if engine not in NAMES:
                raise SystemExit(f"bench/runs/{d}/report.md: engine {engine!r} has no README name in NAMES")
            table.setdefault(engine, {})[d] = cell
    return table


def complete(cells: dict[str, Cell | None]) -> bool:
    return all(cells.get(d) is not None for d, _ in SETS)


def mean_acc(cells: dict[str, Cell | None]) -> float:
    return statistics.fmean(cells[d].acc for d, _ in SETS)


def ranked(table: dict[str, dict[str, Cell | None]]) -> list[str]:
    """Engines measured on every set by mean accuracy, best first, then the rest by name."""
    full = sorted((e for e in table if complete(table[e])), key=lambda e: (-mean_acc(table[e]), NAMES[e]))
    return full + sorted((e for e in table if not complete(table[e])), key=lambda e: NAMES[e])


def colour(acc: float) -> tuple[str, str]:
    """(fill, text colour) for an accuracy on the viridis scale."""
    t = min(max((acc - LOW) / (HIGH - LOW), 0.0), 1.0) * (len(VIRIDIS) - 1)
    i = min(int(t), len(VIRIDIS) - 2)
    f = t - i
    rgb = [round(a + (b - a) * f) for a, b in zip(VIRIDIS[i], VIRIDIS[i + 1])]
    return "#{:02x}{:02x}{:02x}".format(*rgb), ("#ffffff" if t < 2.4 else "#111111")


def text(x: float, y: float, s: str, *, size: int = 12, anchor: str = "start", weight: str = "normal",
         fill: str = "#111111") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{fill}">{escape(s)}</text>')


def svg(width: int, height: int, body: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" {FONT}>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        *body,
        "</svg>",
        "",
    ])


def heatmap(table: dict[str, dict[str, Cell | None]]) -> str:
    name_w, cell_w, row_h, top, left = 150, 100, 22, 66, 12
    cols = [*SETS, ("mean", "mean of 8")]
    width = left + name_w + cell_w * len(cols) + 12
    order = ranked(table)
    height = top + row_h * len(order) + 52
    body = [
        text(left, 22, "Accuracy by engine and set (first timed call)", size=16, weight="bold"),
        text(left, 40, "Sorted by the unweighted mean of the 8 sets. † = trained on this set's train split. "
                       "- = not run, failed = refused the set.", size=11, fill="#555555"),
    ]
    for j, (_, label) in enumerate(cols):
        x = left + name_w + cell_w * j + cell_w / 2
        body.append(text(x, top - 8, label, size=11, anchor="middle", weight="bold"))
    for i, engine in enumerate(order):
        y = top + row_h * i
        cells = table[engine]
        body.append(text(left, y + 15, NAMES[engine], size=12))
        for j, (d, _) in enumerate(cols):
            x = left + name_w + cell_w * j
            if d == "mean":
                acc = mean_acc(cells) if complete(cells) else None
                label = f"{acc:.1f}%" if acc is not None else "-"
            else:
                cell = cells.get(d)
                acc = cell.acc if cell else None
                label = "-" if d not in cells else "failed" if cell is None else (
                    f"{round(acc)}%" + ("†" if (engine, d) in IN_DISTRIBUTION else ""))
            fill, ink = colour(acc) if acc is not None else ("#eeeeee", "#777777")
            body.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cell_w - 2}" height="{row_h - 2}" '
                        f'rx="3" fill="{fill}"/>')
            body.append(text(x + cell_w / 2, y + 15, label, size=11, anchor="middle",
                             weight="bold" if d == "mean" else "normal", fill=ink))
    # colour key
    ky = top + row_h * len(order) + 20
    kx = left + name_w
    steps = 12
    for k in range(steps):
        acc = LOW + (HIGH - LOW) * k / (steps - 1)
        body.append(f'<rect x="{kx + 24 * k}" y="{ky}" width="24" height="12" fill="{colour(acc)[0]}"/>')
    body.append(text(kx - 6, ky + 10, f"≤{LOW:.0f}%", size=11, anchor="end", fill="#555555"))
    body.append(text(kx + 24 * steps + 6, ky + 10, f"≥{HIGH:.0f}%", size=11, fill="#555555"))
    body.append(text(left, ky + 30, "Source: bench/runs/<set>/report.md, drawn by bench/charts.py.", size=11,
                     fill="#555555"))
    return svg(width, height, body)


def speed_vs_accuracy(table: dict[str, dict[str, Cell | None]]) -> str:
    width, height = 960, 680
    px0, px1, py0, py1 = 70, width - 30, 80, height - 70  # plot box: x left/right, y top/bottom
    xmin, xmax = 8.0, 30_000.0
    ymin, ymax = 50.0, 90.0
    engines = [e for e in ranked(table) if complete(table[e])]
    pts = {e: (statistics.median(table[e][d].p50 for d, _ in SETS), mean_acc(table[e])) for e in engines}

    def sx(ms: float) -> float:
        return px0 + (math.log10(ms) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (px1 - px0)

    def sy(acc: float) -> float:
        return py1 - (acc - ymin) / (ymax - ymin) * (py1 - py0)

    # Pareto front: no faster engine is at least as accurate.
    front, best = [], -1.0
    for e in sorted(engines, key=lambda e: pts[e]):
        if pts[e][1] > best:
            front.append(e)
            best = pts[e][1]

    body = [
        text(px0, 26, "Speed vs accuracy (engines measured on all 8 sets)", size=16, weight="bold"),
        text(px0, 44, "x: median over the 8 sets of first-call p50 (log scale). y: unweighted mean accuracy "
                      "of the 8 sets.", size=11, fill="#555555"),
        text(px0, 60, "Line: the fastest engine at each accuracy (Pareto front). Hollow = ran on CPU here, "
                      "so its latency does not compare with the Metal engines.", size=11, fill="#555555"),
    ]
    for acc in range(int(ymin), int(ymax) + 1, 5):
        y = sy(acc)
        body.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        body.append(text(px0 - 8, y + 4, f"{acc}%", size=11, anchor="end", fill="#555555"))
    for ms in (10, 30, 100, 300, 1000, 3000, 10_000, 30_000):
        x = sx(ms)
        body.append(f'<line x1="{x:.1f}" y1="{py0}" x2="{x:.1f}" y2="{py1}" stroke="#e5e5e5"/>')
        body.append(text(x, py1 + 18, f"{ms / 1000:g} s" if ms >= 1000 else f"{ms} ms", size=11,
                         anchor="middle", fill="#555555"))
    body.append(f'<rect x="{px0}" y="{py0}" width="{px1 - px0}" height="{py1 - py0}" fill="none" stroke="#999999"/>')
    body.append(text((px0 + px1) / 2, height - 22, "first-call p50, median of 8 sets (faster is left)", size=12,
                     anchor="middle"))
    path = " ".join(f"{'M' if k == 0 else 'L'}{sx(pts[e][0]):.1f},{sy(pts[e][1]):.1f}" for k, e in enumerate(front))
    body.append(f'<path d="{path}" fill="none" stroke="#21918c" stroke-width="2" stroke-dasharray="6 4"/>')

    # Labels must not cover another label or any point.
    placed = [(sx(ms) - 6, sy(acc) - 6, sx(ms) + 6, sy(acc) + 6) for ms, acc in pts.values()]
    for e in sorted(engines, key=lambda e: -pts[e][1]):
        x, y = sx(pts[e][0]), sy(pts[e][1])
        on_front = e in front
        ink = "#21918c" if on_front else "#3b528b"
        fill = "#ffffff" if e in CPU else ink
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{ink}" stroke-width="2"/>')
        label, size = NAMES[e], 11
        w, h = len(label) * 6.2, 12
        for dx, dy, anchor in ((9, 4, "start"), (-9, 4, "end"), (0, -10, "middle"), (0, 19, "middle"),
                               (8, -9, "start"), (-8, -9, "end"), (8, 17, "start"), (-8, 17, "end"),
                               (0, -22, "middle"), (0, 31, "middle")):
            x0 = x + dx - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
            box = (x0, y + dy - h + 2, x0 + w, y + dy + 2)
            if box[0] >= px0 and box[2] <= px1 and not any(
                    box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                placed.append(box)
                body.append(text(x + dx, y + dy, label, size=size, anchor=anchor,
                                 weight="bold" if on_front else "normal"))
                break
        else:
            raise SystemExit(f"speed_vs_accuracy: no free spot for the {label!r} label; move a label offset")
    return svg(width, height, body)


def charts() -> dict[Path, str]:
    table = load()
    return {OUT / "accuracy.svg": heatmap(table), OUT / "speed-vs-accuracy.svg": speed_vs_accuracy(table)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if a committed chart differs from the reports")
    args = ap.parse_args()
    stale = []
    for path, content in charts().items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print(path.relative_to(ROOT))
    if stale:
        print(f"out of date: {', '.join(map(str, stale))}; run `uv run python bench/charts.py`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
