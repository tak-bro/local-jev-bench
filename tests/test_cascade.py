"""bench/cascade.py: cascade curves on hand-made engines, offline. Every expected value here is worked by hand."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import cascade  # noqa: E402
from cascade import Item  # noqa: E402

# Four one-decision items. The small engine is wrong on a and b; the big engine is wrong on c only.
SMALL = {"a": Item([False], 0.55, 10.0), "b": Item([False], 0.9, 10.0), "c": Item([True], 0.6, 10.0),
         "d": Item([True], 0.95, 10.0)}
BIG = {"a": Item([True], 0.99, 100.0), "b": Item([True], 0.99, 100.0), "c": Item([False], 0.99, 100.0),
       "d": Item([True], 0.99, 100.0)}
IDS = ["a", "b", "c", "d"]


def test_curve_routes_the_least_confident_first():
    pts = cascade.curve(SMALL, BIG, IDS)
    # order a (0.55), c (0.6), b (0.9), d (0.95)
    assert [(p.k, p.acc) for p in pts] == [(0, 0.5), (1, 0.75), (2, 0.5), (3, 0.75), (4, 0.75)]
    assert [p.ms for p in pts] == [10.0, 35.0, 60.0, 85.0, 110.0]
    assert pts[2].conf == 0.6


def test_ties_in_confidence_route_together():
    small = {**SMALL, "c": Item([True], 0.55, 10.0)}
    assert [p.k for p in cascade.curve(small, BIG, IDS)] == [0, 2, 3, 4]


def test_item_without_a_forecast_is_never_routed():
    small = {**SMALL, "d": Item([True], math.inf, 10.0)}
    assert cascade.curve(small, BIG, IDS)[-1].k == 3


def test_operating_point_is_the_lowest_share_within_one_point():
    pts = cascade.curve(SMALL, BIG, IDS)
    assert cascade.operating(pts, 0.75).k == 1


def test_mcnemar_counts_only_unrouted_items():
    # route a: unrouted b (small wrong, big right), c (small right, big wrong), d (both right)
    assert cascade.mcnemar_vs_big(SMALL, BIG, IDS, {"a"}) == (1, 1, 1.0)


def test_oracle_and_efficiency():
    pts, best = cascade.curve(SMALL, BIG, IDS), cascade.oracle(SMALL, BIG, IDS)
    # oracle order: a, b (big fixes), d (tie), c (big breaks): 0.5, 0.75, 1.0, 1.0, 0.75
    assert [p.acc for p in best] == [0.5, 0.75, 1.0, 1.0, 0.75]
    random_area = (0.5 + 0.75) / 2
    # trapezoids over steps of 0.25: cascade (0.625 + 0.625 + 0.625 + 0.75) / 4
    assert cascade.area(pts) == pytest.approx(0.65625)
    assert cascade.area(pts) - random_area == pytest.approx(0.03125)
    # oracle trapezoids (0.625 + 0.875 + 1.0 + 0.875) / 4 = 0.84375
    assert cascade.area(best) - random_area == pytest.approx(0.21875)
    assert (cascade.area(pts) - random_area) / (cascade.area(best) - random_area) == pytest.approx(1 / 7)


def test_auroc_counts_ties_half():
    assert cascade.auroc([(0.9, True), (0.8, True), (0.1, False)]) == 1.0
    assert cascade.auroc([(0.5, True), (0.5, False)]) == 0.5
    assert cascade.auroc([(0.2, True), (0.9, False)]) == 0.0
    assert cascade.auroc([(0.2, True)]) is None


def test_split_half_applies_one_halfs_threshold_to_the_other():
    # halves: [a, c] and [b, d]. On [a, c] the big engine scores 0.5; routing nothing already reaches it, so the
    # threshold routes nothing on [b, d]: share 0, accuracy 0.5 against the big engine's 1.0.
    (rate1, gap1), (rate2, gap2) = cascade.split_half(SMALL, BIG, IDS)
    assert (rate1, gap1) == (0.0, -0.5)
    # On [b, d] the big engine scores 1.0; reaching it needs b routed (conf 0.9), which on [a, c] routes both.
    assert (rate2, gap2) == (1.0, 0.0)


def test_spearman_of_a_monotone_relation_is_one():
    rho, p = cascade.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], perms=200)
    assert rho == pytest.approx(1.0) and p < 0.05
