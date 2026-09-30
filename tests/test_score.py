"""bench/score.py: raw benchmark logs -> report, offline. Every expected value here is worked by hand."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import score  # noqa: E402

ITEMS = [{"id": f"i{i}", "state": f"state {i}",
          "questions": {"u": {"type": "noul", "instructions": "?"},
                        "d": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}},
          "expected": {"u": True, "d": "a"}} for i in range(2)]

RIGHT = {"u": {"type": "noul", "noul": 0.9}, "d": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}}
WRONG = {"u": {"type": "noul", "noul": 0.1}, "d": {"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9}}}


def rec(item, call, rep=None, answers=None, error=None, ms=10.0, pressure="normal"):
    return {"item": item, "call": call, "rep": rep, "answers": answers, "error": error, "ms": ms, "pressure": pressure}


def calls(reps=3, rep0=RIGHT, later=RIGHT, reversed_=RIGHT):
    out = [rec("i0", "cold", answers=RIGHT, ms=500.0)] + [rec("i0", "warmup", answers=RIGHT) for _ in range(3)]
    for it in ITEMS:
        out += [rec(it["id"], "timed", rep, rep0 if rep == 0 else later) for rep in range(reps)]
        out.append(rec(it["id"], "reversed", answers=reversed_))
    return out


def write_set(d: Path, items=ITEMS) -> Path:
    p = d / "set.jsonl"
    p.write_text("".join(json.dumps(it) + "\n" for it in items))
    return p


def write_raw(d: Path, engine: str, set_path: Path, records, reps=3, sha=None) -> Path:
    header = {"set": str(set_path), "set_sha256": sha or score.sha256(set_path), "engine": engine, "model": None,
              "reps": reps, "warmup": 3, "started": "2026-09-30T00:00:00", "host": "test"}
    footer = {"finished": "2026-09-30T00:01:00"}
    p = d / f"{engine}.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in [{"header": header}, *records, {"footer": footer}]))
    return p


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d, write_set(tmp_path)


def row(text: str, engine: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {engine} |"))


def test_accuracy_counts_each_decision_once_from_rep_0(runs):
    d, s = runs
    write_raw(d, "e", s, calls(rep0=RIGHT, later=WRONG))
    text = score.render(d)
    # 2 items x 2 questions = 4 decisions; the 8 wrong answers of reps 1 and 2 are not in the denominator
    assert "| 4/4 (100%, 51-100) |" in row(text, "e")
    assert "| 4/4 |" in row(text, "e")  # rep-disagree: every decision changed at a later rep


def test_rep_disagree_is_na_with_one_rep(runs):
    d, s = runs
    write_raw(d, "e", s, calls(reps=1), reps=1)
    r = row(score.render(d), "e")
    assert "| 4/4 (100%, 51-100) |" in r and "| n/a | 0/2 (0%) |" in r


def test_failed_rep_0_drops_its_decisions(runs):
    d, s = runs
    records = [r if (r["item"], r["call"], r["rep"]) != ("i1", "timed", 0) else {**r, "answers": None, "error": "boom"}
               for r in calls()]
    write_raw(d, "e", s, records)
    r = row(score.render(d), "e")
    assert "| 2/2 (100%, 34-100) |" in r  # rep 1 of i1 answered, but only rep 0 is graded
    cols = [c.strip() for c in r.split("|")]
    assert cols[7] == "5" and cols[13] == "1"  # the failed call is neither a latency sample nor graded


def test_refuses_raw_from_another_set(runs):
    d, s = runs
    write_raw(d, "e", s, calls())
    write_raw(d, "old", s, calls(), sha="0" * 64)
    with pytest.raises(SystemExit, match=r"old\.jsonl.*0{12}"):
        score.render(d)


def test_refuses_a_set_edited_after_measuring(runs):
    d, s = runs
    write_raw(d, "e", s, calls())
    s.write_text(s.read_text() + json.dumps({**ITEMS[0], "id": "i2"}) + "\n")
    with pytest.raises(SystemExit, match="measure again"):
        score.render(d)


def test_refuses_mixed_reps(runs):
    d, s = runs
    write_raw(d, "a", s, calls())
    write_raw(d, "b", s, calls(reps=1), reps=1)
    with pytest.raises(SystemExit, match="reps"):
        score.render(d)


def test_truncated_last_line_is_an_error(runs):
    d, s = runs
    p = write_raw(d, "e", s, calls())
    p.write_text(p.read_text() + '{"item": "i1", "ca')
    with pytest.raises(SystemExit, match=r"e\.jsonl:15"):
        score.render(d)


def test_log_of_an_interrupted_run_is_an_error(runs):
    # Every record is written whole, so a killed run ends on a complete line; only the missing footer tells.
    d, s = runs
    p = write_raw(d, "e", s, calls())
    p.write_text("".join(p.read_text().splitlines(keepends=True)[:-3]))
    with pytest.raises(SystemExit, match="no footer"):
        score.render(d)


def test_refuses_two_logs_of_one_engine(runs):
    d, s = runs
    write_raw(d, "e", s, calls())
    (d / "e-copy.jsonl").write_text((d / "e.jsonl").read_text())
    with pytest.raises(SystemExit, match="e-copy.jsonl.*engine 'e'"):
        score.render(d)


def test_empty_dir_is_an_error(runs):
    d, _ = runs
    with pytest.raises(SystemExit, match="no raw logs"):
        score.render(d)


def test_unknown_score_rule_is_an_error(runs, tmp_path):
    d, _ = runs
    s = write_set(tmp_path, [{**ITEMS[0], "scoring": {"score": "median"}}])
    write_raw(d, "e", s, calls())
    with pytest.raises(SystemExit, match="median"):
        score.render(d)


def test_score_rounds_half_up():
    q = {"type": "score"}
    assert score.decide(q, {"score": 0.5}) == 1 and score.decide(q, {"score": 2.5}) == 3


def test_order_flip_counts_changed_choices(runs):
    d, s = runs
    write_raw(d, "flips", s, calls(reversed_=WRONG))
    write_raw(d, "steady", s, calls())
    text = score.render(d)
    assert "| 2/2 (100%) |" in row(text, "flips")  # only the choice question counts: 2 items, not 4 decisions
    assert "| 0/2 (0%) |" in row(text, "steady")


def test_order_flip_failed_or_na(runs, tmp_path):
    d, s = runs
    write_raw(d, "e", s, [r if r["call"] != "reversed" else {**r, "answers": None, "error": "boom"} for r in calls()])
    r = row(score.render(d), "e")
    assert "| 4/4 (100%, 51-100) |" in r and "| failed | 2 |" in r  # forward answers are still graded
    noul_only = [{**it, "questions": {"u": it["questions"]["u"]}, "expected": {"u": True}} for it in ITEMS]
    (tmp_path / "noul").mkdir()
    d2 = tmp_path / "noul" / "runs"
    d2.mkdir()
    write_raw(d2, "e", write_set(tmp_path / "noul", noul_only), [r for r in calls() if r["call"] != "reversed"])
    assert "| n/a | 0 |" in row(score.render(d2), "e")


def test_engine_that_failed_its_cold_call(runs):
    d, s = runs
    write_raw(d, "e", s, [rec("i0", "cold", error="down", pressure="normal")])
    assert row(score.render(d), "e") == \
        "| e | - | failed | failed | failed | failed | 0 | failed | failed | failed | failed | failed | 1 | normal |"


def test_worst_pressure(runs):
    d, s = runs
    records = calls()
    records[5]["pressure"], records[7]["pressure"] = "critical", "warn"
    write_raw(d, "e", s, records)
    assert row(score.render(d), "e").endswith("| critical |")


def test_latency_columns(runs):
    d, s = runs
    records = calls()
    for i, r in enumerate(r for r in records if r["call"] == "timed"):
        r["ms"] = float(i + 1)  # i0 reps 1..3 ms, i1 reps 4..6 ms
    write_raw(d, "e", s, records)
    cols = [c.strip() for c in row(score.render(d), "e").split("|")[3:8]]
    # cold 500; first-call p50 of {1, 4}; all-call p50 and p95 of 1..6 (inclusive quantiles); 6 timed calls
    assert cols == ["500.0", "2.5", "3.5", "5.8", "6"]


def test_wilson():
    lo, hi = score.wilson(237, 315)
    assert (round(lo, 3), round(hi, 3)) == (0.702, 0.797)


def test_rows_are_sorted_and_report_is_written(runs):
    d, s = runs
    write_raw(d, "zeta", s, calls())
    write_raw(d, "alpha", s, calls())
    text = score.render(d)
    assert text.index("| alpha |") < text.index("| zeta |")
    assert (d / "report.md").read_text() == text
    assert f"sha256 `{score.sha256(s)[:12]}`" in text and "2 items, 4 decisions" in text


def test_check_flags_a_stale_report(runs, monkeypatch, capsys):
    d, s = runs
    write_raw(d, "e", s, calls())
    score.render(d)
    monkeypatch.setattr(sys, "argv", ["score.py", str(d), "--check"])
    assert score.main() == 0
    (d / "report.md").write_text("hand edit\n")
    assert score.main() == 1
    assert "stale" in capsys.readouterr().err


# --- slice 02: pairwise McNemar, Brier, ECE ------------------------------------------------------

@pytest.mark.parametrize("b, c, p", [(0, 0, 1.0), (1, 9, 0.0215), (9, 1, 0.0215), (3, 5, 0.7266), (0, 1, 1.0)])
def test_mcnemar_exact(b, c, p):
    # two-sided exact binomial on the discordant pairs: 2 * P(X <= min(b, c)), X ~ Bin(b + c, 1/2), at most 1
    assert round(score.mcnemar_exact(b, c), 4) == p


def test_ece_is_zero_when_confidence_matches_accuracy():
    assert score.ece([(1.0, True)] * 5) == 0.0
    assert score.ece([(0.7, True)] * 7 + [(0.7, False)] * 3) == pytest.approx(0.0)


def test_ece_of_an_overconfident_coin():
    assert score.ece([(1.0, True), (1.0, False)] * 10) == pytest.approx(0.5)


def test_ece_weights_bins_by_count():
    # bin of 0.9: 4 decisions, all right (gap 0.1); bin of 0.6: 1 decision, wrong (gap 0.6) -> (4*0.1 + 0.6) / 5
    assert score.ece([(0.9, True)] * 4 + [(0.6, False)]) == pytest.approx(0.2)


def test_brier_and_ece_columns(runs):
    d, s = runs
    write_raw(d, "right", s, calls())
    write_raw(d, "wrong", s, calls(rep0=WRONG))
    text = score.render(d)
    # RIGHT: noul 0.9 on a yes = (0.1)^2 + (0.1)^2 = 0.02, choice {a: .9, b: .1} on a = 0.02; confidence 0.9, all right
    assert "| 0.020 (4) | 0.100 |" in row(text, "right")
    # WRONG: noul 0.1 on a yes = 0.81 + 0.81 = 1.62, choice {a: .1, b: .9} on a = 1.62; confidence 0.9, all wrong
    assert "| 1.620 (4) | 0.900 |" in row(text, "wrong")


def test_brier_skips_score_questions(runs, tmp_path):
    d, _ = runs
    items = [{**it, "questions": {**it["questions"], "s": {"type": "score", "instructions": "?",
                                                              "criteria": ["lo", "mid", "hi"]}},
              "expected": {**it["expected"], "s": 1}} for it in ITEMS]
    s = write_set(tmp_path, items)
    ans = {**RIGHT, "s": {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}}
    write_raw(d, "e", s, calls(rep0=ans, later=ans, reversed_=ans))
    r = row(score.render(d), "e")
    assert "| 6/6 (100%" in r and "| 0.020 (4) |" in r  # 6 decisions graded, 4 of them noul or choice


def test_pairs_table(runs):
    d, s = runs
    half = {**RIGHT, "d": WRONG["d"]}  # right on the noul, wrong on the choice
    write_raw(d, "a", s, calls())
    write_raw(d, "b", s, calls(rep0=half))
    write_raw(d, "c", s, calls(rep0=half))
    text = score.render(d)
    # a is right on 4, b on 2 (the nouls): b=2 discordant for a, c=0, p = 2 * P(X <= 0 | n=2) = 0.5
    assert "| a | b | 4 | 2 | 0 | 0.500 |" in text
    assert "overlapping" not in text  # the pairwise table, not interval overlap, says whether engines differ
    assert "| b | c | 4 | 0 | 0 | 1.000 |" in text
    pair_rows = [line[:9] for line in text.splitlines() if line.count("|") == 7 and line[2] in "abc"]
    assert pair_rows == ["| a | b |", "| a | c |", "| b | c |"]


def test_pairs_leave_out_a_decision_either_engine_failed(runs):
    d, s = runs
    failed_i1 = [r if (r["item"], r["call"], r["rep"]) != ("i1", "timed", 0) else {**r, "answers": None, "error": "x"}
                 for r in calls()]
    write_raw(d, "a", s, calls())
    write_raw(d, "b", s, failed_i1)
    assert "| a | b | 2 | 0 | 0 | 1.000 |" in score.render(d)  # i1 is out on both sides


def test_no_pairs_table_for_one_engine(runs):
    d, s = runs
    write_raw(d, "a", s, calls())
    assert "## Pairwise" not in score.render(d)


def test_choice_probabilities_keyed_otherwise_are_an_error(runs):
    # An engine that keys probabilities by option text (or index) would otherwise score every option 0.
    d, s = runs
    by_text = {**RIGHT, "d": {"type": "choice", "choice": "a", "probabilities": {"A": 0.9, "B": 0.1}}}
    write_raw(d, "e", s, calls(rep0=by_text, later=by_text, reversed_=by_text))
    with pytest.raises(SystemExit, match="e i0 d: chose 'a'.*probabilities"):
        score.render(d)


def test_small_p_values_are_not_printed_as_zero(runs):
    assert score.p_cell(score.mcnemar_exact(40, 80)) == "<0.001"
    assert score.p_cell(0.5) == "0.500"


# --- slice 03: argmax rule, breakdown by source and variant ------------------------------------------

SCORE_Q = {"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]}


def test_argmax_rule_reads_score_probabilities_keyed_by_index_or_criterion():
    by_index = {"score": 0.9, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}}
    by_text = {"score": 0.9, "probabilities": {"lo": 0.1, "mid": 0.3, "hi": 0.6}}
    # the expected value 0.9 rounds to 1; the most probable level is 2
    assert score.decide(SCORE_Q, by_index, "round") == 1
    assert score.decide(SCORE_Q, by_index, "argmax") == 2 and score.decide(SCORE_Q, by_text, "argmax") == 2


def test_argmax_rule_without_probabilities_is_an_error():
    with pytest.raises(SystemExit, match="argmax"):
        score.decide(SCORE_Q, {"score": 0.9, "probabilities": {}}, "argmax")


def test_breakdown_by_source_and_variant(runs, tmp_path):
    d, _ = runs
    items = [{**ITEMS[0], "meta": {"source": "mmlu", "variant": "clean"}},
             {**ITEMS[1], "meta": {"source": "paws", "variant": "permuted"}}]
    s = write_set(tmp_path, items)
    wrong_i1 = [r if (r["item"], r["call"], r["rep"]) != ("i1", "timed", 0) else {**r, "answers": WRONG}
                for r in calls()]
    write_raw(d, "e", s, wrong_i1)
    text = score.render(d)
    assert "## By source" in text and "## By variant" in text
    assert "| mmlu | 2 | 2/2 (100%) |" in text and "| paws | 2 | 0/2 (0%) |" in text
    assert "| clean | 2 | 2/2 (100%) |" in text and "| permuted | 2 | 0/2 (0%) |" in text


def test_no_breakdown_without_meta(runs):
    d, s = runs
    write_raw(d, "e", s, calls())
    assert "## By source" not in score.render(d)


def test_argmax_refuses_probabilities_that_read_both_ways():
    # criteria "1", "0": index keys and criterion-text keys are the same strings but name different levels
    q = {"type": "score", "instructions": "?", "criteria": ["1", "0"]}
    with pytest.raises(SystemExit, match="ambiguous"):
        score.decide(q, {"score": 0.0, "probabilities": {"0": 0.8, "1": 0.2}}, "argmax")


def test_argmax_reads_digit_criteria_keyed_by_text():
    # index keys are incomplete ("3" is not an index of three levels), so the keys are criterion texts
    q = {"type": "score", "instructions": "?", "criteria": ["1", "2", "3"]}
    assert score.decide(q, {"score": 0.0, "probabilities": {"1": 0.1, "2": 0.3, "3": 0.6}}, "argmax") == 2


def test_argmax_error_names_engine_item_and_question(runs, tmp_path):
    d, _ = runs
    items = [{**it, "questions": {"s": SCORE_Q}, "expected": {"s": 1}, "scoring": {"score": "argmax"}} for it in ITEMS]
    s = write_set(tmp_path, items)
    ans = {"s": {"type": "score", "score": 1.0, "probabilities": {}}}
    write_raw(d, "e", s, [r for r in calls(rep0=ans, later=ans) if r["call"] != "reversed"])
    with pytest.raises(SystemExit, match="e i0 s: .*argmax"):
        score.render(d)


def test_breakdown_counts_decisions_an_engine_failed(runs, tmp_path):
    d, _ = runs
    items = [{**it, "meta": {"source": "mmlu"}} for it in ITEMS]
    s = write_set(tmp_path, items)
    failed_i1 = [r if (r["item"], r["call"], r["rep"]) != ("i1", "timed", 0) else {**r, "answers": None, "error": "x"}
                 for r in calls()]
    write_raw(d, "e", s, failed_i1)
    assert "| mmlu | 4 | 2/2 (100%), 2 failed |" in score.render(d)


# --- slice 04: KL and Brier against gold distributions, uniform and prior references ---------------

def test_kl_from_gold():
    # 0.5 ln(0.5/0.25) + 0.5 ln(0.5/0.75) = 0.5 (0.693147) + 0.5 (-0.405465)
    assert score.kl([0.5, 0.5], [0.25, 0.75]) == pytest.approx(0.143841, abs=1e-6)
    assert score.kl([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.log(1 / 1e-6), rel=1e-3)  # floored, not infinite


GOLD_ITEMS = [{"id": "g0", "state": "s", "questions": {"u": {"type": "noul", "instructions": "?"}},
               "expected": {"u": True}, "scoring": {"score": "argmax"},
               "gold": {"u": {"false": 0.2, "true": 0.8}}, "prior": {"u": {"false": 0.4, "true": 0.6}}}]


def test_gold_table_rows(runs, tmp_path):
    d, _ = runs
    s = write_set(tmp_path, GOLD_ITEMS)
    ans = {"u": {"type": "noul", "noul": 0.9}}
    write_raw(d, "e", s, [rec("g0", "cold", answers=ans), rec("g0", "timed", 0, ans)], reps=1)
    text = score.render(d)
    assert "## Against the gold distributions" in text
    # engine: [yes 0.9, no 0.1] vs gold [0.8, 0.2]: KL = 0.8 ln(0.8/0.9) + 0.2 ln(0.2/0.1) = 0.044395; Brier = 0.02
    assert "| e | 1 | 1/1 (100%) | 0.044 | 0.020 |" in text
    # uniform: KL = 0.8 ln 1.6 + 0.2 ln 0.4 = 0.192745; Brier = 0.09 + 0.09; a random pick of 2 is right half the time
    assert "| (uniform) | 1 | 0.5 expected | 0.193 | 0.180 |" in text
    # prior [0.6, 0.4]: KL = 0.8 ln(0.8/0.6) + 0.2 ln(0.2/0.4) = 0.091516; Brier = 0.04 + 0.04; argmax yes is right
    assert "| (prior) | 1 | 1/1 (100%) | 0.092 | 0.080 |" in text


def test_gold_table_aligns_choice_and_score_and_drops_failed_calls(runs, tmp_path):
    d, _ = runs
    q = {"c": {"type": "choice", "instructions": "?", "criteria": {"x": "X", "y": "Y"}}, "s": SCORE_Q}
    items = [{"id": f"g{i}", "state": "s", "questions": q, "expected": {"c": "y", "s": 2},
              "scoring": {"score": "argmax"},
              "gold": {"c": {"x": 0.25, "y": 0.75}, "s": {"0": 0.0, "1": 0.5, "2": 0.5}}} for i in range(2)]
    s = write_set(tmp_path, items)
    ans = {"c": {"type": "choice", "choice": "y", "probabilities": {"x": 0.25, "y": 0.75}},
           "s": {"type": "score", "score": 1.5, "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}}}
    write_raw(d, "e", s, [rec("g0", "cold", answers=ans), rec("g0", "timed", 0, ans),
                          rec("g1", "timed", 0, error="boom"), rec("g0", "reversed", answers=ans),
                          ], reps=1)
    text = score.render(d)
    # the engine matches gold exactly on the 2 decisions of g0 (the score argmax ties 1 and 2 and takes 1: wrong);
    # g1 failed and is out of the engine row, not of the references
    assert "| e | 2 | 1/2 (50%) | 0.000 | 0.000 |" in text
    assert "| (uniform) | 4 |" in text


def test_argmax_sets_count_score_questions_in_brier(runs, tmp_path):
    d, _ = runs
    items = [{**it, "questions": {"s": SCORE_Q}, "expected": {"s": 2}, "scoring": {"score": "argmax"}} for it in ITEMS]
    s = write_set(tmp_path, items)
    ans = {"s": {"type": "score", "score": 1.5, "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}}}
    write_raw(d, "e", s, [r for r in calls(reps=1, rep0=ans) if r["call"] != "reversed"], reps=1)
    # [0, .5, .5] against level 2: 0 + 0.25 + 0.25 = 0.5 per decision
    assert "| 0.500 (2) |" in row(score.render(d), "e")
