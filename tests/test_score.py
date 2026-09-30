"""bench/score.py: raw benchmark logs -> report, offline. Every expected value here is worked by hand."""

from __future__ import annotations

import json
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
    assert "| 4/4 (100%, 51-100) | n/a |" in row(score.render(d), "e")


def test_failed_rep_0_drops_its_decisions(runs):
    d, s = runs
    records = [r if (r["item"], r["call"], r["rep"]) != ("i1", "timed", 0) else {**r, "answers": None, "error": "boom"}
               for r in calls()]
    write_raw(d, "e", s, records)
    r = row(score.render(d), "e")
    assert "| 2/2 (100%, 34-100) |" in r  # rep 1 of i1 answered, but only rep 0 is graded
    cols = [c.strip() for c in r.split("|")]
    assert cols[7] == "5" and cols[11] == "1"  # the failed call is neither a latency sample nor graded


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
        "| e | - | failed | failed | failed | failed | 0 | failed | failed | failed | 1 | normal |"


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
