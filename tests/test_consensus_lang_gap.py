"""bench/consensus.py and bench/lang_gap.py on hand-made logs, offline. Every expected value is worked by hand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import consensus  # noqa: E402
import lang_gap  # noqa: E402
import make_sets  # noqa: E402
import score  # noqa: E402

OPTIONS = {"a": "", "b": ""}


def items(gold: list[str], states: list[str]) -> list[dict]:
    return [{"id": f"i{n}", "state": s, "expected": {"q": g},
             "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": OPTIONS}}}
            for n, (g, s) in enumerate(zip(gold, states))]


def run_dir(tmp_path: Path, name: str, set_items: list[dict], picks: dict[str, str]) -> Path:
    """bench/runs-style directory: the set and one log per engine, `picks` = {engine: one choice per item}."""
    d = tmp_path / name
    d.mkdir()
    set_path = tmp_path / f"{name}.jsonl"
    make_sets.write_set(set_items, set_path)
    header = {"set": str(set_path), "set_sha256": score.sha256(set_path), "model": None, "reps": 1, "warmup": 3,
              "started": "2026-11-12T00:00:00", "host": "test"}
    for engine, choices in picks.items():
        recs = [{"item": it["id"], "call": "timed", "rep": 0, "error": None, "ms": 1.0, "pressure": "normal",
                 "answers": {"q": {"type": "choice", "choice": c}}} for it, c in zip(set_items, choices)]
        lines = [{"header": {**header, "engine": engine}}, *recs, {"footer": {"finished": "x"}}]
        (d / f"{engine}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return d


def test_candidates_are_unanimous_panel_errors(tmp_path):
    # gold a a a a. i1: whole panel says b (candidate). i2: panel split. i3: panel right.
    d = run_dir(tmp_path, "s", items(list("aaaa"), list("wxyz")),
                {"e1": "abba", "e2": "abaa", "e3": "bbba", "weak": "bbbb"})
    _, set_items, logs = score.load_dir(d)
    runs = {e: recs for e, (_, recs) in logs.items()}
    panel, cand = consensus.panel_and_candidates(set_items, runs, top=2, skip=set())
    assert panel == ["e2", "e1"]  # 3/4 then 2/4; weak (0/4) and e3 (1/4) left out
    assert cand == {("i1", "q"): "b"}
    # skipping e2 brings e3 into the panel: i1 and i2 are then unanimous errors (b, b)
    _, cand = consensus.panel_and_candidates(set_items, runs, top=2, skip={"e2"})
    assert sorted(cand) == [("i1", "q"), ("i2", "q")]


def test_section_reports_accuracy_without_candidates(tmp_path):
    d = run_dir(tmp_path, "s", items(list("aaaa"), list("wxyz")), {"e1": "abba", "e2": "abaa"})
    text = "\n".join(consensus.section(d, top=2, skip=set()))
    assert "1 of 4 decisions (25.0%)" in text
    assert "| e2 | 3/4 (75%) | 3/3 (100%) |" in text
    assert "| i1 | q | a | b | x |" in text


def test_lang_gap_pairs_by_item(tmp_path):
    gold = list("abab")
    en = run_dir(tmp_path, "en", items(gold, ["one", "two", "three", "four"]), {"e": "abab", "only-en": "aaaa"})
    ko = run_dir(tmp_path, "ko", items(gold, ["하나", "둘", "셋", "넷"]), {"e": "aaaa"})
    (row,) = lang_gap.gap_rows(lang_gap.load(en), lang_gap.load(ko))
    # e: right 4/4 in en, 2/4 in ko; i1 and i3 right only in en
    assert row == ("e", 4, 4, 2, 2, 0)
    assert "| e | 4 | 4/4 (100%) | 2/4 (50%) | -50.0 | 2 | 0 | 0.500 |" in lang_gap.table("en", "ko", [row])


def test_lang_gap_refuses_sets_that_are_not_parallel(tmp_path):
    en = run_dir(tmp_path, "en", items(list("ab"), ["one", "two"]), {"e": "ab"})
    ko = run_dir(tmp_path, "ko", items(list("aa"), ["하나", "둘"]), {"e": "ab"})
    with pytest.raises(SystemExit, match="not parallel"):
        lang_gap.gap_rows(lang_gap.load(en), lang_gap.load(ko))
