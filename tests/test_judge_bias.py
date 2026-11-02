"""bench/make_sets.py judge sets and bench/judge_bias.py, offline. Every expected value here is worked by hand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import judge_bias  # noqa: E402
import make_sets  # noqa: E402
import score  # noqa: E402


def mt_row(qid, a, b, winner, turn=1):
    conv = lambda m: [{"role": "user", "content": f"q{qid}"}, {"role": "assistant", "content": f"{m} says"}]  # noqa: E731
    return {"question_id": qid, "model_a": a, "model_b": b, "winner": winner, "judge": "expert_0",
            "conversation_a": conv(a), "conversation_b": conv(b), "turn": turn}


def test_mt_bench_pairs_take_a_strict_majority_over_both_orientations():
    rows = [mt_row(1, "vicuna", "alpaca", "model_a"), mt_row(1, "alpaca", "vicuna", "model_b"),
            mt_row(1, "vicuna", "alpaca", "tie"),                      # 2 of 3 for vicuna: kept
            mt_row(2, "gpt-4", "llama", "model_a"), mt_row(2, "gpt-4", "llama", "tie"),  # 1 of 2: dropped
            mt_row(3, "gpt-4", "llama", "model_b", turn=2)]            # second turn: dropped
    (p,) = make_sets.mt_bench_pairs(rows)
    assert p["id"] == "mtb/1/alpaca-vs-vicuna"
    assert (p["x"], p["y"], p["winner"]) == ("alpaca says", "vicuna says", "y")  # x sorts first
    assert p["meta"] == {"models": ["alpaca", "vicuna"], "votes": 3, "winner_votes": 2}


def test_ko_pairs_drop_ties():
    rows = [{"instruction": "질문", "response_a": "가", "response_b": "나", "decision": d, "source": "s"}
            for d in ("A", "Tie", "B")]
    assert [(p["id"], p["winner"]) for p in make_sets.ko_pairs(rows)] == [("ko/000", "x"), ("ko/002", "y")]


PAIR = {"id": "p", "lang": "en", "source": "s", "question": "Why?", "x": "Short.", "y": "Much longer answer.",
        "winner": "y"}


def test_judge_items_put_each_response_in_each_position():
    ab, ba = make_sets.judge_items(PAIR)
    assert ab["state"] == "Question:\nWhy?\n\nResponse A:\nShort.\n\nResponse B:\nMuch longer answer."
    assert ba["state"] == "Question:\nWhy?\n\nResponse A:\nMuch longer answer.\n\nResponse B:\nShort."
    assert (ab["expected"], ba["expected"]) == ({"better": "B"}, {"better": "A"})
    assert (ab["meta"]["lengths"], ba["meta"]["lengths"]) == ([6, 19], [19, 6])
    assert ab["questions"]["better"]["criteria"] == {"A": "Response A", "B": "Response B"}


def test_pad_adds_no_new_content():
    assert make_sets.pad("Paris is the capital. It is big.", "en") == (
        "Paris is the capital. It is big.\n\nIn short: Paris is the capital.\n\n" + make_sets.PAD["en"][1])
    assert make_sets.first_sentence("첫 줄\n둘째 줄") == "첫 줄"


def test_fits_counts_state_and_instructions():
    words = lambda text: len(text.split())  # noqa: E731
    n = words(make_sets.judge_state("en", PAIR["question"], PAIR["x"], PAIR["y"])) + 6  # 6 instruction words
    assert make_sets.fits(PAIR, words, n) and not make_sets.fits(PAIR, words, n - 1)


def answers(choice):
    return {"better": {"type": "choice", "choice": choice,
                       "probabilities": {"A": 0.8, "B": 0.2} if choice == "A" else {"A": 0.2, "B": 0.8}}}


def write_run(d: Path, set_path: Path, engine: str, picks: dict[str, str]) -> None:
    recs = [{"item": i, "call": "timed", "rep": 0, "answers": answers(c), "error": None, "ms": 1.0,
             "pressure": "normal"} for i, c in picks.items()]
    header = {"set": str(set_path), "set_sha256": score.sha256(set_path), "engine": engine, "model": None,
              "reps": 1, "warmup": 3, "started": "2026-10-31T00:00:00", "host": "test"}
    (d / f"{engine}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in
                                               [{"header": header}, *recs, {"footer": {"finished": "x"}}]))


def human_set(tmp_path):
    # p0: human y (longer). p1: human x (longer). p2: human x (shorter).
    pairs = [{**PAIR, "id": "p0"}, {**PAIR, "id": "p1", "x": "A much longer one.", "y": "Tiny.", "winner": "x"},
             {**PAIR, "id": "p2", "winner": "x"}]
    items = [it for p in pairs for it in make_sets.judge_items(p)]
    set_path = tmp_path / "judge.jsonl"
    make_sets.write_set(items, set_path)
    d = tmp_path / "runs"
    d.mkdir()
    return d, set_path


def test_engine_row_on_a_human_set(tmp_path):
    d, set_path = human_set(tmp_path)
    # p0: B, A -> y, y (consistent, human). p1: A, B -> x, x (consistent, human). p2: A, A -> x, y (flip, A both).
    write_run(d, set_path, "e", {"p0/ab": "B", "p0/ba": "A", "p1/ab": "A", "p1/ba": "B",
                                 "p2/ab": "A", "p2/ba": "A"})
    _, items, runs = score.load_dir(d)
    row = judge_bias.engine_row(items, runs["e"][1])
    assert row["pairs"] == "3"
    assert row["position flip"].startswith("1/3 ")
    assert row["A both"].startswith("1/1 ")
    assert row["human agreement"].startswith("5/6 ")  # p2/ba picked A, the shorter x's slot is B
    # judge: y, x, tie; human: y, x, x. po = 2/3; pe = (1/3)(1/3) + (1/3)(2/3) + 0 = 1/3; kappa = 0.5
    assert row["kappa"].startswith("0.50 ")
    # consistent pairs p0, p1: both picks are the longer response, and so are both human winners.
    assert row["longer pick (judge)"].startswith("2/2 ") and row["longer pick (human)"].startswith("2/2 ")
    assert row["longer gap"].startswith("+0.00")


def test_engine_row_on_a_length_set(tmp_path):
    pairs = [{**PAIR, "id": f"len/{i}", "source": "s-length", "y": make_sets.pad(PAIR["x"], "en"), "winner": "x"}
             for i in range(2)]
    set_path = tmp_path / "judge-length.jsonl"
    make_sets.write_set([it for p in pairs for it in make_sets.judge_items(p)], set_path)
    d = tmp_path / "runs"
    d.mkdir()
    # len/0: padded both times (B, A). len/1: A both times -> x then y.
    write_run(d, set_path, "e", {"len/0/ab": "B", "len/0/ba": "A", "len/1/ab": "A", "len/1/ba": "A"})
    _, items, runs = score.load_dir(d)
    row = judge_bias.engine_row(items, runs["e"][1])
    assert row["padded pick"].startswith("3/4 ")
    assert row["padded in both orders"].startswith("1/2 ")
    assert "kappa" not in row


def test_mcnemar_counts_pairs_not_items(tmp_path):
    d, set_path = human_set(tmp_path)
    write_run(d, set_path, "good", {"p0/ab": "B", "p0/ba": "A", "p1/ab": "A", "p1/ba": "B",
                                    "p2/ab": "A", "p2/ba": "B"})
    write_run(d, set_path, "flip", {f"p{i}/{o}": "A" for i in range(3) for o in ("ab", "ba")})
    text = "\n".join(judge_bias.section(d))
    assert "| flip | good | 3 | 0 | 3 | 0.250 |" in text


def test_self_pairs_drop_truncated_answers():
    import make_judge_self

    gen = lambda m, q: None if m == "gemma" and q.startswith("q2") else f"{m}: {q[:2]}"  # noqa: E731
    pairs = make_judge_self.self_pairs([("a", "en", "q1"), ("b", "ko", "q2")], "qwen", "gemma", gen, {"d": 1})
    (p,) = pairs
    assert (p["id"], p["x"], p["y"], p["winner"], p["source"]) == ("self/a", "qwen: q1", "gemma: q1", "x", "en-self")
    assert p["meta"] == {"models": ["qwen", "gemma"], "d": 1}


def test_self_set_compares_only_judges_of_different_families(tmp_path):
    pairs = [{**PAIR, "id": f"self/{i}", "source": "en-self", "winner": "x"} for i in range(2)]
    set_path = tmp_path / "judge-self.jsonl"
    make_sets.write_set([it for p in pairs for it in make_sets.judge_items(p)], set_path)
    d = tmp_path / "runs"
    d.mkdir()
    x_both = {"self/0/ab": "A", "self/0/ba": "B", "self/1/ab": "A", "self/1/ba": "B"}
    write_run(d, set_path, "kev-4b", x_both)
    write_run(d, set_path, "kev-9b", x_both)
    write_run(d, set_path, "winnow", {k: "B" if v == "A" else "A" for k, v in x_both.items()})
    text = "\n".join(judge_bias.section(d))
    assert "| kev-4b | qwen |" in text and "| 2/2 (100%" in text
    assert "| kev-4b | winnow | 2 | 2 | 0 | 0.500 |" in text
    assert "| kev-4b | kev-9b |" not in text
