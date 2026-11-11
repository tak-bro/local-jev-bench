"""bench/make_sets.py: set conversions on small hand-made rows, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import make_sets  # noqa: E402

# Shaped like jaredpalmer/kev-suites v4/transfer-v4/development.jsonl rows (fields trimmed).
TRANSFER_ROWS = [
    {"state": {"subject": "chemistry", "question": "Which is a noble gas?"},
     "questions": {"answer": {"type": "choice", "instructions": "Which option is right?",
                              "criteria": {"a": "Neon", "b": "Iron"}, "label": "a", "src": "mmlu"}},
     "_meta": {"id": "mmlu/test/1", "source": "mmlu", "variant": "clean", "row": 1}},
    {"state": "you are an idiot",
     "questions": {"offensive": {"type": "noul", "instructions": "Is this post offensive?",
                                 "criteria": {"true": "Insults", "false": "Not offensive"}, "label": True,
                                 "src": "tweet_offensive"}},
     "_meta": {"id": "tweet_offensive/test/7", "source": "tweet_offensive", "variant": "clean"}},
    {"state": {"policy": "On time by the deadline.", "case": "Filed a day early."},
     "questions": {"decision": {"type": "score", "instructions": "How late?", "criteria": ["On time", "Late"],
                                "label": 0, "src": "contrastive_deadline"}},
     "_meta": {"id": "composition_holdout/dev/3", "source": "composition_holdout", "variant": "permuted"}},
]


def test_transfer_v4_keeps_labels_out_of_the_engine_input():
    items = make_sets.transfer_v4(TRANSFER_ROWS)
    for it in items:
        for q in it["questions"].values():
            assert "label" not in q and "src" not in q
    assert [it["expected"] for it in items] == [{"answer": "a"}, {"offensive": True}, {"decision": 0}]


def test_transfer_v4_items():
    choice, noul, score_ = make_sets.transfer_v4(TRANSFER_ROWS)
    assert choice["id"] == "mmlu/test/1"
    assert choice["state"] == {"subject": "chemistry", "question": "Which is a noble gas?"}  # rendered by run.py
    assert noul["questions"]["offensive"]["criteria"] == {"true": "Insults", "false": "Not offensive"}
    assert score_["meta"] == {"source": "composition_holdout", "variant": "permuted"}
    # Kev grades every type by the argmax of the returned distribution (kev/metrics.py)
    assert all(it["scoring"] == {"score": "argmax"} for it in (choice, noul, score_))


def test_write_set_is_deterministic(tmp_path):
    items = make_sets.transfer_v4(TRANSFER_ROWS)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert make_sets.write_set(items, a) == make_sets.write_set(items, b)
    assert a.read_bytes() == b.read_bytes()
    assert json.loads(a.read_text().splitlines()[0])["id"] == "mmlu/test/1"


def test_hf_file_pins_the_revision(monkeypatch):
    seen = {}

    def fake_download(repo_id, filename, repo_type, revision):
        seen.update(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
        return "/cache/x"

    monkeypatch.setattr(make_sets, "hf_hub_download", fake_download)
    assert make_sets.hf_file("org/set", "a/b.jsonl", "abc123") == Path("/cache/x")
    assert seen == {"repo_id": "org/set", "filename": "a/b.jsonl", "repo_type": "dataset", "revision": "abc123"}


def test_cli_writes_the_set_and_prints_its_sha(monkeypatch, tmp_path, capsys):
    src = tmp_path / "development.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in TRANSFER_ROWS))
    monkeypatch.setattr(make_sets, "hf_file", lambda repo, path, revision: src)
    out = tmp_path / "data"
    monkeypatch.setattr(sys, "argv", ["make_sets.py", "transfer-v4", "--out", str(out)])
    assert make_sets.main() == 0
    written = out / "transfer-v4.jsonl"
    assert len(written.read_text().splitlines()) == 3
    assert make_sets.sha256(written) in capsys.readouterr().out


# Shaped like LocalLLaMA/typed-decisions all/{test,train} rows: state, questions and gold are JSON strings.
def td_row(i, split, gold):
    questions = {"urgent": {"type": "noul", "instructions": "Is it urgent?"},
                 "team": {"type": "choice", "instructions": "Which team?", "criteria": {"a": "A", "b": "B"}},
                 "risk": {"type": "score", "instructions": "How risky?", "criteria": ["Low", "Mid", "High"]}}
    return {"id": f"cs_{split}_{i}", "workflow": "customer_service", "split": split,
            "state": json.dumps({"ticket": f"t{i}"}), "questions": json.dumps(questions), "gold": json.dumps(gold)}


GOLD_A = {"urgent": {"type": "noul", "label": "true", "probabilities": {"false": 0.2, "true": 0.8}},
          "team": {"type": "choice", "label": "b", "probabilities": {"a": 0.4, "b": 0.6}},
          "risk": {"type": "score", "label": "2", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}}
GOLD_B = {"urgent": {"type": "noul", "label": "false", "probabilities": {"false": 0.6, "true": 0.4}},
          "team": {"type": "choice", "label": "a", "probabilities": {"a": 1.0, "b": 0.0}},
          "risk": {"type": "score", "label": "0", "probabilities": {"0": 0.5, "1": 0.3, "2": 0.2}}}


def test_typed_decisions_items():
    (item,) = make_sets.typed_decisions([td_row(0, "test", GOLD_A)], [td_row(1, "train", GOLD_A),
                                                                      td_row(2, "train", GOLD_B)])
    assert item["id"] == "cs_test_0" and item["state"] == {"ticket": "t0"}
    assert item["expected"] == {"urgent": True, "team": "b", "risk": 2}  # labels in the types score.py grades
    assert item["gold"]["risk"] == {"0": 0.1, "1": 0.2, "2": 0.7}
    # prior: each question's mean gold distribution over train, per workflow
    assert item["prior"]["urgent"] == {"false": 0.4, "true": 0.6}
    assert item["prior"]["team"] == {"a": 0.7, "b": 0.3}
    assert item["scoring"] == {"score": "argmax"} and item["meta"] == {"source": "customer_service"}
    assert set(item["questions"]) == {"urgent", "team", "risk"}


def test_sample_is_seeded_and_without_replacement():
    rows = list(range(100))
    a, b = make_sets.sample(rows, 10, seed=0), make_sets.sample(rows, 10, seed=0)
    assert a == b and len(set(a)) == 10
    assert make_sets.sample(rows, 10, seed=1) != a


def test_nsmc_items():
    rows = [{"id": "1", "document": "재밌다", "label": 1}, {"id": "2", "document": "지루함", "label": 0},
            {"id": "3", "document": "", "label": 1}, {"id": "4", "document": None, "label": 0}]
    items = make_sets.nsmc(rows, n=2)
    assert sorted(it["id"] for it in items) == ["nsmc/1", "nsmc/2"]  # empty reviews are never sampled
    it = next(x for x in items if x["id"] == "nsmc/2")
    assert it["state"] == "지루함" and it["expected"] == {"positive": False}
    assert it["questions"]["positive"] == {"type": "noul", "instructions": "이 리뷰는 긍정적인가?"}


def test_klue_ynat_items():
    rows = [{"guid": "ynat-v1_dev_00000", "title": "코스피 상승 마감", "label": 1},
            {"guid": "ynat-v1_dev_00001", "title": "손흥민 결승골", "label": 5}]
    items = make_sets.klue_ynat(rows, n=2)
    it = next(x for x in items if x["id"] == "ynat-v1_dev_00001")
    assert it["state"] == "손흥민 결승골" and it["expected"] == {"topic": "스포츠"}
    q = it["questions"]["topic"]
    assert q["type"] == "choice" and q["instructions"] == "이 기사 제목의 분야는?"
    # KLUE's ClassLabel names, in label order (ynat parquet metadata)
    assert list(q["criteria"]) == ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]


CLINC_NAMES = ["transfer", "translate", "oos"]


def clinc_rows() -> tuple[list[dict], list[dict]]:
    # `oos` (label 2) is the most frequent train label, as in CLINC150 plus
    train = [{"text": "t", "intent": i} for i in (0, 1, 2, 2, 2)]
    test = ([{"text": f"transfer {i}", "intent": 0} for i in range(3)]
            + [{"text": f"translate {i}", "intent": 1} for i in range(3)]
            + [{"text": f"weather {i}", "intent": 2} for i in range(3)])
    return train, test


def test_clinc_oos_never_lists_oos_as_an_intent():
    items = make_sets.clinc_oos(*clinc_rows(), CLINC_NAMES, k=2, n=3, m=2, seed=0)
    choices = [it for it in items if "intent" in it["expected"]]
    assert len(choices) == 3
    assert list(choices[0]["questions"]["intent"]["criteria"]) == ["transfer", "translate"]
    assert all(it["expected"]["intent"] != "oos" for it in choices)
    gate = next(it for it in items if "in_scope" in it["expected"])["questions"]["in_scope"]
    assert gate["type"] == "noul" and "oos" not in gate["instructions"]


def test_clinc_oos_nouls_are_balanced_and_unshared():
    items = make_sets.clinc_oos(*clinc_rows(), CLINC_NAMES, k=2, n=3, m=2, seed=0)
    nouls = [it for it in items if "in_scope" in it["expected"]]
    # an engine that always answers one way gets half the nouls
    assert sorted(it["expected"]["in_scope"] for it in nouls) == [False, False, True, True]
    assert all(it["state"].startswith("weather") for it in nouls if not it["expected"]["in_scope"])
    assert len({it["state"] for it in items}) == len(items)  # no utterance twice
    assert len({it["id"] for it in items}) == len(items)


def test_clinc_oos_refuses_too_few_rows():
    with pytest.raises(SystemExit, match="in-scope"):
        make_sets.clinc_oos(*clinc_rows(), CLINC_NAMES, k=2, n=5, m=2, seed=0)


MASSIVE_NAMES = ["alarm_set", "play_music", "general_quirky", "weather_query"]


def massive_rows() -> tuple[list[dict], list[dict]]:
    en = [{"id": "1", "intent": 0, "utt": "wake me at six"}, {"id": "2", "intent": 1, "utt": "play jazz"},
          {"id": "3", "intent": 2, "utt": "tell me a joke"}, {"id": "4", "intent": 1, "utt": "play rock"},
          {"id": "5", "intent": 3, "utt": "is it raining"}, {"id": "6", "intent": 1, "utt": "play pop"}]
    ko = [{"id": "1", "intent": 0, "utt": "여섯 시에 깨워 줘", "judgments": {"intent_score": [1, 1, 2]}},
          {"id": "2", "intent": 1, "utt": "재즈 틀어 줘", "judgments": {"intent_score": [1, 1, 1]}},
          {"id": "3", "intent": 2, "utt": "농담 해 줘", "judgments": {"intent_score": [1, 1, 1]}},
          {"id": "4", "intent": 1, "utt": "록 틀어", "judgments": {"intent_score": [1, 0, 1]}},
          {"id": "5", "intent": 3, "utt": "비 와", "judgments": {"intent_score": [1, 1, 1]}}]
    return en, ko


def test_massive_pairs_keep_parallel_rows_whose_intent_survived_translation():
    pairs = make_sets.massive_pairs(*massive_rows(), MASSIVE_NAMES, k=2, n=10)
    # play_music (3 rows) and alarm_set (1, ties weather_query, lower id) are the top 2; general_quirky is skipped,
    # id 4 has a judge's intent_score 0, id 6 has no ko-KR row
    assert sorted(en["id"] for en, _ in pairs) == ["1", "2"]
    assert all(en["id"] == ko["id"] for en, ko in pairs)


def test_massive_en_and_ko_differ_only_in_the_utterance():
    pairs = make_sets.massive_pairs(*massive_rows(), MASSIVE_NAMES, k=2, n=10)
    en, ko = make_sets.massive(pairs, MASSIVE_NAMES, "en"), make_sets.massive(pairs, MASSIVE_NAMES, "ko")
    assert [it["id"] for it in en] == [it["id"] for it in ko]
    for e, k in zip(en, ko):
        assert e["questions"] == k["questions"] and e["expected"] == k["expected"]
        assert e["state"] != k["state"]
    assert list(en[0]["questions"]["intent"]["criteria"]) == ["alarm_set", "play_music"]
    assert {it["state"] for it in ko} == {"여섯 시에 깨워 줘", "재즈 틀어 줘"}


def test_clinc_names_reads_classlabel_metadata():
    import json

    meta = json.dumps({"info": {"features": {"intent": {"names": CLINC_NAMES}}}}).encode()
    assert make_sets.clinc_names(meta) == CLINC_NAMES
