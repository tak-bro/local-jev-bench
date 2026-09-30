"""bench/make_sets.py: set conversions on small hand-made rows, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
