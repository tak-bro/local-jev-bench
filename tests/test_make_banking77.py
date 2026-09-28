"""Offline checks for bench/make_banking77.py: fake dataset rows, no network."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import make_banking77 as mb  # noqa: E402

TRAIN = [{"text": f"t{i}", "label": lab, "label_text": f"intent_{lab}"}
         for i, lab in enumerate([5, 5, 5, 2, 2, 9, 9, 9, 9, 7])]
TEST = [{"text": f"q{i}", "label": lab, "label_text": f"intent_{lab}"}
        for i, lab in enumerate([9, 7, 5, 2, 5, 9, 2, 5, 9, 5, 7, 5])]


def anyjev_reference(train, test, k, n, seed):
    """AnyJev bench/tasks/banking.py `load` and bench/tasks/base.py `Task.split`, restated over plain rows."""
    names = {int(r["label"]): r["label_text"] for r in train}
    top = sorted(c for c, _ in Counter(int(r["label"]) for r in train).most_common(k))
    pretty = [names[c].replace("_", " ") for c in top]
    remap = {c: i for i, c in enumerate(top)}
    items = [(r["text"], remap[int(r["label"])]) for r in test if int(r["label"]) in remap]
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    return pretty, [items[i] for i in idx[:n]]


def test_build_matches_anyjev_banking20_split():
    pretty, want = anyjev_reference(TRAIN, TEST, k=2, n=4, seed=0)
    items = mb.build(TRAIN, TEST, k=2, n=4, seed=0)
    assert pretty == ["intent 5", "intent 9"]  # top 2 by train frequency, listed by label id
    assert [(it["state"], pretty.index(it["expected"]["intent"])) for it in items] == want
    for it in items:
        q = it["questions"]["intent"]
        assert q["type"] == "choice" and list(q["criteria"]) == pretty
        assert it["expected"]["intent"] in q["criteria"]
    assert len({it["id"] for it in items}) == len(items)


def test_build_output_is_pinned():
    # Pinned output for seed 0: a change to the selection or the shuffle shows up here, not only as a mismatch with
    # the restated reference above.
    items = mb.build(TRAIN, TEST, k=2, n=4, seed=0)
    assert [(it["state"], it["expected"]["intent"]) for it in items] == [
        ("q7", "intent 5"), ("q2", "intent 5"), ("q8", "intent 9"), ("q4", "intent 5")]
    assert mb.dumps(items).count("\n") == 4 and mb.dumps(items).startswith('{"id": "b000", "state": "q7"')


def test_shipped_banking77_set():
    lines = (ROOT / "bench" / "questions_banking77.jsonl").read_text().splitlines()
    items = [json.loads(line) for line in lines]
    assert len(items) == 300
    options = {tuple(it["questions"]["intent"]["criteria"]) for it in items}
    assert len(options) == 1 and len(next(iter(options))) == 20
    assert all(it["expected"]["intent"] in it["questions"]["intent"]["criteria"] for it in items)
