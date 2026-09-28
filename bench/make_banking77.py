# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.20", "pyarrow"]
# ///
"""Write bench/questions_banking77.jsonl: the BANKING77 20-way set AnyJev reports on, as bench/run.py items.

    uv run bench/make_banking77.py

It restates AnyJev's `banking20` task (bench/tasks/banking.py and Task.split in github.com/nokia-applied-research/
AnyJev): the 20 intents most frequent in train, listed by label id with underscores as spaces, the test items of
those intents shuffled with random.Random(0), the first 300 kept. The rows come from the same parquet files
`datasets.load_dataset("mteb/banking77")` reads, at a pinned revision; the repo's jsonl copies hold other rows.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

REPO = "mteb/banking77"
REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"
QUESTION = "What is the customer's intent?"


def build(train: list[dict], test: list[dict], k: int = 20, n: int = 300, seed: int = 0) -> list[dict]:
    names = {int(r["label"]): r["label_text"] for r in train}
    top = sorted(c for c, _ in Counter(int(r["label"]) for r in train).most_common(k))
    pretty = {c: names[c].replace("_", " ") for c in top}
    rows = [r for r in test if int(r["label"]) in pretty]
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    # No descriptions: every engine sees the bare intent names, as AnyJev's options are.
    criteria = {pretty[c]: "" for c in top}
    return [{"id": f"b{i:03d}", "state": rows[j]["text"],
             "questions": {"intent": {"type": "choice", "instructions": QUESTION, "criteria": criteria}},
             "expected": {"intent": pretty[int(rows[j]["label"])]}}
            for i, j in enumerate(idx[:n])]


def dumps(items: list[dict]) -> str:
    return "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)


def main() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def split(name: str) -> list[dict]:
        path = hf_hub_download(REPO, f"data/{name}-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
        return pq.read_table(path).to_pylist()

    out = Path(__file__).with_name("questions_banking77.jsonl")
    out.write_text(dumps(build(split("train"), split("test"))))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
