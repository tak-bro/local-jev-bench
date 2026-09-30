"""Build the public question sets from pinned Hugging Face revisions into bench/data/ (gitignored: the sources carry
their own licenses). The same revision gives the same bytes, so a report's set sha256 says which set it measured.

    uv run python bench/make_sets.py transfer-v4       # Kev's out-of-distribution development set, 764 items
    uv run python bench/make_sets.py typed-decisions   # LocalLLaMA/typed-decisions test, 400 cases, 2,000 decisions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATA = Path(__file__).with_name("data")

KEV_SUITES = ("jaredpalmer/kev-suites", "a88f56db5341397299137cb68775c2ea6e3f68cb")
TYPED_DECISIONS = ("LocalLLaMA/typed-decisions", "d51d993547ad8355b1c25157fbc1fea0649e8ffa")


def hf_file(repo: str, path: str, revision: str) -> Path:
    """One file of a dataset repo at a pinned revision; a revision that no longer exists is an error."""
    return Path(hf_hub_download(repo_id=repo, filename=path, repo_type="dataset", revision=revision))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_set(items: list[dict], path: Path) -> str:
    """Write one item per line and return the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    return sha256(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def transfer_v4(rows: list[dict]) -> list[dict]:
    """Kev's transfer-v4 records -> set items. Each question's `label` becomes `expected` and `src` is dropped: neither
    is model input. Kev grades every type by the argmax of the returned distribution (kev/metrics.py), so score
    questions use the argmax rule; the state stays a dict when it is one and run.py renders it."""
    items = []
    for r in rows:
        items.append({
            "id": r["_meta"]["id"],
            "state": r["state"],
            "questions": {qid: {k: v for k, v in q.items() if k not in ("label", "src")}
                          for qid, q in r["questions"].items()},
            "expected": {qid: q["label"] for qid, q in r["questions"].items()},
            "scoring": {"score": "argmax"},
            "meta": {"source": r["_meta"]["source"], "variant": r["_meta"]["variant"]},
        })
    return items


def typed_label(q: dict, label: str) -> Any:
    """The dataset writes every label as a string; score.py grades noul as bool and score as a level index."""
    return label == "true" if q["type"] == "noul" else int(label) if q["type"] == "score" else label


def typed_decisions(test: list[dict], train: list[dict]) -> list[dict]:
    """typed-decisions rows -> set items. `state` and `questions` are the request body; `gold` keeps each decision's
    soft label, `expected` its argmax label, and `prior` the question's mean gold distribution over train, per
    workflow. The leaderboard grades accuracy by argmax, so score questions use the argmax rule."""
    sums: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in train:
        for qid, g in json.loads(r["gold"]).items():
            counts[(r["workflow"], qid)] += 1
            for k, p in g["probabilities"].items():
                sums[(r["workflow"], qid)][k] += p
    items = []
    for r in test:
        questions, gold = json.loads(r["questions"]), json.loads(r["gold"])
        items.append({
            "id": r["id"],
            "state": json.loads(r["state"]),
            "questions": questions,
            "expected": {qid: typed_label(q, gold[qid]["label"]) for qid, q in questions.items()},
            "gold": {qid: gold[qid]["probabilities"] for qid in questions},
            "prior": {qid: {k: round(p / counts[(r["workflow"], qid)], 6) for k, p in sums[(r["workflow"], qid)].items()}
                      for qid in questions},
            "scoring": {"score": "argmax"},
            "meta": {"source": r["workflow"]},
        })
    return items


def parquet(repo_rev: tuple[str, str], path: str) -> list[dict]:
    return pq.read_table(hf_file(repo_rev[0], path, repo_rev[1])).to_pylist()


SETS: dict[str, Callable[[], list[dict]]] = {
    "transfer-v4": lambda: transfer_v4(read_jsonl(hf_file(KEV_SUITES[0], "v4/transfer-v4/development.jsonl",
                                                          KEV_SUITES[1]))),
    "typed-decisions": lambda: typed_decisions(parquet(TYPED_DECISIONS, "all/test-00000-of-00001.parquet"),
                                               parquet(TYPED_DECISIONS, "all/train-00000-of-00001.parquet")),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()
    path = args.out / f"{args.set}.jsonl"
    items = SETS[args.set]()
    print(f"{path}: {len(items)} items, sha256 {write_set(items, path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
