"""Build the public question sets from pinned Hugging Face revisions into bench/data/ (gitignored: the sources carry
their own licenses). The same revision gives the same bytes, so a report's set sha256 says which set it measured.

    uv run python bench/make_sets.py transfer-v4    # Kev's out-of-distribution development set, 764 items
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

DATA = Path(__file__).with_name("data")

KEV_SUITES = ("jaredpalmer/kev-suites", "a88f56db5341397299137cb68775c2ea6e3f68cb")


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


SETS: dict[str, Callable[[], list[dict]]] = {
    "transfer-v4": lambda: transfer_v4(read_jsonl(hf_file(KEV_SUITES[0], "v4/transfer-v4/development.jsonl",
                                                          KEV_SUITES[1]))),
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
