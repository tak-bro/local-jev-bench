"""Build the public question sets from pinned Hugging Face revisions into bench/data/ (gitignored: the sources carry
their own licenses). The same revision gives the same bytes, so a report's set sha256 says which set it measured.

    uv run python bench/make_sets.py transfer-v4       # Kev's out-of-distribution development set, 764 items
    uv run python bench/make_sets.py typed-decisions   # LocalLLaMA/typed-decisions test, 400 cases, 2,000 decisions
    uv run python bench/make_sets.py nsmc              # Korean movie reviews, positive or not, 300 of the test split
    uv run python bench/make_sets.py klue-ynat         # Korean news headlines, 7 topics, 300 of the validation split
    uv run python bench/make_sets.py clinc-oos         # CLINC150 20-way intents (300) plus out-of-scope (100 noul)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from score import sha256

DATA = Path(__file__).with_name("data")

KEV_SUITES = ("jaredpalmer/kev-suites", "a88f56db5341397299137cb68775c2ea6e3f68cb")
TYPED_DECISIONS = ("LocalLLaMA/typed-decisions", "d51d993547ad8355b1c25157fbc1fea0649e8ffa")
# e9t/nsmc ships a loading script; the Hub's parquet conversion of it is pinned instead.
NSMC = ("e9t/nsmc", "153263700c285ac0997b3e2a2d80b826aaf672d6")
KLUE = ("klue/klue", "349481ec73fff722f88e0453ca05c77a447d967c")
# CLINC150 in-scope rows come from the HF parquet mirror at a pinned revision; the intent names ride in the
# schema's ClassLabel metadata. Out-of-scope utterances are not in that mirror, so they come from the original
# authors' data_full.json at a pinned commit (master untouched since 2021).
CLINC = ("clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec")
CLINC_OOS_URL = ("https://raw.githubusercontent.com/clinc/oos-eval/"
                 "828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json")
CLINC_QUESTION = "What is the customer's intent?"
# Keys: ynat's ClassLabel names in label order (the parquet's huggingface metadata). Values: our own short descriptions,
# which KLUE does not provide; engines read them as the options' criteria.
YNAT_TOPICS = {"IT과학": "정보기술, 과학, 인터넷, 모바일", "경제": "경제, 금융, 산업, 부동산",
               "사회": "사건사고, 교육, 노동, 사회 일반", "생활문화": "생활, 문화, 건강, 연예, 여행",
               "세계": "국제, 해외 소식", "스포츠": "스포츠 경기와 선수", "정치": "정치, 국회, 행정, 외교, 북한"}


def hf_file(repo: str, path: str, revision: str) -> Path:
    """One file of a dataset repo at a pinned revision; a revision that no longer exists is an error."""
    return Path(hf_hub_download(repo_id=repo, filename=path, repo_type="dataset", revision=revision))


def write_set(items: list[dict], path: Path) -> str:
    """Write one item per line and return the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    return sha256(path)


def sample(rows: list, n: int, seed: int = 0) -> list:
    """n rows without replacement; the same rows, seed and n give the same sample."""
    return random.Random(seed).sample(rows, n)


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


def nsmc(rows: list[dict], n: int = 300) -> list[dict]:
    """NSMC test reviews -> one noul each: is the review positive (label 1)?"""
    rows = [r for r in rows if r["document"]]
    return [{"id": f"nsmc/{r['id']}", "state": r["document"],
             "questions": {"positive": {"type": "noul", "instructions": "이 리뷰는 긍정적인가?"}},
             "expected": {"positive": r["label"] == 1}, "meta": {"source": "nsmc"}}
            for r in sample(rows, n)]


def klue_ynat(rows: list[dict], n: int = 300) -> list[dict]:
    """KLUE-YNAT validation headlines (test labels are not public) -> one 7-way choice each: the headline's topic."""
    names = list(YNAT_TOPICS)
    return [{"id": r["guid"], "state": r["title"],
             "questions": {"topic": {"type": "choice", "instructions": "이 기사 제목의 분야는?", "criteria": YNAT_TOPICS}},
             "expected": {"topic": names[r["label"]]}, "meta": {"source": "klue-ynat"}}
            for r in sample(rows, n)]


def clinc_names(schema_metadata: bytes) -> list[str]:
    """The plus parquet's ClassLabel intent names, in label order."""
    return json.loads(schema_metadata)["info"]["features"]["intent"]["names"]


def clinc_oos(train: list[dict], test: list[dict], oos_test: list[str], names: list[str],
              k: int = 20, n: int = 300, m: int = 100, seed: int = 0) -> list[dict]:
    """CLINC150 top-k intents + out-of-scope utterances -> k-way choices and in-scope nouls.

    In-scope items read like BANKING77-20: the k intents most frequent in train, bare names with no
    descriptions, the test items of those intents shuffled with seed 0, the first n kept. Out-of-scope
    items get one noul each ("is this about one of the listed intents?", expected False), so the set
    grades OOS detection the way Cloudflare's CLINC150+OOS framing does. Winnow-style engines cap at
    64 options, hence k = 20 like the banking set."""
    top = sorted(c for c, _ in Counter(int(r["intent"]) for r in train).most_common(k))
    criteria = {names[c]: "" for c in top}
    in_scope = [r for r in test if int(r["intent"]) in set(top)]
    idx = list(range(len(in_scope)))
    random.Random(seed).shuffle(idx)
    items = [{"id": f"clinc/{i:03d}", "state": in_scope[j]["text"],
              "questions": {"intent": {"type": "choice", "instructions": CLINC_QUESTION, "criteria": criteria}},
              "expected": {"intent": names[int(in_scope[j]["intent"])]}, "meta": {"source": "clinc150"}}
             for i, j in enumerate(idx[:n])]
    oos_idx = list(range(len(oos_test)))
    random.Random(seed).shuffle(oos_idx)
    listed = ", ".join(criteria)
    items += [{"id": f"clinc-oos/{i:03d}", "state": oos_test[j],
               "questions": {"in_scope": {"type": "noul",
                                          "instructions": f"Is this request about one of these intents: {listed}?"}},
               "expected": {"in_scope": False}, "meta": {"source": "clinc-oos"}}
              for i, j in enumerate(oos_idx[:m])]
    return items


def clinc_oos_full() -> list[dict]:
    """Read both pinned sources and build the set."""
    import pyarrow.parquet as pq

    train_path = hf_file(CLINC[0], "plus/train-00000-of-00001.parquet", CLINC[1])
    names = clinc_names(pq.read_schema(train_path).metadata[b"huggingface"])
    train = pq.read_table(train_path).to_pylist()
    test = parquet(CLINC, "plus/test-00000-of-00001.parquet")
    with urllib.request.urlopen(CLINC_OOS_URL, timeout=120) as r:
        oos_test = [row[0] for row in json.loads(r.read().decode())["oos_test"]]
    return clinc_oos(train, test, oos_test, names)


def parquet(repo_rev: tuple[str, str], path: str) -> list[dict]:
    return pq.read_table(hf_file(repo_rev[0], path, repo_rev[1])).to_pylist()


SETS: dict[str, Callable[[], list[dict]]] = {
    "transfer-v4": lambda: transfer_v4(read_jsonl(hf_file(KEV_SUITES[0], "v4/transfer-v4/development.jsonl",
                                                          KEV_SUITES[1]))),
    "typed-decisions": lambda: typed_decisions(parquet(TYPED_DECISIONS, "all/test-00000-of-00001.parquet"),
                                               parquet(TYPED_DECISIONS, "all/train-00000-of-00001.parquet")),
    "nsmc": lambda: nsmc(parquet(NSMC, "default/test/0000.parquet")),
    "klue-ynat": lambda: klue_ynat(parquet(KLUE, "ynat/validation-00000-of-00001.parquet")),
    "clinc-oos": clinc_oos_full,
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
