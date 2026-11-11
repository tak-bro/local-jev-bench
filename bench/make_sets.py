"""Build the public question sets from pinned Hugging Face revisions into bench/data/ (gitignored: the sources carry
their own licenses). The same revision gives the same bytes, so a report's set sha256 says which set it measured.

    uv run python bench/make_sets.py transfer-v4       # Kev's out-of-distribution development set, 764 items
    uv run python bench/make_sets.py typed-decisions   # LocalLLaMA/typed-decisions test, 400 cases, 2,000 decisions
    uv run python bench/make_sets.py nsmc              # Korean movie reviews, positive or not, 300 of the test split
    uv run python bench/make_sets.py klue-ynat         # Korean news headlines, 7 topics, 300 of the validation split
    uv run python bench/make_sets.py clinc-oos         # CLINC150 20-way intents (300), in scope or not (100 + 100 noul)
    uv run python bench/make_sets.py massive-en        # MASSIVE 20-way intents, 300 en-US utterances
    uv run python bench/make_sets.py massive-ko        # the same 300 utterances in ko-KR: same ids, options, question
    uv run python bench/make_sets.py judge-mtbench     # MT-Bench human pairwise judgments, 200 pairs x 2 orders
    uv run python bench/make_sets.py judge-ko          # Korean-Human-Judgements, 200 pairs x 2 orders
    uv run python bench/make_sets.py judge-length      # one answer against itself padded, 100 pairs x 2 orders

The judge sets ask an engine which of two responses is better. Every pair appears twice, once in each order (meta
`variant` ab / ba), so bench/judge_bias.py can tell a judgement that follows the content from one that follows the
position.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
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
# CLINC150 rows come from the HF parquet mirror at a pinned revision; the intent names ride in the schema's
# ClassLabel metadata. The `plus` config also holds the out-of-scope utterances, under the label `oos`.
CLINC = ("clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec")
CLINC_OOS_LABEL = "oos"
CLINC_QUESTION = "What is the customer's intent?"
# Amazon MASSIVE (CC-BY-4.0): one set of utterances localised into 51 languages, a row's `id` the same utterance in
# every locale. The Hub's parquet conversion of the loading-script repo is pinned.
MASSIVE = ("AmazonScience/massive", "ed58ac423a2f4121720918bf5301577edce4ffd3")
MASSIVE_QUESTION = "What is the user's intent?"
# A catch-all for chit-chat, not an intent a router would act on.
MASSIVE_SKIP = {"general_quirky"}
# Keys: ynat's ClassLabel names in label order (the parquet's huggingface metadata). Values: our own short descriptions,
# which KLUE does not provide; engines read them as the options' criteria.
YNAT_TOPICS = {"IT과학": "정보기술, 과학, 인터넷, 모바일", "경제": "경제, 금융, 산업, 부동산",
               "사회": "사건사고, 교육, 노동, 사회 일반", "생활문화": "생활, 문화, 건강, 연예, 여행",
               "세계": "국제, 해외 소식", "스포츠": "스포츠 경기와 선수", "정치": "정치, 국회, 행정, 외교, 북한"}
# Pairwise human judgments. MT-Bench: CC-BY-4.0, expert and author votes on six models' answers to 80 questions.
# Korean-Human-Judgements: no license on the card, marked evaluation only; never redistributed, built locally.
MT_BENCH = ("lmsys/mt_bench_human_judgments", "f7d2896d2cc5d80f8b55c2bbc722613555233c25")
KO_JUDGEMENTS = ("HAERAE-HUB/Korean-Human-Judgements", "24dae25b0f4c0d3c0d3a10f2997f51cf6eb6cd5f")
JUDGE_PROMPTS = {
    "en": {"labels": ("Question", "Response A", "Response B"),
           "question": {"type": "choice", "instructions": "Which response answers the question better?",
                        "criteria": {"A": "Response A", "B": "Response B"}}},
    "ko": {"labels": ("질문", "답변 A", "답변 B"),
           "question": {"type": "choice", "instructions": "어느 답변이 질문에 더 잘 답했는가?",
                        "criteria": {"A": "답변 A", "B": "답변 B"}}},
}
# Appended by `pad`: a restatement of the answer's first sentence and a closing courtesy. Neither adds a fact, so a
# judge that prefers the padded copy prefers length (and politeness), not content.
PAD = {"en": ("In short: {first}", "I hope this answer is helpful. Let me know if you have any other questions or "
                                   "would like more detail on any part of it."),
       "ko": ("요약하면: {first}", "도움이 되셨기를 바랍니다. 다른 궁금한 점이 있거나 더 자세한 설명이 필요하시면 "
                               "언제든지 말씀해 주세요.")}


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


def clinc_oos(train: list[dict], test: list[dict], names: list[str],
              k: int = 20, n: int = 300, m: int = 100, seed: int = 0) -> list[dict]:
    """CLINC150 k intents + out-of-scope utterances -> k-way choices and balanced in-scope nouls.

    The k intents are drawn with seed 0 from the 150 real ones; `oos` is never one of them (every real intent has
    the same train count and `oos` more, so "the k most frequent" picked `oos` as a bare option). Options are the
    bare names, no descriptions, like BANKING77-20; Winnow-style engines cap at 64 options, hence k = 20. The test
    rows of those intents are shuffled with seed 0: the first n become choices, the next m in-scope nouls
    (expected True). m out-of-scope test rows, shuffled with seed 0, become nouls expected False. No utterance is
    used twice, and an engine that always answers one way scores half of the nouls."""
    oos = names.index(CLINC_OOS_LABEL)
    real = sorted({int(r["intent"]) for r in train} - {oos})
    picked = sorted(random.Random(seed).sample(real, k))
    criteria = {names[c]: "" for c in picked}
    in_scope = [r for r in test if int(r["intent"]) in set(picked)]
    random.Random(seed).shuffle(in_scope)
    out_scope = [r for r in test if int(r["intent"]) == oos]
    random.Random(seed).shuffle(out_scope)
    if len(in_scope) < n + m or len(out_scope) < m:
        raise SystemExit(f"clinc-oos: {len(in_scope)} in-scope and {len(out_scope)} out-of-scope test rows, "
                         f"want {n + m} and {m}")
    listed = ", ".join(criteria)
    gate = {"in_scope": {"type": "noul", "instructions": f"Is this request about one of these intents: {listed}?"}}
    items = [{"id": f"clinc/{i:03d}", "state": r["text"],
              "questions": {"intent": {"type": "choice", "instructions": CLINC_QUESTION, "criteria": criteria}},
              "expected": {"intent": names[int(r["intent"])]}, "meta": {"source": "clinc150"}}
             for i, r in enumerate(in_scope[:n])]
    items += [{"id": f"clinc-in/{i:03d}", "state": r["text"], "questions": gate,
               "expected": {"in_scope": True}, "meta": {"source": "clinc-in"}}
              for i, r in enumerate(in_scope[n:n + m])]
    items += [{"id": f"clinc-oos/{i:03d}", "state": r["text"], "questions": gate,
               "expected": {"in_scope": False}, "meta": {"source": "clinc-oos"}}
              for i, r in enumerate(out_scope[:m])]
    return items


def clinc_oos_full() -> list[dict]:
    """Read the pinned `plus` config and build the set."""
    train_path = hf_file(CLINC[0], "plus/train-00000-of-00001.parquet", CLINC[1])
    names = clinc_names(pq.read_schema(train_path).metadata[b"huggingface"])
    train = pq.read_table(train_path).to_pylist()
    return clinc_oos(train, parquet(CLINC, "plus/test-00000-of-00001.parquet"), names)


def massive_pairs(en: list[dict], ko: list[dict], names: list[str], k: int = 20, n: int = 300,
                  seed: int = 0) -> list[tuple[dict, dict]]:
    """Parallel MASSIVE test rows, (en-US, ko-KR) with the same id, for the k intents most frequent in en-US test
    (after MASSIVE_SKIP; ties by label id). A pair is kept only if both rows carry the same intent and no MASSIVE
    judge marked the ko-KR row's intent as lost in translation (intent_score 0). The pairs, in id order, are
    shuffled with seed 0 and the first n kept."""
    count = Counter(int(r["intent"]) for r in en if names[int(r["intent"])] not in MASSIVE_SKIP)
    top = {c for c, _ in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}
    by_id = {r["id"]: r for r in ko}
    pairs = [(r, by_id[r["id"]]) for r in en if int(r["intent"]) in top and r["id"] in by_id
             and int(by_id[r["id"]]["intent"]) == int(r["intent"])
             and 0 not in ((by_id[r["id"]].get("judgments") or {}).get("intent_score") or [])]
    pairs.sort(key=lambda p: int(p[0]["id"]))
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def massive(pairs: list[tuple[dict, dict]], names: list[str], lang: str) -> list[dict]:
    """One k-way choice per pair, the utterance in `lang` (en or ko). Both sets share ids, options and the English
    question, so the only thing that differs between them is the utterance's language."""
    criteria = {names[c]: "" for c in sorted({int(en["intent"]) for en, _ in pairs})}
    side = {"en": 0, "ko": 1}[lang]
    return [{"id": f"massive/{p[0]['id']}", "state": p[side]["utt"],
             "questions": {"intent": {"type": "choice", "instructions": MASSIVE_QUESTION, "criteria": criteria}},
             "expected": {"intent": names[int(p[0]["intent"])]}, "meta": {"source": f"massive-{lang}"}}
            for p in pairs]


def massive_full(lang: str) -> list[dict]:
    """Read both pinned locales' test splits and build one language's set."""
    en_path = hf_file(MASSIVE[0], "en-US/test/0000.parquet", MASSIVE[1])
    names = clinc_names(pq.read_schema(en_path).metadata[b"huggingface"])
    pairs = massive_pairs(pq.read_table(en_path).to_pylist(), parquet(MASSIVE, "ko-KR/test/0000.parquet"), names)
    return massive(pairs, names, lang)


def judge_state(lang: str, question: str, a: str, b: str) -> str:
    q, ra, rb = JUDGE_PROMPTS[lang]["labels"]
    return f"{q}:\n{question.strip()}\n\n{ra}:\n{a.strip()}\n\n{rb}:\n{b.strip()}"


def judge_items(pair: dict) -> list[dict]:
    """One pair -> two items, `ab` with response x as A and `ba` with it as B. `expected` is the position the human
    winner (`pair["winner"]`, "x" or "y") holds in that order; `meta.lengths` the characters of A and B."""
    items = []
    for variant, (a, b) in (("ab", (pair["x"], pair["y"])), ("ba", (pair["y"], pair["x"]))):
        x_is_a = variant == "ab"
        items.append({
            "id": f"{pair['id']}/{variant}",
            "state": judge_state(pair["lang"], pair["question"], a, b),
            "questions": {"better": JUDGE_PROMPTS[pair["lang"]]["question"]},
            "expected": {"better": "A" if (pair["winner"] == "x") == x_is_a else "B"},
            "meta": {"source": pair["source"], "variant": variant, "pair": pair["id"],
                     "lengths": [len(a.strip()), len(b.strip())],**pair.get("meta", {})},
        })
    return items


def fits(pair: dict, count_tokens: Callable[[str], int], limit: int) -> bool:
    """Whether the pair's state plus instructions stays within the engines' input limit (run.py refuses longer)."""
    state = judge_state(pair["lang"], pair["question"], pair["x"], pair["y"])
    return count_tokens(f"{state}\n\n{JUDGE_PROMPTS[pair['lang']]['question']['instructions']}") <= limit


def mt_bench_pairs(rows: list[dict]) -> list[dict]:
    """MT-Bench human votes -> first-turn pairs with a majority winner. Several judges often rated the same pair, in
    either orientation; a pair is kept when one model has more than half its votes (ties count against both).
    x is the answer of the model whose name sorts first."""
    votes: dict[tuple[int, str, str], Counter] = defaultdict(Counter)
    text: dict[tuple[int, str, str], tuple[str, dict[str, str]]] = {}
    for r in rows:
        if r["turn"] != 1:
            continue
        m1, m2 = sorted((r["model_a"], r["model_b"]))
        key = (r["question_id"], m1, m2)
        votes[key][{"model_a": r["model_a"], "model_b": r["model_b"]}.get(r["winner"], "tie")] += 1
        text[key] = (r["conversation_a"][0]["content"], {r["model_a"]: r["conversation_a"][1]["content"],
                                                          r["model_b"]: r["conversation_b"][1]["content"]})
    pairs = []
    for key in sorted(votes):
        qid, m1, m2 = key
        v, n = votes[key], sum(votes[key].values())
        winner = next((m for m in (m1, m2) if v[m] * 2 > n), None)
        if winner is None:
            continue
        question, answers = text[key]
        pairs.append({"id": f"mtb/{qid}/{m1}-vs-{m2}", "lang": "en", "source": "mt-bench", "question": question,
                      "x": answers[m1], "y": answers[m2], "winner": "x" if winner == m1 else "y",
                      "meta": {"models": [m1, m2], "votes": n, "winner_votes": v[winner]}})
    return pairs


def ko_pairs(rows: list[dict]) -> list[dict]:
    """Korean-Human-Judgements rows -> pairs, ties dropped; x is the dataset's response_a."""
    return [{"id": f"ko/{i:03d}", "lang": "ko", "source": r["source"], "question": r["instruction"],
             "x": r["response_a"], "y": r["response_b"], "winner": "x" if r["decision"] == "A" else "y"}
            for i, r in enumerate(rows) if r["decision"] in ("A", "B")]


def first_sentence(text: str, cap: int = 200) -> str:
    line = text.strip().split("\n", 1)[0]
    m = re.search(r"[.!?。](\s|$)", line)
    return (line[:m.end()] if m else line)[:cap].strip()


def pad(text: str, lang: str) -> str:
    """The same answer, longer: its first sentence restated and a closing courtesy appended (see PAD)."""
    restate, close = PAD[lang]
    return f"{text.strip()}\n\n{restate.format(first=first_sentence(text))}\n\n{close}"


def length_pairs(pools: list[list[dict]], n_each: int, count_tokens: Callable[[str], int], limit: int) -> list[dict]:
    """From each pool, the human winner of n_each sampled pairs against its own padded copy; the original (x) is
    expected to win, since the copy adds no content."""
    out = []
    for pool in pools:
        cands = [{"id": f"len/{p['id']}", "lang": p["lang"], "source": f"{p['source']}-length",
                  "question": p["question"], "x": (w := p[p["winner"]]), "y": pad(w, p["lang"]), "winner": "x"}
                 for p in pool]
        out += sample([c for c in cands if fits(c, count_tokens, limit)], n_each, seed=1)
    return out


def judge_set(pairs: list[dict], n: int, count_tokens: Callable[[str], int], limit: int, seed: int = 0) -> list[dict]:
    """n pairs that fit, sampled with the seed, each as its two orders."""
    kept = [p for p in pairs if fits(p, count_tokens, limit)]
    return [it for p in sample(kept, n, seed) for it in judge_items(p)]


def token_counter() -> tuple[Callable[[str], int], int]:
    """run.py's Qwen3 tokenizer count and input limit, so a built set always passes run.py's length check."""
    import run

    return run.qwen_token_counter(), run.MAX_STATE_TOKENS


def judge_sets(name: str) -> list[dict]:
    count, limit = token_counter()
    mtb = mt_bench_pairs(parquet(MT_BENCH, "data/human-00000-of-00001-25f4910818759289.parquet"))
    ko = ko_pairs(parquet(KO_JUDGEMENTS, "data/train-00000-of-00001.parquet"))
    if name == "judge-mtbench":
        return judge_set(mtb, 200, count, limit)
    if name == "judge-ko":
        return judge_set(ko, 200, count, limit)
    return [it for p in length_pairs([mtb, ko], 50, count, limit) for it in judge_items(p)]


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
    "massive-en": lambda: massive_full("en"),
    "massive-ko": lambda: massive_full("ko"),
    "judge-mtbench": lambda: judge_sets("judge-mtbench"),
    "judge-ko": lambda: judge_sets("judge-ko"),
    "judge-length": lambda: judge_sets("judge-length"),
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
