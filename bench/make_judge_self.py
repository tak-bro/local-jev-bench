"""Build bench/data/judge-self.jsonl: the same questions answered by two model families, for self-preference.

    uv run python bench/make_judge_self.py --x qwen3:8b --y gemma3:4b    # Ollama on 127.0.0.1:11434

The questions are MT-Bench's first turns and the Korean-Human-Judgements instructions of judge-ko, at the revisions
bench/make_sets.py pins. Each is answered once by model x and once by model y through Ollama's /api/chat, greedy
(temperature 0, seed 0), thinking off for Qwen3, with the same length hint (LENGTH_HINT). An answer cut off at the token limit drops its pair, so no pair
reads as "complete against truncated". There is no human label: `expected` is the slot of x's answer, so a report's
accuracy on this set is how often an engine picks x. bench/judge_bias.py compares that rate across judge families
(self-preference = judges of x's family picking x more often than judges of another family on the same pairs).

Generation is not deterministic across Ollama versions or quantisations, so the set records both model digests in
each item's meta and the file's sha256 pins what was judged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import requests

import make_sets

OLLAMA = "http://127.0.0.1:11434"
MAX_TOKENS = 1024
# Appended to every generation prompt, the same for both models. Without it a third of the English and four in five
# Korean answers ran past the token limit (first attempt, 768 tokens), and the pairs that survived were the
# short-answer questions. The state judges read shows the question without it.
LENGTH_HINT = {"en": "\n\nAnswer in at most 250 words.", "ko": "\n\n500자 이내로 답하세요."}


def digest(url: str, model: str) -> str:
    tags = requests.get(f"{url}/api/tags", timeout=30).json()["models"]
    found = next((m["digest"] for m in tags if m["name"] == model), None)
    if found is None:
        raise SystemExit(f"{model}: not pulled on {url}; `ollama pull {model}` first")
    return found


def generate(url: str, model: str, prompt: str) -> str | None:
    """Greedy answer, or None when it hit MAX_TOKENS."""
    body = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": 0, "num_predict": MAX_TOKENS}}
    if model.startswith("qwen3"):
        body["think"] = False
    r = requests.post(f"{url}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    out = r.json()
    return None if out.get("done_reason") == "length" else out["message"]["content"]


def self_pairs(questions: list[tuple[str, str, str]], x: str, y: str, gen: Callable[[str, str], str | None],
               meta: dict) -> list[dict]:
    """(id, lang, question) -> pairs of x's and y's answers, winner x by convention (see the module docstring)."""
    pairs = []
    for qid, lang, q in questions:
        prompt = q + LENGTH_HINT[lang]
        ax, ay = gen(x, prompt), gen(y, prompt)
        print(f"{qid}: {'cut' if ax is None else len(ax)} / {'cut' if ay is None else len(ay)}", file=sys.stderr)
        if ax is None or ay is None:
            continue
        pairs.append({"id": f"self/{qid}", "lang": lang, "source": f"{lang}-self", "question": q, "x": ax, "y": ay,
                      "winner": "x", "meta": {"models": [x, y], **meta}})
    return pairs


def questions() -> list[tuple[str, str, str]]:
    """MT-Bench's 80 first turns and the Korean instructions behind judge-ko's sampled pairs."""
    mtb = make_sets.parquet(make_sets.MT_BENCH, "data/human-00000-of-00001-25f4910818759289.parquet")
    en = sorted({(r["question_id"], r["conversation_a"][0]["content"]) for r in mtb if r["turn"] == 1})
    ko_set = make_sets.read_jsonl(make_sets.DATA / "judge-ko.jsonl")
    ko_rows = make_sets.ko_pairs(make_sets.parquet(make_sets.KO_JUDGEMENTS, "data/train-00000-of-00001.parquet"))
    by_id = {p["id"]: p["question"] for p in ko_rows}
    ko_ids = sorted({it["meta"]["pair"] for it in ko_set})[:80]
    return [(f"mtb/{qid}", "en", q) for qid, q in en] + [(pid, "ko", by_id[pid]) for pid in ko_ids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="Ollama tag of the first family's model, e.g. qwen3:8b")
    ap.add_argument("--y", required=True, help="Ollama tag of the other family's model, e.g. gemma3:4b")
    ap.add_argument("--url", default=OLLAMA)
    ap.add_argument("--out", type=Path, default=make_sets.DATA / "judge-self.jsonl")
    args = ap.parse_args()
    if not (make_sets.DATA / "judge-ko.jsonl").exists():
        raise SystemExit("bench/data/judge-ko.jsonl missing; `uv run python bench/make_sets.py judge-ko` first")
    meta = {"digests": [digest(args.url, args.x), digest(args.url, args.y)]}
    count, limit = make_sets.token_counter()
    pairs = [p for p in self_pairs(questions(), args.x, args.y, lambda m, q: generate(args.url, m, q), meta)
             if make_sets.fits(p, count, limit)]
    items = [it for p in pairs for it in make_sets.judge_items(p)]
    print(f"{args.out}: {len(pairs)} pairs, sha256 {make_sets.write_set(items, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
