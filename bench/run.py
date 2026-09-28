"""Smoke-test and benchmark local System One engines over the TypeSafe `/v1/systemone` wire format.

    uv run python bench/run.py --smoke --engine clm      # README example, CLM tolerances
    uv run python bench/run.py --smoke --engine ollaya   # README example, shape only
    uv run python bench/run.py --out bench/report.md     # questions.jsonl on clm and ollaya
    uv run python bench/run.py --engine kev --out bench/report.md --append   # add one engine's row

Both engines speak the same format, so one client serves both; only the base URL and model differ.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

ENGINES: dict[str, dict[str, Any]] = {
    "clm": {"url": os.environ.get("CLM_URL", "http://127.0.0.1:8700"), "model": None},
    "ollaya": {"url": os.environ.get("OLLAYA_URL", "http://127.0.0.1:11435"), "model": os.environ.get("OLLAYA_MODEL", "laya")},
    # One AnyJev adapter serves both correction levels; the model field picks the level.
    "anyjev-raw": {"url": os.environ.get("ANYJEV_URL", "http://127.0.0.1:8710"), "model": "anyjev-raw"},
    "anyjev-l0": {"url": os.environ.get("ANYJEV_URL", "http://127.0.0.1:8710"), "model": "anyjev-l0"},
    "kev": {"url": os.environ.get("KEV_URL", "http://127.0.0.1:8009"), "model": "kev-latest"},
}
# A bare benchmark runs the two engines that can share the machine. The others each need most of the Metal
# memory, so they are started, measured with --engine and --append, and stopped one at a time.
DEFAULT_ENGINES = ["clm", "ollaya"]

# The example from github.com/Contrastive-LM/CLM README, with its published CLM-8B answers (RTX 4090).
README_STATE = "Customer: my invoice was charged twice and nobody answers the phone!"
README_QUESTIONS = {
    "urgency": {"type": "noul", "instructions": "Is this urgent?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}

MAX_STATE_TOKENS = int(os.environ.get("MAX_MODEL_LEN", 2048))
WARMUP = 3
REPS = 3


class EngineError(RuntimeError):
    pass


def ask(engine: str, state: str, questions: dict, timeout: float = 120.0) -> tuple[dict, float]:
    """POST one request; return (answers, wall-clock ms). Raises EngineError on any failure."""
    cfg = ENGINES[engine]
    body: dict[str, Any] = {"state": state, "questions": questions}
    if cfg["model"]:
        body["model"] = cfg["model"]
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{cfg['url']}/v1/systemone", json=body, timeout=timeout)
    except requests.RequestException as e:
        raise EngineError(f"{engine}: cannot reach {cfg['url']}: {e}") from e
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise EngineError(f"{engine}: HTTP {r.status_code}: {r.text[:200]}")
    try:
        answers = r.json()["answers"]
    except (ValueError, KeyError) as e:
        raise EngineError(f"{engine}: response has no answers: {r.text[:200]}") from e
    check_shape(engine, questions, answers)
    return answers, ms


def check_shape(engine: str, questions: dict, answers: dict) -> None:
    """Every question must come back with the value field its type promises."""
    for qid, q in questions.items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            raise EngineError(f"{engine}: no answer for {qid!r}")
        kind, value = q["type"], a.get(q["type"])
        if kind == "choice":
            if not isinstance(value, str) or value not in q["criteria"]:
                raise EngineError(f"{engine}: {qid!r} chose {value!r}, not one of {list(q['criteria'])}")
            if not isinstance(a.get("probabilities"), dict):
                raise EngineError(f"{engine}: choice {qid!r} has no probabilities: {a}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EngineError(f"{engine}: answer {qid!r} has no numeric {kind!r}: {a}")


def smoke(engine: str) -> list[str]:
    """README example once; return failures (empty = pass)."""
    answers, ms = ask(engine, README_STATE, README_QUESTIONS)
    print(f"{engine}: {ms:.1f} ms  {json.dumps(answers)}")
    if engine != "clm":
        return []  # only CLM has published reference answers
    fails = []
    dept = answers["department"]
    if dept["choice"] != "billing" or dept["probabilities"].get("billing", 0.0) < 0.85:
        fails.append(f"department: want billing >= 0.85, got {dept['choice']} {dept['probabilities']}")
    if abs(answers["urgency"]["noul"] - 0.41) > 0.10:
        fails.append(f"urgency.noul {answers['urgency']['noul']:.3f} outside 0.41 +- 0.10")
    if abs(answers["frustration"]["score"] - 1.98) > 0.20:
        fails.append(f"frustration.score {answers['frustration']['score']:.3f} outside 1.98 +- 0.20")
    return fails


def correct(q: dict, answer: dict, expected: Any) -> bool:
    kind = q["type"]
    if kind == "noul":
        return (answer["noul"] >= 0.5) == bool(expected)
    if kind == "choice":
        return answer["choice"] == expected
    return math.floor(answer["score"] + 0.5) == int(expected)  # round() sends 0.5 to 0 and 2.5 to 2


def qwen_token_counter() -> Callable[[str], int]:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-8B", "tokenizer.json"))
    return lambda text: len(tok.encode(text).ids)


def embedded_texts(item: dict) -> list[str]:
    """What CLM sends to the encoder for each question: state, blank line, instructions (clm.schema.state_text)."""
    return [f"{item['state'].strip()}\n\n{q['instructions'].strip()}" for q in item["questions"].values()]


def load_questions(path: Path, count_tokens: Callable[[str], int]) -> list[dict]:
    """Reject over-long inputs up front: clm-serve truncates them silently (truncate_prompt_tokens), which
    would only show up as a lower score."""
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    too_long = [(it["id"], n) for it in items
                if (n := max(count_tokens(t) for t in embedded_texts(it))) > MAX_STATE_TOKENS]
    if too_long:
        raise SystemExit(f"inputs over {MAX_STATE_TOKENS} tokens: {too_long}")
    return items


@dataclass
class Result:
    engine: str
    pressure: list[str] = field(default_factory=list)
    cold_ms: float | None = None
    warm_ms: list[float] = field(default_factory=list)
    first_ms: list[float] = field(default_factory=list)  # first call per item: before any per-state cache hit
    right: int = 0
    graded: int = 0
    flips: int = 0       # choice answers that changed when the options were listed in reverse
    flip_items: int = 0  # choice answers compared that way
    errors: list[str] = field(default_factory=list)


def reversed_options(questions: dict) -> dict:
    """The same questions with every choice's options listed in reverse order."""
    return {qid: {**q, "criteria": dict(reversed(list(q["criteria"].items())))} if q["type"] == "choice" else q
            for qid, q in questions.items()}


def order_flip(engine: str, item: dict, answers: dict, res: Result) -> None:
    """Ask once more with reversed options and count the choice answers that changed.
    A failed call is recorded as an error and counts neither as a flip nor as a comparison."""
    choices = [qid for qid, q in item["questions"].items() if q["type"] == "choice"]
    if not choices:
        return
    try:
        flipped, _ = ask(engine, item["state"], reversed_options(item["questions"]))
    except EngineError as e:
        res.errors.append(f"{item['id']} reversed: {e}")
        return
    for qid in choices:
        res.flip_items += 1
        res.flips += flipped[qid]["choice"] != answers[qid]["choice"]


def bench(engine: str, items: list[dict], reps: int = REPS) -> Result:
    res = Result(engine, pressure=[memory_pressure()])
    first = items[0]
    # Cold call first (CLM embeds option texts once and caches them), then warm-up calls that are not timed.
    for i in range(1 + WARMUP):
        try:
            _, ms = ask(engine, first["state"], first["questions"])
        except EngineError as e:
            res.errors.append(str(e))
            return res  # an engine that fails its warm-up gets no numbers at all
        if i == 0:
            res.cold_ms = ms
    for it in items:
        first_answers = None
        for rep in range(reps):
            try:
                answers, ms = ask(engine, it["state"], it["questions"])
            except EngineError as e:
                res.errors.append(f"{it['id']}: {e}")
                continue  # a failed call adds neither a latency sample nor an answer
            res.warm_ms.append(ms)
            if rep == 0:
                res.first_ms.append(ms)
                first_answers = answers
            for qid, want in it["expected"].items():
                res.graded += 1
                res.right += correct(it["questions"][qid], answers[qid], want)
            res.pressure.append(memory_pressure())
        # After the timed reps, so the extra call does not warm their cache. An item whose rep 0 failed has no
        # reference answer and is left out of order-flip, as it is left out of first-call latency.
        if first_answers is not None:
            order_flip(engine, it, first_answers, res)
    return res


def pct(xs: list[float], p: float) -> float:
    return statistics.quantiles(xs, n=100, method="inclusive")[p - 1] if len(xs) > 1 else xs[0]


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    p = right / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


PRESSURE = {"1": "normal", "2": "warn", "4": "critical"}


def memory_pressure() -> str:
    """macOS kernel memory pressure level, the signal the success criterion is about."""
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({e})"
    return PRESSURE.get(out, f"unknown ({out!r})")


def report(results: list[Result], n_items: int, questions: str = "", reps: int = REPS) -> str:
    lines = [
        "# System One local benchmark",
        "",
        f"Questions: `{questions}`",
        f"{n_items} questions x {reps} reps, sequential, {WARMUP} warm-up calls excluded.",
        "`first-call p50` covers only each item's first call. Repeats of the same state can be served from an engine's",
        "embedding cache (CLM caches state vectors), so all-call percentiles understate uncached latency.",
        "Accuracy carries a 95% Wilson interval; overlapping intervals are not a difference.",
        "`order-flip` asks each item once more with every choice's options reversed and counts changed choices.",
        "",
        "| engine | cold ms | first-call p50 ms | all-call p50 ms | all-call p95 ms | samples | accuracy | order-flip | errors | worst memory pressure |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.warm_ms:
            p50, p95 = f"{pct(r.warm_ms, 50):.1f}", f"{pct(r.warm_ms, 95):.1f}"
        else:
            p50 = p95 = "failed"
        first = f"{pct(r.first_ms, 50):.1f}" if r.first_ms else "failed"
        cold = f"{r.cold_ms:.1f}" if r.cold_ms is not None else "failed"
        if r.graded:
            lo, hi = wilson(r.right, r.graded)
            acc = f"{r.right}/{r.graded} ({r.right / r.graded:.0%}, {lo * 100:.0f}-{hi * 100:.0f})"
        else:
            acc = "failed"
        if r.flip_items:
            flip = f"{r.flips}/{r.flip_items} ({r.flips / r.flip_items:.0%})"
        else:
            reversed_failed = any(" reversed: " in e for e in r.errors)
            flip = "failed" if not r.warm_ms or reversed_failed else "n/a"
        worst = max(r.pressure, key=list(PRESSURE.values()).index, default="unsampled") \
            if all(p in PRESSURE.values() for p in r.pressure) else ", ".join(sorted(set(r.pressure)))
        lines.append(f"| {r.engine} | {cold} | {first} | {p50} | {p95} | {len(r.warm_ms)} | {acc} | {flip} | {len(r.errors)} | {worst} |")
    for r in results:
        for e in r.errors[:5]:
            lines.append(f"- {r.engine} error: {e}")
    return "\n".join(lines) + "\n"


def _split(text: str) -> tuple[list[str], list[str], list[str]]:
    """(header through the table separator, table rows, error lines) of a report."""
    lines = text.rstrip("\n").splitlines()
    sep = next((i for i, line in enumerate(lines) if line.startswith("|---")), None)
    if sep is None:
        raise SystemExit("report has no results table; refusing to append")
    rows = [line for line in lines[sep + 1:] if line.startswith("| ")]
    errors = [line for line in lines[sep + 1:] if line.startswith("- ")]
    return lines[:sep + 1], rows, errors


def write_report(out: Path, results: list[Result], n_items: int, questions: str, append: bool,
                 reps: int = REPS) -> str:
    """Write the report; with append, merge into an existing report for the same question set, replacing the
    rows of engines measured again. Engines are measured one at a time, so each run adds its own rows."""
    text = report(results, n_items, questions, reps)
    if append and out.exists():
        header, rows, errors = _split(text)
        old_header, old_rows, old_errors = _split(out.read_text())
        if old_header != header:
            raise SystemExit(f"{out} has a different header (question set, item count, reps, or an older report "
                             "format); refusing to append. Write it once without --append first.")
        fresh = {r.engine for r in results}
        rows = [r for r in old_rows if r.split("|")[1].strip() not in fresh] + rows
        errors = [e for e in old_errors if e[2:].split(" error:")[0] not in fresh] + errors
        text = "\n".join(header + rows + errors) + "\n"
    out.write_text(text)
    return text


def questions_label(path: Path) -> str:
    """The question set as it appears in the report header: repo-relative when inside the repo."""
    path = path.resolve()
    root = Path(__file__).resolve().parents[1]
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--engine", choices=sorted(ENGINES), action="append")
    ap.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("report.md"))
    ap.add_argument("--append", action="store_true", help="merge into --out, replacing rows of re-measured engines")
    ap.add_argument("--reps", type=positive_int, default=REPS, help="timed calls per item (the public set uses 1)")
    args = ap.parse_args()
    engines = args.engine or DEFAULT_ENGINES

    if args.smoke:
        failed = False
        for engine in engines:
            try:
                fails = smoke(engine)
            except EngineError as e:
                fails = [str(e)]
            for f in fails:
                print(f"FAIL {engine}: {f}", file=sys.stderr)
            failed |= bool(fails)
        return 1 if failed else 0

    items = load_questions(args.questions, qwen_token_counter())
    results = [bench(engine, items, reps=args.reps) for engine in engines]
    print(write_report(args.out, results, len(items), questions_label(args.questions), args.append, args.reps))
    return 1 if any(r.errors for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
