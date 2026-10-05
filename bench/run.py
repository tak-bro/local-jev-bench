"""Smoke-test and benchmark local System One engines over the TypeSafe `/v1/systemone` wire format.

    uv run python bench/run.py --smoke --engine clm      # README example, CLM tolerances
    uv run python bench/run.py --smoke --engine ollaya   # README example, shape only
    uv run python bench/run.py                           # questions.jsonl on clm and ollaya
    uv run python bench/run.py --engine kev-4b --questions bench/questions_ko.jsonl   # one engine, one set

Every engine speaks the same format, so one client serves all; only the base URL and model differ. Each run writes
every call to bench/runs/<set>/<engine>.jsonl (replacing that engine's earlier log when it finishes) and regenerates
bench/runs/<set>/report.md from all the logs there with bench/score.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

import requests

import score

ENGINES: dict[str, dict[str, Any]] = {
    "clm": {"url": os.environ.get("CLM_URL", "http://127.0.0.1:8700"), "model": None},
    "ollaya": {"url": os.environ.get("OLLAYA_URL", "http://127.0.0.1:11435"), "model": os.environ.get("OLLAYA_MODEL", "laya")},
    # One AnyJev adapter serves both correction levels; the model field picks the level.
    "anyjev-raw": {"url": os.environ.get("ANYJEV_URL", "http://127.0.0.1:8710"), "model": "anyjev-raw"},
    "anyjev-l0": {"url": os.environ.get("ANYJEV_URL", "http://127.0.0.1:8710"), "model": "anyjev-l0"},
    # Ollaya answers an unpulled tag with 404 MODEL_NOT_FOUND and refuses a request without `model`, so the
    # request's model pins what is measured on the shared port.
    "winnow": {"url": os.environ.get("OLLAYA_URL", "http://127.0.0.1:11435"),
               "model": os.environ.get("WINNOW_MODEL", "winnow:e4b")},
    "jeff": {"url": os.environ.get("JEFF_URL", "http://127.0.0.1:8765"), "model": "jeff-latest",
             "served": ("/health", ("model",), "jeff-qwen3.5-2b")},
}
# Kev serves one adapter at a time on one port (scripts/serve-kev.sh, KEV_RUN); `served` checks which one is up.
for size in ("0.8b", "4b", "9b"):
    ENGINES[f"kev-{size}"] = {"url": os.environ.get("KEV_URL", "http://127.0.0.1:8009"), "model": "kev-latest",
                              "served": ("/v1/models", ("models", 0, "run"), f"jaredpalmer/kev-{size}")}
# llama-server (scripts/serve-llama.sh) serves one Clef GGUF at a time on one port and names the file and its repo
# revision in its alias; `served` checks it.
for name, alias in (("clef-flash", "Clef-Flash-Q8_0@4a7a08c"), ("clef", "Clef-Q4_K_M@5f70656")):
    ENGINES[name] = {"url": os.environ.get("LLAMA_URL", "http://127.0.0.1:8020"), "model": alias,
                     "served": ("/v1/models", ("data", 0, "id"), alias)}
# Von's /health reports only its von-sdk version, so `served` checks that; the weights' revision is pinned in
# scripts/serve-von.sh, not checked here.
ENGINES["von"] = {"url": os.environ.get("VON_URL", "http://127.0.0.1:8030"), "model": "von-latest",
                  "served": ("/health", ("version",), "1.3.7")}
# A bare benchmark runs the two engines that can share the machine. The others each need most of the Metal
# memory, so they are started, measured with --engine, and stopped one at a time.
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


def served(engine: str) -> str | None:
    """What the engine's server says it runs, when the engine name alone does not pin it (several engines share a
    port). Raises EngineError when it is another model, so a log is never filed under the wrong name."""
    if "served" not in ENGINES[engine]:
        return None
    path, keys, want = ENGINES[engine]["served"]
    url = ENGINES[engine]["url"] + path
    try:
        got: Any = requests.get(url, timeout=30).json()
        for k in keys:
            got = got[k]
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        raise EngineError(f"{engine}: cannot read what {url} serves: {e}") from e
    if got != want:
        raise EngineError(f"{engine}: {url} serves {got!r}, not {want!r}; start the right model first")
    return got


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
            if set(a["probabilities"]) != set(q["criteria"]):
                raise EngineError(f"{engine}: choice {qid!r} has probabilities for {sorted(a['probabilities'])}, "
                                  f"not the options {sorted(q['criteria'])}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EngineError(f"{engine}: answer {qid!r} has no numeric {kind!r}: {a}")


def smoke(engine: str) -> list[str]:
    """README example once; return failures (empty = pass)."""
    served(engine)
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


def qwen_token_counter() -> Callable[[str], int]:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-8B", "tokenizer.json"))
    return lambda text: len(tok.encode(text).ids)


def embedded_texts(item: dict) -> list[str]:
    """What CLM sends to the encoder for each question: state, blank line, instructions (clm.schema.state_text)."""
    return [f"{item['state'].strip()}\n\n{q['instructions'].strip()}" for q in item["questions"].values()]


def render_state(v: Any, indent: int = 0) -> str:
    """A structured state as the text Kev's model reads (kev/api.py `render`): field names kept as labels in their
    order, nested objects indented, lists as "- " lines. Every engine gets this text, so a dict state reads the same
    to all of them and Kev sees what it would render itself."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render_state(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render_state(x, indent + 1)}" if isinstance(x, (dict, list))
                     else f"{pad}{k}: {render_state(x)}" for k, x in v.items())


def load_questions(path: Path, count_tokens: Callable[[str], int]) -> list[dict]:
    """Reject over-long inputs up front: clm-serve truncates them silently (truncate_prompt_tokens), which
    would only show up as a lower score."""
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for it in items:
        it["state"] = render_state(it["state"])
    too_long = [(it["id"], n) for it in items
                if (n := max(count_tokens(t) for t in embedded_texts(it))) > MAX_STATE_TOKENS]
    if too_long:
        raise SystemExit(f"inputs over {MAX_STATE_TOKENS} tokens: {too_long}")
    return items


def reversed_options(questions: dict) -> dict:
    """The same questions with every choice's options listed in reverse order."""
    return {qid: {**q, "criteria": dict(reversed(list(q["criteria"].items())))} if q["type"] == "choice" else q
            for qid, q in questions.items()}


def bench(engine: str, items: list[dict], raw: TextIO, reps: int = REPS) -> int:
    """Call the engine and log every call to `raw` as one JSON line (format: bench/score.py). Return the number of
    failed calls. Scoring is bench/score.py's job; nothing here grades an answer."""
    errors = 0

    def call(kind: str, item: dict, questions: dict, rep: int | None = None) -> dict | None:
        nonlocal errors
        try:
            answers, ms = ask(engine, item["state"], questions)
            error = None
        except EngineError as e:
            answers, ms, error = None, None, str(e)
            errors += 1
        raw.write(json.dumps({"item": item["id"], "call": kind, "rep": rep, "answers": answers, "error": error,
                              "ms": ms, "pressure": memory_pressure()}, ensure_ascii=False) + "\n")
        raw.flush()
        return answers

    first = items[0]
    # Cold call first (CLM embeds option texts once and caches them), then warm-up calls that are not timed.
    for i in range(1 + WARMUP):
        if call("cold" if i == 0 else "warmup", first, first["questions"]) is None:
            return errors  # an engine that fails its warm-up gets no numbers at all
    for it in items:
        answered = [call("timed", it, it["questions"], rep) for rep in range(reps)]
        # After the timed reps, so the extra call does not warm their cache. An item whose rep 0 failed has no
        # reference answer and is left out of order-flip, as it is left out of first-call latency.
        if answered[0] is not None and any(q["type"] == "choice" for q in it["questions"].values()):
            call("reversed", it, reversed_options(it["questions"]))
    return errors


PRESSURE = {"1": "normal", "2": "warn", "4": "critical"}


def memory_pressure() -> str:
    """macOS kernel memory pressure level, the signal the success criterion is about."""
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({e})"
    return PRESSURE.get(out, f"unknown ({out!r})")


def host() -> str:
    """Chip and memory, so a log says which machine measured it."""
    try:
        chip, mem = (subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=10).stdout.strip()
                     for key in ("machdep.cpu.brand_string", "hw.memsize"))
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({e})"
    return f"{chip}, {int(mem) // 2**30} GB" if chip and mem.isdigit() else f"unavailable ({chip!r}, {mem!r})"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    ap.add_argument("--runs", type=Path, default=Path(__file__).with_name("runs"),
                    help="raw logs go to <runs>/<questions file stem>/<engine>.jsonl")
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
    out = args.runs / args.questions.stem
    out.mkdir(parents=True, exist_ok=True)
    try:
        for engine in engines:  # all of them before any log is replaced, and each again right before its own
            served(engine)
    except EngineError as e:
        raise SystemExit(str(e)) from e
    errors = 0
    for engine in engines:
        try:
            identity = served(engine)
        except EngineError as e:
            raise SystemExit(str(e)) from e
        header = {"set": questions_label(args.questions), "set_sha256": score.sha256(args.questions),
                  "engine": engine, "model": ENGINES[engine]["model"], "reps": args.reps, "warmup": WARMUP,
                  "started": now(), "host": host(), "served": identity}
        log = out / f"{engine}.jsonl"
        partial = log.with_suffix(".jsonl.partial")  # not *.jsonl, so bench/score.py never reads it
        with partial.open("w") as raw:
            raw.write(json.dumps({"header": header}) + "\n")
            errors += bench(engine, items, raw, reps=args.reps)
            # Only a run that got here has a footer; bench/score.py refuses a log without one.
            raw.write(json.dumps({"footer": {"finished": now()}}) + "\n")
        partial.replace(log)  # an interrupted run leaves the previous log in place
    print(score.render(out), end="")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
