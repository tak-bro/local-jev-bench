"""Compare OpenAI-compatible chat servers on the same prompts: vllm-metal vs Ollama, same Qwen3-8B.

    uv run python bench/llm_compare.py --name vllm-4bit --url http://127.0.0.1:8092 --model qwen3-8b-4bit
    uv run python bench/llm_compare.py --name ollama-q4 --url http://127.0.0.1:11434 --model qwen3:8b

Each run appends one row to bench/llm_report.md. Measures, per server:
- single-request time to first token (TTFT) and decode speed, streaming, sequential
- aggregate throughput with N concurrent requests
Thinking is turned off so both servers generate comparable answers: vllm-metal honours Qwen3's `/no_think`
switch, but Ollama's OpenAI endpoint ignores it (and `think: false`) and needs `--extra '{"reasoning_effort": "none"}'`.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import subprocess
import time
from pathlib import Path

import requests

PROMPTS = [
    "Explain in about 150 words why the sky is blue.",
    "Write a short Python function that checks whether a string is a palindrome, then explain it briefly.",
    "Summarise the pros and cons of remote work in about 150 words.",
    "Describe how a hash map works to a junior developer, in about 150 words.",
    "Give five tips for writing clear commit messages, one sentence each.",
    "Explain the difference between TCP and UDP in about 150 words.",
    "Write a haiku about autumn, then explain its imagery in two sentences.",
    "What is unified memory on Apple Silicon and why does it matter for local LLMs? About 150 words.",
]
MAX_TOKENS = 256
EXTRA: dict = {}  # per-server request fields, from --extra


def body(model: str, prompt: str, stream: bool) -> dict:
    return {"model": model, "stream": stream, "temperature": 0, "max_tokens": MAX_TOKENS,
            **({"stream_options": {"include_usage": True}} if stream else {}), **EXTRA,
            "messages": [{"role": "user", "content": prompt + " /no_think"}]}


def stream_one(url: str, model: str, prompt: str) -> dict:
    """Streaming call: TTFT = first non-empty content delta; decode tok/s over the rest."""
    t0 = time.perf_counter()
    first = None
    tokens = None
    chunks = 0
    with requests.post(f"{url}/v1/chat/completions", json=body(model, prompt, True), stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                tokens = d["usage"]["completion_tokens"]
            for c in d.get("choices", []):
                if (c.get("delta") or {}).get("content"):
                    chunks += 1
                    if first is None:
                        first = time.perf_counter()
    end = time.perf_counter()
    n = tokens if tokens is not None else chunks  # prefer the server's own token count
    ttft = (first or end) - t0
    decode = (n - 1) / (end - first) if first and n > 1 and end > first else float("nan")
    return {"ttft_ms": ttft * 1000, "decode_tps": decode, "tokens": n, "counted_by": "usage" if tokens else "chunks"}


def one_blocking(url: str, model: str, prompt: str) -> int:
    r = requests.post(f"{url}/v1/chat/completions", json=body(model, prompt, False), timeout=900)
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


def pressure() -> str:
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True).stdout.strip()
    return {"1": "normal", "2": "warn", "4": "critical"}.get(out, out)


def free_pct() -> str:
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout.strip().splitlines()
    return out[-1].split(":")[-1].strip() if out else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--extra", type=json.loads, default={}, help="JSON object merged into every request body")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("llm_report.md"))
    args = ap.parse_args()
    EXTRA.update(args.extra)

    stream_one(args.url, args.model, "Say hi. /no_think")  # load + warm-up, not timed
    singles = [stream_one(args.url, args.model, p) for p in PROMPTS]
    mem_single = free_pct()

    prompts = (PROMPTS * ((args.concurrency // len(PROMPTS)) + 1))[: args.concurrency]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        toks = list(ex.map(lambda p: one_blocking(args.url, args.model, p), prompts))
    wall = time.perf_counter() - t0
    press = pressure()

    ttft = statistics.median(s["ttft_ms"] for s in singles)
    dec = statistics.median(s["decode_tps"] for s in singles)
    agg = sum(toks) / wall
    counted = {s["counted_by"] for s in singles}
    row = (f"| {args.name} | {ttft:.0f} | {dec:.1f} | {agg:.1f} | {wall:.1f} | {sum(toks)} | "
           f"{mem_single} | {press} | {','.join(sorted(counted))} |")
    if not args.out.exists():
        args.out.write_text(
            "# vllm-metal vs Ollama, Qwen3-8B chat\n\n"
            f"{len(PROMPTS)} prompts, max_tokens={MAX_TOKENS}, temperature 0, `/no_think`. "
            "Single = sequential streaming; concurrent = parallel non-streaming.\n\n"
            "| server | single TTFT p50 ms | single decode tok/s p50 | concurrent tok/s | concurrent wall s | "
            "concurrent tokens | free mem after single | pressure after concurrent | token count source |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    with args.out.open("a") as f:
        f.write(row + "\n")
    print(row)


if __name__ == "__main__":
    main()
