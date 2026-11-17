# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.8.0", "transformers==5.17.0", "peft==0.21.0", "safetensors", "huggingface_hub"]
# ///
"""Serve AutoTrust's JEV-9B over TypeSafe's `POST /v1/systemone` wire format, so bench/run.py can measure it.

    uv run serve/jev_server.py --port 8110 --model <local snapshot of autotrust/JEV-9B> --name JEV-9B@7ac669b

JEV-9B ships no System One server: its model card runs one decision per prompt, either through vLLM (CUDA) or plain
`transformers` + `peft`. This adapter is that `transformers` path on Metal (MPS): the System 1 LoRA merged into the
Qwen3.5-9B backbone, the last token's final-norm hidden state through the 24-slot fp32 head, a per-kind temperature
and a softmax over the kind's slots, as in the card's "Plain transformers + peft" example. One forward pass per
question.

Mapping the wire format onto JEV's three kinds:
- noul: options false/true; the answer is P(true).
- choice: each option is shown as "A) <key>: <description>". JEV has 16 choice slots, so a question with more than 16
  options is refused with 422 TOO_MANY_OPTIONS (as JevK5 refuses), not truncated.
- score: JEV answers on a fixed 0-5 scale. The question names the levels ("... on a 0-2 scale: 0 = Calm, 1 = ..."),
  the prompt lists 0-5 as the card requires, and the softmax runs over the first n slots only, so every probability
  is on one of the n levels. More than 6 levels is refused the same way.
"""

from __future__ import annotations

import argparse
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_OPTIONS = {"choice": len(LETTERS), "score": 6}


class BadRequest(ValueError):
    pass


class TooManyOptions(ValueError):
    pass


def as_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state, ensure_ascii=False)
    raise BadRequest(f"state must be a string or JSON, got {state!r}")


def to_prompt(state: str, qid: str, q: dict) -> tuple[str, str, list]:
    """(JEV kind, prompt, the wire-format labels its first slots map back to) for one question."""
    kind, text = q.get("type"), q.get("instructions", "")
    if kind == "noul":
        kind_, lines, labels = "noul", ["false", "true"], [False, True]
    elif kind == "choice":
        criteria = q.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise BadRequest(f"question {qid!r}: a choice needs at least 2 criteria, got {criteria!r}")
        if len(criteria) > MAX_OPTIONS["choice"]:
            raise TooManyOptions(f"question {qid!r}: {len(criteria)} options, JEV-9B answers at most "
                                 f"{MAX_OPTIONS['choice']}")
        labels = list(criteria)
        lines = [f"{LETTERS[i]}) {k}: {d}" if d else f"{LETTERS[i]}) {k}" for i, (k, d) in enumerate(criteria.items())]
        kind_ = "choice"
    elif kind == "score":
        levels = q.get("criteria")
        if not isinstance(levels, list) or len(levels) < 2:
            raise BadRequest(f"question {qid!r}: a score needs at least 2 levels, got {levels!r}")
        if len(levels) > MAX_OPTIONS["score"]:
            raise TooManyOptions(f"question {qid!r}: {len(levels)} levels, JEV-9B answers at most "
                                 f"{MAX_OPTIONS['score']}")
        scale = ", ".join(f"{i} = {level}" for i, level in enumerate(levels))
        text = f"{text} Answer on a 0-{len(levels) - 1} scale: {scale}."
        kind_, lines, labels = "score", [str(i) for i in range(6)], list(range(len(levels)))
    else:
        raise BadRequest(f"question {qid!r}: unsupported type {kind!r}")
    prompt = f"[kind] {kind_}\n[state] {state}\n[question] {text}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    return kind_, prompt, labels


def to_answer(kind: str, labels: list, probs: list[float]) -> dict:
    """The wire-format answer from the probabilities over the question's first len(labels) slots."""
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} probabilities for {len(labels)} options")
    if kind == "noul":
        return {"type": "noul", "noul": probs[1]}
    if kind == "choice":
        best = max(range(len(labels)), key=probs.__getitem__)
        return {"type": "choice", "choice": labels[best], "probabilities": dict(zip(labels, probs))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
            "probabilities": {str(i): p for i, p in enumerate(probs)}}


# decide(kind, prompt, n) -> probabilities over the kind's first n slots
Decide = Callable[[str, str, int], list[float]]


def answer(decide: Decide, lock: threading.Lock, name: str, body: Any) -> tuple[int, dict]:
    try:
        state = as_text(body.get("state"))
        questions = body["questions"]
        if not isinstance(questions, dict) or not questions:
            raise BadRequest("questions must be a non-empty object")
        prompts = {qid: to_prompt(state, qid, q) for qid, q in questions.items()}
    except TooManyOptions as e:
        return 422, {"detail": {"code": "TOO_MANY_OPTIONS", "message": str(e)}}
    except BadRequest as e:
        return 400, {"detail": str(e)}
    except (AttributeError, KeyError, TypeError) as e:
        return 400, {"detail": f"bad request: {e!r}"}
    try:
        with lock:  # one forward pass at a time on the device
            answers = {qid: to_answer(kind, labels, decide(kind, prompt, len(labels)))
                       for qid, (kind, prompt, labels) in prompts.items()}
    except Exception as e:  # answer it, so the bench does not see a dropped connection
        return 500, {"detail": f"adapter failed: {e!r}"}
    return 200, {"model": name, "answers": answers}


def load(model_dir: Path, device: str) -> Decide:
    """The card's transformers + peft System 1 path, on `device`."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, model_dir / "adapter").merge_and_unload().eval()
    head = load_file(model_dir / "head.safetensors")
    W, b = head["proj.weight"].to(device, torch.float32), head["proj.bias"].to(device, torch.float32)
    ranges = json.loads((model_dir / "judge_config.json").read_text())["slots"]["ranges"]
    temp = json.loads((model_dir / "calibration.json").read_text())["per_kind"]

    def decide(kind: str, prompt: str, n: int) -> list[float]:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            h = model.model(**ids).last_hidden_state[0, -1].float()
            z = (W @ h + b) / temp[kind]
            s = ranges[kind][0]
            return torch.softmax(z[s: s + n], 0).tolist()

    return decide


class LocalHTTPServer(ThreadingHTTPServer):
    """Sets server_name from the address instead of resolving it (see serve/anyjev_server.py LocalHTTPServer)."""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def make_server(decide: Decide, name: str, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"detail": f"no route {self.path}"})
            self._send(200, {"status": "ok", "model": name})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"detail": f"no route {self.path}"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            except ValueError as e:
                return self._send(400, {"detail": f"body is not JSON: {e}"})
            if not isinstance(body, dict):
                return self._send(400, {"detail": "body must be a JSON object"})
            self._send(*answer(decide, lock, name, body))

        def _send(self, status: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    return LocalHTTPServer((host, port), Handler)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--model", type=Path, required=True, help="local snapshot of autotrust/JEV-9B")
    ap.add_argument("--name", required=True, help="what /health reports, e.g. JEV-9B@<revision>")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    server = make_server(load(args.model, args.device), args.name, args.port)
    print(f"JEV-9B adapter on 127.0.0.1:{args.port} ({args.name}, {args.device})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
