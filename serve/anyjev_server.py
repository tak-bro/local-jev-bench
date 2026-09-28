"""Serve AnyJev over TypeSafe's `POST /v1/systemone` wire format, so bench/run.py can measure it like the others.

    uv run python serve/anyjev_server.py --port 8710 --llm-url http://127.0.0.1:8092 --llm-model qwen3-8b

AnyJev reads each answer from the next-token logprobs of a generate server (vllm-metal here), restricted to the
label tokens. The request's `model` picks the correction level: `anyjev-raw` (none) or `anyjev-l0` (cyclic shifts
plus a batch prior). Anything else is refused rather than served at a default level.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from anyjev import Decider, Question
from anyjev.backends.vllm import VLLMBackend

LEVELS = {"anyjev-raw": "raw", "anyjev-l0": "L0"}

class BadRequest(ValueError):
    pass


class LabelMissing(RuntimeError):
    pass


class StrictVLLMBackend(VLLMBackend):
    """VLLMBackend that asks for each label's logprob by id and fails when one is absent.

    The stock backend sends `allowed_token_ids` with `logprobs=K` and expects the top K to be exactly the labels,
    since vLLM can report logprobs after its logit processors. vllm-metal 0.30.0 builds its sampler without the
    configured `--logprobs-mode` (vllm_metal/v1/model_runner.py `Sampler()`), so the top K come from the full
    vocabulary and a label can drop out, which the stock backend fills with -30.0. `logprob_token_ids` returns
    the labels' raw logprobs instead; every AnyJev level softmaxes each row, so raw and processed decide alike.
    Overrides the public `next_token_logprobs`, not the stock private `_one`. Drop this class once vllm-metal
    passes `--logprobs-mode` to its sampler.
    """

    def next_token_logprobs(self, prompts, token_ids):
        with cf.ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._label_logprobs, prompts, token_ids))

    def _label_logprobs(self, prompt, ids):
        body = {"model": self.name, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
                "logprobs": len(ids), "logprob_token_ids": list(ids)}  # vLLM refuses the ids without logprobs
        req = urllib.request.Request(self.base_url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            top = json.load(r)["choices"][0]["logprobs"]["top_logprobs"][0]
        lp = []
        for tid in ids:
            key = self.tokenizer.decode([tid])  # how vLLM renders the token; the raw token string is the fallback
            if key not in top:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            if key not in top:
                raise LabelMissing(f"label token {tid} missing from the server's logprobs {sorted(top)}")
            lp.append(float(top[key]))
        return np.array(lp, dtype=np.float64)


def to_question(qid: str, q: dict) -> tuple[Question, list[str]]:
    """The AnyJev question for one wire-format question, and the labels its options map back to by index."""
    kind, text = q.get("type"), q.get("instructions", "")
    if kind == "noul":
        return Question.noul(text, name=qid), ["Yes", "No"]
    if kind == "choice":
        labels = list(q["criteria"])
        # The model sees options as "A. <text>"; put each description beside its label so it informs the choice.
        shown = [f"{label}: {desc}" if desc else label for label, desc in q["criteria"].items()]
        return Question.choice(text, shown, name=qid), labels
    if kind == "score":
        levels = list(q["criteria"])
        return Question.score(text, levels=levels, name=qid), levels
    raise BadRequest(f"question {qid!r}: unsupported type {kind!r}")


def to_answer(kind: str, labels: list[str], dec) -> dict:
    if kind == "noul":
        return {"type": "noul", "noul": dec.p_true, "confidence": dec.confidence}
    probs = {label: float(p) for label, p in zip(labels, dec.probs)}
    if kind == "choice":
        return {"type": "choice", "choice": labels[int(np.argmax(dec.probs))], "probabilities": probs,
                "confidence": dec.confidence}
    return {"type": "score", "score": dec.value, "probabilities": probs, "confidence": dec.confidence}


def answer(decider: Decider, lock: threading.Lock, body: Any) -> tuple[int, dict]:
    try:
        model = body.get("model")
        if model not in LEVELS:
            raise BadRequest(f"unknown model {model!r}; expected one of {sorted(LEVELS)}")
        state = body.get("state")
        if not isinstance(state, str):
            raise BadRequest(f"state must be a string, got {state!r}")
        converted = {qid: to_question(qid, q) for qid, q in body["questions"].items()}
    except BadRequest as e:
        return 400, {"detail": str(e)}
    except (AttributeError, KeyError, TypeError, ValueError) as e:  # malformed body, or AnyJev refused the spec
        return 400, {"detail": f"bad request: {e!r}"}
    try:
        with lock:  # the batch prior is shared state; the bench sends one request at a time anyway
            decisions = decider.decide(state, [q for q, _ in converted.values()], level=LEVELS[model])
    except (LabelMissing, urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 502, {"detail": f"LLM backend failed: {e!r}"}
    except Exception as e:  # e.g. a 200 without logprobs; answer it so the bench does not see a dropped connection
        return 500, {"detail": f"adapter failed: {e!r}"}
    answers = {qid: to_answer(body["questions"][qid]["type"], labels, decisions[qid])
               for qid, (_, labels) in converted.items()}
    return 200, {"model": model, "answers": answers}


def make_server(decider: Decider, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"detail": f"no route {self.path}"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            except ValueError as e:
                return self._send(400, {"detail": f"body is not JSON: {e}"})
            self._send(*answer(decider, lock, body))

        def _send(self, status: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve_in_thread(server: ThreadingHTTPServer) -> None:
    threading.Thread(target=server.serve_forever, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8092")
    ap.add_argument("--llm-model", default="qwen3-8b", help="the generate server's served model name")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="HF id the label tokens are resolved with")
    args = ap.parse_args()
    backend = StrictVLLMBackend(args.llm_url, args.llm_model, tokenizer_name=args.tokenizer)
    server = make_server(Decider(backend), args.port)
    print(f"anyjev adapter on 127.0.0.1:{args.port} -> {args.llm_url} ({args.llm_model})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
