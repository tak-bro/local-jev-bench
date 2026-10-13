"""Offline checks for bench/llm_compare.py: fake SSE/chat servers, no real model."""

from __future__ import annotations

import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import llm_compare  # noqa: E402


def chat_server(chunks=("Hello", " world"), usage_tokens=5):
    """SSE chunks plus an optional usage event; non-streaming calls get a usage reply."""

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n)) if n else {}
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for tok in chunks:
                    self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": tok}}]})
                                      + "\n").encode())
                    self.wfile.flush()
                if usage_tokens is not None:
                    self.wfile.write(("data: " + json.dumps({"usage": {"completion_tokens": usage_tokens}})
                                      + "\n").encode())
                self.wfile.write(b"data: [DONE]\n")
            else:
                data = json.dumps({"usage": {"completion_tokens": 7},
                                   "choices": [{"message": {"content": "x"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def url_of(httpd) -> str:
    return f"http://127.0.0.1:{httpd.server_address[1]}"


def test_body_marks_thinking_off_and_stream_options():
    b = llm_compare.body("m", "Say hi.", True)
    assert b["messages"] == [{"role": "user", "content": "Say hi. /no_think"}]
    assert b["stream_options"] == {"include_usage": True}
    assert b["temperature"] == 0 and b["max_tokens"] == llm_compare.MAX_TOKENS
    assert "stream_options" not in llm_compare.body("m", "Say hi.", False)


def test_stream_one_prefers_the_server_token_count():
    srv = chat_server()
    try:
        s = llm_compare.stream_one(url_of(srv), "m", "Say hi.")
    finally:
        srv.shutdown()
    assert s["tokens"] == 5 and s["counted_by"] == "usage"
    assert s["ttft_ms"] >= 0 and s["decode_tps"] > 0


def test_stream_one_falls_back_to_chunk_count():
    srv = chat_server(usage_tokens=None)
    try:
        s = llm_compare.stream_one(url_of(srv), "m", "Say hi.")
    finally:
        srv.shutdown()
    assert s["tokens"] == 2 and s["counted_by"] == "chunks"


def test_stream_one_empty_stream():
    srv = chat_server(chunks=(), usage_tokens=None)
    try:
        s = llm_compare.stream_one(url_of(srv), "m", "Say hi.")
    finally:
        srv.shutdown()
    assert s["tokens"] == 0 and s["counted_by"] == "chunks"
    assert math.isnan(s["decode_tps"])


def test_one_blocking_reads_usage_tokens():
    srv = chat_server()
    try:
        assert llm_compare.one_blocking(url_of(srv), "m", "Say hi.") == 7
    finally:
        srv.shutdown()


def test_main_appends_a_row_to_a_new_report(monkeypatch, tmp_path):
    srv = chat_server()
    out = tmp_path / "llm_report.md"
    monkeypatch.setattr(sys, "argv", ["llm_compare.py", "--name", "fake", "--url", url_of(srv),
                                      "--model", "m", "--concurrency", "2", "--out", str(out)])
    extra_before = llm_compare.EXTRA.copy()
    try:
        llm_compare.main()
    finally:
        llm_compare.EXTRA.clear()
        llm_compare.EXTRA.update(extra_before)
        srv.shutdown()
    lines = out.read_text().splitlines()
    assert "token count source" in lines[4]
    row = lines[-1].split("|")
    assert row[1].strip() == "fake" and float(row[2].strip()) >= 0
