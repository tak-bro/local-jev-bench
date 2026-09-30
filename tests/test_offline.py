"""Offline checks for the scripts: no model is loaded, fake HTTP servers stand in for the engines."""

from __future__ import annotations

import io
import json
import math
import os
import socket
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run  # noqa: E402
import score  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeServer:
    """Serves `routes[path](body) -> (status, json)` on localhost."""

    def __init__(self, routes):
        routes_ = routes

        class H(BaseHTTPRequestHandler):
            def _reply(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                status, payload = routes_[self.path](body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _reply

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def embed_server(vec):
    return FakeServer({
        "/v1/models": lambda _: (200, {"data": [{"id": "fake"}]}),
        "/v1/embeddings": lambda _: (200, {"data": [{"embedding": vec}]}),
    })


def sh(*args, env=None):
    return subprocess.run(["bash", *args], cwd=ROOT, capture_output=True, text=True, env=env, timeout=60)


# --- smoke-embed.sh -------------------------------------------------------------------------------

def test_smoke_embed_fails_without_server():
    assert sh("scripts/smoke-embed.sh", str(free_port())).returncode == 1


@pytest.mark.parametrize("dim", [None, "4"])
def test_smoke_embed_accepts_normalised_vector(dim):
    srv = embed_server([1 / math.sqrt(4)] * 4)
    try:
        r = sh("scripts/smoke-embed.sh", str(srv.port), *([dim] if dim else []))
    finally:
        srv.close()
    assert r.returncode == 0, r.stderr
    assert "dim=4" in r.stdout


@pytest.mark.parametrize("vec,dim,reason", [
    ([1.0] * 4, "4", "L2 norm"),          # right dimension, not normalised
    ([0.5, 0.5], None, "L2 norm"),        # no dimension given, not normalised
    ([1.0, 0.0], "4", "dimension"),
    ([], None, "empty"),
])
def test_smoke_embed_rejects_bad_vectors(vec, dim, reason):
    srv = embed_server(vec)
    try:
        r = sh("scripts/smoke-embed.sh", str(srv.port), *([dim] if dim else []))
    finally:
        srv.close()
    assert r.returncode == 1
    assert reason in r.stderr


# --- serve scripts --------------------------------------------------------------------------------

def fake_on_path(tmp_path, *names) -> dict:
    """Env with stand-ins for `names` that exit 99: if a port check ever falls through, no real model gets loaded."""
    for name in names:
        p = tmp_path / name
        p.write_text("#!/bin/sh\nexit 99\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}


@pytest.mark.parametrize("script,args", [
    ("scripts/serve-embed.sh", ["m", "{port}", "n"]),
    ("scripts/serve-clm.sh", []),
    ("scripts/serve-anyjev.sh", []),
    ("scripts/serve-kev.sh", []),
])
def test_serve_refuses_busy_port(tmp_path, script, args):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))  # never the real :8700 — clm-serve may be up while tests run
        s.listen()
        port = s.getsockname()[1]
        env = {**fake_on_path(tmp_path, "vllm", "uv"), "CLM_PORT": str(port), "ANYJEV_PORT": str(port),
               "KEV_PORT": str(port)}
        r = sh(script, *[a.format(port=port) for a in args], env=env)
    assert r.returncode == 1, (r.returncode, r.stderr)
    assert "already in use" in r.stderr


def test_serve_accepts_a_port_in_time_wait(tmp_path):
    # A server stopped right after answering leaves TIME_WAIT sockets on its port; restarting it must not fail.
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()
        conn.close()  # the server side closes first, so its end of the port sits in TIME_WAIT
        client.close()
    r = sh("scripts/serve-anyjev.sh", env={**fake_on_path(tmp_path, "uv"), "ANYJEV_PORT": str(port)})
    assert "already in use" not in r.stderr
    assert r.returncode == 99  # reached the (stubbed) server


def test_serve_refuses_a_bound_port_that_is_not_listening(tmp_path):
    # vLLM binds before it listens; a hung one holds the port that way.
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        r = sh("scripts/serve-anyjev.sh", env={**fake_on_path(tmp_path, "uv"), "ANYJEV_PORT": str(s.getsockname()[1])})
    assert r.returncode == 1 and "already in use" in r.stderr


def test_serve_refuses_a_port_listening_on_all_interfaces(tmp_path):
    # A stale server on 0.0.0.0 still answers 127.0.0.1, even though SO_REUSEADDR lets 127.0.0.1 bind beside it.
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        s.listen()
        r = sh("scripts/serve-anyjev.sh", env={**fake_on_path(tmp_path, "uv"), "ANYJEV_PORT": str(s.getsockname()[1])})
    assert r.returncode == 1 and "already in use" in r.stderr


def test_serve_kev_needs_a_checkout(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = sh("scripts/serve-kev.sh", env={**fake_on_path(tmp_path, "uv"), "KEV_PORT": str(port),
                                        "KEV_DIR": str(tmp_path / "nope")})
    assert r.returncode == 1 and "no Kev checkout" in r.stderr


# --- run.py ---------------------------------------------------------------------------------------

def systemone_server(status_for=lambda body: 200):
    def handle(body):
        st = status_for(body)
        if st != 200:
            return st, {"detail": "boom"}
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.9}
            elif q["type"] == "choice":
                first = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": first, "confidence": 0.9,
                                "probabilities": {first: 0.9}}
            else:
                answers[qid] = {"type": "score", "score": 1.0, "confidence": 0.9, "probabilities": {}}
        return 200, {"answers": answers}
    return FakeServer({"/v1/systemone": handle})


@pytest.fixture
def engine(monkeypatch):
    def point(port):
        monkeypatch.setitem(run.ENGINES, "fake", {"url": f"http://127.0.0.1:{port}", "model": None})
        return "fake"
    return point


def test_ask_raises_when_engine_is_down(engine):
    with pytest.raises(run.EngineError):
        run.ask(engine(free_port()), "s", run.README_QUESTIONS, timeout=5)


def test_ask_rejects_missing_answer_field(engine):
    srv = FakeServer({"/v1/systemone": lambda b: (200, {"answers": {"urgency": {"type": "noul"}}})})
    try:
        with pytest.raises(run.EngineError):
            run.ask(engine(srv.port), "s", run.README_QUESTIONS)
    finally:
        srv.close()


@pytest.mark.parametrize("answers", [
    {"urgency": {"type": "noul", "noul": "0.4"}},                       # number sent as a string
    {"urgency": {"type": "noul", "noul": True}},
    {"department": {"type": "choice", "choice": "billing"}},           # no probabilities
])
def test_ask_rejects_wrongly_typed_answers(engine, answers):
    qs = {k: run.README_QUESTIONS[k] for k in answers}
    srv = FakeServer({"/v1/systemone": lambda b: (200, {"answers": answers})})
    try:
        with pytest.raises(run.EngineError):
            run.ask(engine(srv.port), "s", qs)
    finally:
        srv.close()


def test_smoke_cli_exits_1_when_engine_down():
    env = {**os.environ, "CLM_URL": f"http://127.0.0.1:{free_port()}"}
    r = subprocess.run([sys.executable, "bench/run.py", "--smoke", "--engine", "clm"],
                       cwd=ROOT, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1


ITEMS = [{"id": f"i{i}", "state": f"state {i}",
          "questions": {"u": {"type": "noul", "instructions": "?"}}, "expected": {"u": True}} for i in range(4)]


def bench_records(engine, items, **kw) -> tuple[int, list[dict]]:
    raw = io.StringIO()
    errors = run.bench(engine, items, raw, **kw)
    return errors, [json.loads(line) for line in raw.getvalue().splitlines()]


def test_bench_logs_every_call(engine):
    srv = systemone_server()
    try:
        errors, recs = bench_records(engine(srv.port), ITEMS)
    finally:
        srv.close()
    assert errors == 0
    assert [r["call"] for r in recs[:1 + run.WARMUP]] == ["cold"] + ["warmup"] * run.WARMUP
    timed = [r for r in recs if r["call"] == "timed"]
    assert [(r["item"], r["rep"]) for r in timed] == [(it["id"], rep) for it in ITEMS for rep in range(run.REPS)]
    assert all(r["answers"]["u"]["noul"] == 0.9 and r["error"] is None and r["ms"] > 0 for r in timed)
    assert all(r["pressure"] for r in recs)
    assert len(recs) == 1 + run.WARMUP + len(ITEMS) * run.REPS  # no choice questions, so no reversed calls


def test_bench_reps(engine):
    srv = systemone_server()
    try:
        _, recs = bench_records(engine(srv.port), ITEMS, reps=1)
    finally:
        srv.close()
    assert {r["rep"] for r in recs if r["call"] == "timed"} == {0}


def test_reps_below_one_is_refused(monkeypatch, tmp_path):
    # --runs points away from bench/: if the check ever lapses, main() runs and must not overwrite real logs
    monkeypatch.setattr(run, "bench", lambda engine, items, raw, reps: 0)
    monkeypatch.setattr(run, "load_questions", lambda path, count: ITEMS)
    monkeypatch.setattr(run, "qwen_token_counter", lambda: len)
    monkeypatch.setattr(sys, "argv", ["run.py", "--engine", "kev", "--reps", "0", "--runs", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        run.main()
    assert e.value.code == 2  # argparse usage error


def test_bench_errors_are_logged_without_answers(engine):
    srv = systemone_server(lambda body: 500 if body["state"] == "state 2" else 200)
    try:
        errors, recs = bench_records(engine(srv.port), ITEMS)
    finally:
        srv.close()
    failed = [r for r in recs if r["error"]]
    assert errors == len(failed) == run.REPS
    assert all(r["item"] == "i2" and r["answers"] is None and r["ms"] is None for r in failed)


def test_bench_stops_when_the_cold_call_fails(engine):
    errors, recs = bench_records(engine(free_port()), ITEMS)
    assert errors == 1 and [r["call"] for r in recs] == ["cold"]  # an engine that fails its warm-up gets no numbers


def test_load_questions_rejects_long_state(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(it) for it in ITEMS))
    with pytest.raises(SystemExit, match="i3"):
        run.load_questions(p, lambda text: 10_000 if text.startswith("state 3") else 5)


def test_structured_state_goes_out_as_the_text_kev_renders(tmp_path):
    # kev/api.py render(): field names kept as labels, in their order; nested objects indented, lists as "- " lines.
    # Kev feeds a dict state to its model that way, so every engine gets the same text Kev reads.
    state = {"b": "é", "a": 1, "case": {"x": True, "items": ["one", {"k": "v"}]}, "none": None}
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({**ITEMS[0], "state": state}) + "\n")
    (item,) = run.load_questions(p, len)
    assert item["state"] == "b: é\na: 1\ncase:\n  x: True\n  items:\n    - one\n    - k: v\nnone: "


def test_token_guard_counts_state_plus_instructions():
    from clm.schema import state_text
    it = {"state": " s ", "questions": {"a": {"type": "noul", "instructions": " long question "}}}
    assert run.embedded_texts(it) == [state_text(" s ", " long question ")]


def test_shipped_questions_are_well_formed():
    items = run.load_questions(ROOT / "bench" / "questions.jsonl", lambda text: len(text.split()))
    assert len(items) == 30
    for it in items:
        assert set(it["expected"]) <= set(it["questions"])
        for qid, want in it["expected"].items():
            q = it["questions"][qid]
            if q["type"] == "choice":
                assert want in q["criteria"]
            if q["type"] == "score":
                assert 0 <= want < len(q["criteria"])


CHOICE_ITEMS = [{"id": f"c{i}", "state": f"state {i}",
                 "questions": {"u": {"type": "noul", "instructions": "?"},
                               "d": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}},
                 "expected": {"u": True, "d": "a"}} for i in range(4)]


def test_reversed_call_follows_the_timed_reps(engine):
    srv = systemone_server()  # always picks the first listed option
    try:
        _, recs = bench_records(engine(srv.port), CHOICE_ITEMS)
    finally:
        srv.close()
    per_item = [r["call"] for r in recs if r["item"] == "c1"]
    assert per_item == ["timed"] * run.REPS + ["reversed"]
    rev = next(r for r in recs if r["item"] == "c1" and r["call"] == "reversed")
    assert rev["answers"]["d"]["choice"] == "b"  # the options went out reversed


def test_no_reversed_call_after_a_failed_rep_0(engine):
    fail_rep_0 = iter([500])  # the warm-up only calls state 0, so state 1's first call is its rep 0
    srv = systemone_server(lambda body: next(fail_rep_0, 200) if body["state"] == "state 1" else 200)
    try:
        _, recs = bench_records(engine(srv.port), CHOICE_ITEMS)
    finally:
        srv.close()
    assert [r["call"] for r in recs if r["item"] == "c1"] == ["timed"] * run.REPS  # no reference answer to flip


def run_main(monkeypatch, tmp_path, *argv) -> int:
    monkeypatch.setattr(run, "qwen_token_counter", lambda: len)
    monkeypatch.setattr(sys, "argv", ["run.py", "--runs", str(tmp_path / "runs"), *argv])
    return run.main()


def test_main_writes_raw_log_and_report(engine, monkeypatch, tmp_path):
    qs = tmp_path / "set.jsonl"
    qs.write_text("".join(json.dumps(it) + "\n" for it in CHOICE_ITEMS))
    srv = systemone_server()
    try:
        name = engine(srv.port)
        assert run_main(monkeypatch, tmp_path, "--engine", name, "--questions", str(qs), "--reps", "1") == 0
    finally:
        srv.close()
    d = tmp_path / "runs" / "set"
    lines = [json.loads(line) for line in (d / f"{name}.jsonl").read_text().splitlines()]
    header = lines[0]["header"]
    assert "finished" in lines[-1]["footer"]
    assert header["set"] == str(qs) and header["reps"] == 1 and header["engine"] == name
    assert header["set_sha256"] == score.sha256(qs)
    assert f"| {name} |" in (d / "report.md").read_text()


def test_main_exits_1_on_engine_errors(engine, monkeypatch, tmp_path):
    qs = tmp_path / "set.jsonl"
    qs.write_text("".join(json.dumps(it) + "\n" for it in ITEMS))
    assert run_main(monkeypatch, tmp_path, "--engine", engine(free_port()), "--questions", str(qs)) == 1


def test_bare_bench_runs_only_the_original_engines(monkeypatch, tmp_path):
    benched = []
    monkeypatch.setattr(run, "load_questions", lambda path, count: ITEMS)
    monkeypatch.setattr(run, "bench", lambda engine, items, raw, reps: benched.append(engine) or 0)
    monkeypatch.setattr(run.score, "render", lambda d: "")
    run_main(monkeypatch, tmp_path)
    assert benched == ["clm", "ollaya"]  # new engines are measured one at a time, by name


@pytest.mark.parametrize("name, model", [("anyjev-raw", "anyjev-raw"), ("anyjev-l0", "anyjev-l0"), ("kev", "kev-latest")])
def test_new_engines_send_their_model(monkeypatch, name, model):
    seen = []
    def handle(body):
        seen.append(body.get("model"))
        return 200, {"answers": {"u": {"type": "noul", "noul": 0.9}}}
    srv = FakeServer({"/v1/systemone": handle})
    monkeypatch.setitem(run.ENGINES, name, {**run.ENGINES[name], "url": f"http://127.0.0.1:{srv.port}"})
    try:
        run.ask(name, "s", {"u": {"type": "noul", "instructions": "?"}})
    finally:
        srv.close()
    assert seen == [model]
