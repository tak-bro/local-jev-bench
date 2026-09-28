"""Offline checks for the scripts: no model is loaded, fake HTTP servers stand in for the engines."""

from __future__ import annotations

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


def test_bench_separates_cold_from_warm(engine):
    srv = systemone_server()
    try:
        res = run.bench(engine(srv.port), ITEMS)
    finally:
        srv.close()
    assert res.cold_ms is not None
    assert len(res.warm_ms) == len(ITEMS) * run.REPS
    assert len(res.first_ms) == len(ITEMS)
    assert (res.right, res.graded) == (len(ITEMS) * run.REPS,) * 2


def test_bench_reps(engine):
    srv = systemone_server()
    try:
        res = run.bench(engine(srv.port), ITEMS, reps=1)
    finally:
        srv.close()
    assert len(res.warm_ms) == len(ITEMS) and res.graded == len(ITEMS)
    assert "4 questions x 1 reps" in run.report([res], len(ITEMS), reps=1)


def test_reps_below_one_is_refused(monkeypatch, tmp_path):
    # --out points away from bench/: if the check ever lapses, main() runs and must not overwrite a real report
    monkeypatch.setattr(run, "bench", lambda engine, items, reps: run.Result(engine))
    monkeypatch.setattr(run, "load_questions", lambda path, count: ITEMS)
    monkeypatch.setattr(run, "qwen_token_counter", lambda: len)
    monkeypatch.setattr(sys, "argv", ["run.py", "--engine", "kev", "--reps", "0", "--out", str(tmp_path / "r.md")])
    with pytest.raises(SystemExit) as e:
        run.main()
    assert e.value.code == 2  # argparse usage error


def test_reps_flag_reaches_bench_and_report(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(run, "qwen_token_counter", lambda: len)
    monkeypatch.setattr(run, "load_questions", lambda path, count: ITEMS)
    monkeypatch.setattr(run, "bench", lambda engine, items, reps: seen.append(reps) or run.Result(engine))
    out = tmp_path / "r.md"
    monkeypatch.setattr(sys, "argv", ["run.py", "--engine", "kev", "--reps", "1", "--out", str(out)])
    run.main()
    assert seen == [1] and "x 1 reps" in out.read_text()


def test_bench_errors_add_no_latency_or_answers(engine):
    srv = systemone_server(lambda body: 500 if body["state"] == "state 2" else 200)
    try:
        res = run.bench(engine(srv.port), ITEMS)
    finally:
        srv.close()
    assert len(res.errors) == run.REPS
    assert len(res.warm_ms) == (len(ITEMS) - 1) * run.REPS
    assert res.graded == (len(ITEMS) - 1) * run.REPS
    assert all(ms > 0 for ms in res.warm_ms)


def test_report_marks_failed_engine():
    text = run.report([run.Result("fake", errors=["down"], pressure=["normal"])], 4)
    assert "| fake | failed | failed | failed | failed | 0 | failed | failed | 1 | normal |" in text


def test_report_shows_worst_pressure():
    text = run.report([run.Result("fake", pressure=["normal", "critical", "warn"])], 4)
    assert text.rstrip().endswith("| critical |")


def test_correct_rounds_score_half_up():
    q = {"type": "score"}
    assert run.correct(q, {"score": 0.5}, 1) and run.correct(q, {"score": 2.5}, 3)


def test_load_questions_rejects_long_state(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(it) for it in ITEMS))
    with pytest.raises(SystemExit, match="i3"):
        run.load_questions(p, lambda text: 10_000 if text.startswith("state 3") else 5)


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


def test_order_flip_counts_only_choice_questions(engine):
    srv = systemone_server()  # always picks the first listed option, so reversing flips every choice
    try:
        res = run.bench(engine(srv.port), CHOICE_ITEMS)
    finally:
        srv.close()
    assert (res.flips, res.flip_items) == (4, 4)  # the noul question is not in the denominator
    assert not res.errors


def test_order_flip_zero_for_order_independent_engine(engine):
    def handle(body):
        return 200, {"answers": {
            "u": {"type": "noul", "noul": 0.9},
            "d": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}}}
    srv = FakeServer({"/v1/systemone": handle})
    try:
        res = run.bench(engine(srv.port), CHOICE_ITEMS)
    finally:
        srv.close()
    assert (res.flips, res.flip_items) == (0, 4)


def test_failed_reversed_request_counts_neither_way(engine):
    reversed_call = lambda body: next(iter(body["questions"]["d"]["criteria"])) == "b"
    srv = systemone_server(lambda body: 500 if reversed_call(body) else 200)
    try:
        res = run.bench(engine(srv.port), CHOICE_ITEMS)
    finally:
        srv.close()
    assert (res.flips, res.flip_items) == (0, 0)
    assert len(res.errors) == len(CHOICE_ITEMS)
    assert res.graded == len(CHOICE_ITEMS) * run.REPS * 2  # the forward answers still count
    assert "| failed | 4 |" in run.report([res], 4)  # not n/a: there were choices, the reversed calls failed


def test_report_accuracy_has_wilson_interval():
    text = run.report([run.Result("fake", right=237, graded=315, warm_ms=[1.0], first_ms=[1.0],
                                  cold_ms=1.0, pressure=["normal"])], 30)
    assert "237/315 (75%, 70-80)" in text  # Wilson 95%: 70.2-79.7, worked by hand


def test_report_order_flip_column():
    r = run.Result("fake", warm_ms=[1.0], first_ms=[1.0], cold_ms=1.0, pressure=["normal"], flips=1, flip_items=4)
    assert "| 1/4 (25%) |" in run.report([r], 4)
    r.flips = r.flip_items = 0
    assert "| n/a |" in run.report([r], 4)


def ok_result(name):
    return run.Result(name, warm_ms=[1.0], first_ms=[1.0], cold_ms=1.0, pressure=["normal"], right=1, graded=2)


def test_append_replaces_same_engine_row(tmp_path):
    out = tmp_path / "r.md"
    run.write_report(out, [ok_result("a"), ok_result("b")], 4, "q.jsonl", append=False)
    first = out.read_text()
    run.write_report(out, [run.Result("b", errors=["down"], pressure=["normal"])], 4, "q.jsonl", append=True)
    text = out.read_text()
    rows = [l for l in text.splitlines() if l.startswith("| a ") or l.startswith("| b ")]
    assert len(rows) == 2
    assert [l for l in first.splitlines() if l.startswith("| a ")][0] in rows
    assert "| b | failed |" in text and "- b error: down" in text


@pytest.mark.parametrize("n_items, questions", [(30, "bench/q.jsonl"), (4, "other/q.jsonl")])
def test_append_rejects_other_question_set(tmp_path, n_items, questions):
    out = tmp_path / "r.md"
    run.write_report(out, [ok_result("a")], 4, "bench/q.jsonl", append=False)
    with pytest.raises(SystemExit, match="header"):
        run.write_report(out, [ok_result("b")], n_items, questions, append=True)


def test_append_refuses_file_without_table(tmp_path):
    out = tmp_path / "r.md"
    out.write_text("hand-edited notes\n")
    with pytest.raises(SystemExit, match="no results table"):
        run.write_report(out, [ok_result("b")], 4, "bench/q.jsonl", append=True)


def test_bare_bench_runs_only_the_original_engines(monkeypatch, tmp_path):
    benched = []
    monkeypatch.setattr(run, "qwen_token_counter", lambda: len)
    monkeypatch.setattr(run, "load_questions", lambda path, count: ITEMS)
    monkeypatch.setattr(run, "bench", lambda engine, items, reps: benched.append(engine) or run.Result(engine))
    monkeypatch.setattr(sys, "argv", ["run.py", "--out", str(tmp_path / "r.md")])
    run.main()
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
