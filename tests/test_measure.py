"""scripts/measure.sh against a stand-in server and a stand-in bench: no model is loaded."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def measure(tmp_path: Path, *args: str, bench_exit: int = 0, start: str | None = None, ready_timeout: int = 20):
    """Run measure.sh with a stand-in server (python -m http.server, pid written to server.pid) and a stand-in
    bench that logs its arguments to bench.log and exits with `bench_exit`."""
    port = free_port()
    (tmp_path / "ready").write_text("ok")
    pidfile = tmp_path / "server.pid"
    bench = tmp_path / "bench.sh"
    bench.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/bench.log\nexit {bench_exit}\n')
    bench.chmod(0o755)
    env = {**os.environ,
           "MEASURE_START": start or (f"echo $$ > {pidfile}; exec python3 -m http.server {port} --bind 127.0.0.1 "
                                      f"--directory {tmp_path}"),
           "MEASURE_READY_URL": f"http://127.0.0.1:{port}/ready",
           "MEASURE_READY_TIMEOUT": str(ready_timeout),
           "MEASURE_BENCH": str(bench),
           "MEASURE_LOGS": str(tmp_path / "logs")}
    r = subprocess.run(["bash", "scripts/measure.sh", *args], cwd=ROOT, capture_output=True, text=True, env=env,
                       timeout=120)
    pid = int(pidfile.read_text()) if pidfile.exists() else None
    return r, pid


def wait_gone(pid: int, seconds: float = 5) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


def wait_listening(port: int) -> None:
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.1)


def test_measures_every_set_then_stops_the_server(tmp_path):
    r, pid = measure(tmp_path, "kev-4b", "a.jsonl", "b.jsonl")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "bench.log").read_text().splitlines() == [
        "--engine kev-4b --questions a.jsonl", "--engine kev-4b --questions b.jsonl"]
    assert pid and wait_gone(pid)


def test_reps_reach_the_bench(tmp_path):
    r, _ = measure(tmp_path, "--reps", "1", "kev-4b", "a.jsonl")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "bench.log").read_text().strip() == "--engine kev-4b --questions a.jsonl --reps 1"


def test_stops_the_server_when_the_bench_fails(tmp_path):
    r, pid = measure(tmp_path, "kev-4b", "a.jsonl", "b.jsonl", bench_exit=1)
    assert r.returncode == 1
    assert pid and wait_gone(pid)
    assert len((tmp_path / "bench.log").read_text().splitlines()) == 2  # the next set is still measured


def test_a_server_that_exits_before_it_is_ready(tmp_path):
    r, _ = measure(tmp_path, "kev-4b", "a.jsonl", start="echo boom >&2; exit 3")
    assert r.returncode == 1
    assert "exited before it was ready" in r.stderr and "boom" in r.stderr
    assert not (tmp_path / "bench.log").exists()


def test_a_server_that_never_gets_ready_is_stopped(tmp_path):
    pidfile = tmp_path / "server.pid"
    r, pid = measure(tmp_path, "kev-4b", "a.jsonl", start=f"echo $$ > {pidfile}; exec sleep 300", ready_timeout=2)
    assert r.returncode == 1 and "not ready after 2s" in r.stderr
    assert pid and wait_gone(pid)


def test_unknown_engine_and_missing_sets(tmp_path):
    r, _ = measure(tmp_path, "nope", "a.jsonl")
    assert r.returncode == 2 and "unknown engine" in r.stderr
    r, _ = measure(tmp_path, "kev-4b")
    assert r.returncode == 2 and "usage" in r.stderr


def test_refuses_a_port_that_already_answers(tmp_path):
    # a stale server on the ready URL would otherwise be measured under this engine's name
    port = free_port()
    (tmp_path / "ready").write_text("ok")
    stale = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory",
                              str(tmp_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_listening(port)
        env = {**os.environ, "MEASURE_START": "exec sleep 30", "MEASURE_READY_URL": f"http://127.0.0.1:{port}/ready",
               "MEASURE_BENCH": "true", "MEASURE_LOGS": str(tmp_path / "logs")}
        r = subprocess.run(["bash", "scripts/measure.sh", "kev-4b", "a.jsonl"], cwd=ROOT, capture_output=True,
                           text=True, env=env, timeout=60)
    finally:
        stale.kill()
    assert r.returncode == 1 and "already answers" in r.stderr


def test_stopping_a_server_stops_its_children(tmp_path):
    child = tmp_path / "child.pid"
    port = free_port()
    start = (f"sleep 300 & echo $! > {child}; echo $$ > {tmp_path}/server.pid; "
             f"exec python3 -m http.server {port} --bind 127.0.0.1 --directory {tmp_path}")
    (tmp_path / "ready").write_text("ok")
    env = {**os.environ, "MEASURE_START": start, "MEASURE_READY_URL": f"http://127.0.0.1:{port}/ready",
           "MEASURE_BENCH": "true", "MEASURE_LOGS": str(tmp_path / "logs")}
    r = subprocess.run(["bash", "scripts/measure.sh", "kev-4b", "a.jsonl"], cwd=ROOT, capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert wait_gone(int(child.read_text()))  # like the python that `uv run` starts under a serve script


def fake_scripts(tmp_path: Path, **servers: int) -> Path:
    """Stand-ins for scripts/serve-<name>.sh: each appends its pid to <name>.pids and serves tmp_path on its port."""
    d = tmp_path / "scripts"
    d.mkdir()
    for name, port in servers.items():
        s = d / f"serve-{name}.sh"
        s.write_text(f"#!/bin/sh\necho $$ >> {tmp_path}/{name}.pids\n"
                     f"exec python3 -m http.server {port} --bind 127.0.0.1 --directory {tmp_path}\n")
        s.chmod(0o755)
    return d


def test_anyjev_adapter_restarts_before_every_set(tmp_path):
    llm, adapter = free_port(), free_port()
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "models").write_text("{}")
    env = {**os.environ, "MEASURE_SCRIPTS": str(fake_scripts(tmp_path, llm=llm, anyjev=adapter)),
           "ANYJEV_LLM_URL": f"http://127.0.0.1:{llm}", "ANYJEV_URL": f"http://127.0.0.1:{adapter}",
           "MEASURE_BENCH": "true", "MEASURE_LOGS": str(tmp_path / "logs")}
    r = subprocess.run(["bash", "scripts/measure.sh", "anyjev-l0", "a.jsonl", "b.jsonl"], cwd=ROOT,
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    adapters = [int(p) for p in (tmp_path / "anyjev.pids").read_text().split()]
    llms = [int(p) for p in (tmp_path / "llm.pids").read_text().split()]
    assert len(adapters) == 2 and len(set(adapters)) == 2 and len(llms) == 1  # one generate server, fresh adapters
    assert all(wait_gone(p) for p in adapters + llms)


@pytest.mark.parametrize("engine", ["clef-flash", "clef"])
def test_clef_engines_start_llama_server_once(tmp_path, engine):
    port = free_port()
    (tmp_path / "health").write_text("ok")
    bench = tmp_path / "bench.sh"
    bench.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/bench.log\n')
    bench.chmod(0o755)
    env = {**os.environ, "MEASURE_SCRIPTS": str(fake_scripts(tmp_path, llama=port)),
           "LLAMA_URL": f"http://127.0.0.1:{port}", "MEASURE_BENCH": str(bench),
           "MEASURE_LOGS": str(tmp_path / "logs")}
    r = subprocess.run(["bash", "scripts/measure.sh", engine, "a.jsonl", "b.jsonl"], cwd=ROOT,
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    pids = [int(p) for p in (tmp_path / "llama.pids").read_text().split()]
    assert len(pids) == 1 and wait_gone(pids[0])
    assert (tmp_path / "bench.log").read_text().splitlines() == [
        f"--engine {engine} --questions a.jsonl", f"--engine {engine} --questions b.jsonl"]


def test_von_is_started_once(tmp_path):
    port = free_port()
    (tmp_path / "health").write_text("ok")
    bench = tmp_path / "bench.sh"
    bench.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/bench.log\n')
    bench.chmod(0o755)
    env = {**os.environ, "MEASURE_SCRIPTS": str(fake_scripts(tmp_path, von=port)),
           "VON_URL": f"http://127.0.0.1:{port}", "MEASURE_BENCH": str(bench), "MEASURE_LOGS": str(tmp_path / "logs")}
    r = subprocess.run(["bash", "scripts/measure.sh", "von", "a.jsonl"], cwd=ROOT,
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    pids = [int(p) for p in (tmp_path / "von.pids").read_text().split()]
    assert len(pids) == 1 and wait_gone(pids[0])
    assert (tmp_path / "bench.log").read_text().splitlines() == ["--engine von --questions a.jsonl"]


def test_ollaya_daemon_is_reused_and_the_model_unloaded(tmp_path):
    port = free_port()
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "tags").write_text("{}")
    daemon = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory",
                               str(tmp_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "ollaya").write_text(f'#!/bin/sh\necho "$OLLAYA_HOST $@" >> {tmp_path}/ollaya.log\n')
    (bin_ / "ollaya").chmod(0o755)
    try:
        wait_listening(port)
        env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "OLLAYA_URL": f"http://127.0.0.1:{port}",
               "MEASURE_BENCH": "true", "MEASURE_LOGS": str(tmp_path / "logs")}
        r = subprocess.run(["bash", "scripts/measure.sh", "winnow", "a.jsonl"], cwd=ROOT, capture_output=True,
                           text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stderr
        assert (tmp_path / "ollaya.log").read_text().strip() == f"127.0.0.1:{port} stop winnow:e4b"
        assert daemon.poll() is None  # not ours to stop
    finally:
        daemon.kill()


def test_a_failed_unload_is_reported(tmp_path):
    port = free_port()
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "tags").write_text("{}")
    daemon = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory",
                               str(tmp_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "ollaya").write_text("#!/bin/sh\necho 'no such model' >&2\nexit 1\n")
    (bin_ / "ollaya").chmod(0o755)
    try:
        wait_listening(port)
        env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "OLLAYA_URL": f"http://127.0.0.1:{port}",
               "MEASURE_BENCH": "true", "MEASURE_LOGS": str(tmp_path / "logs")}
        r = subprocess.run(["bash", "scripts/measure.sh", "winnow", "a.jsonl"], cwd=ROOT, capture_output=True,
                           text=True, env=env, timeout=60)
    finally:
        daemon.kill()
    assert "could not unload winnow:e4b" in r.stderr and "no such model" in r.stderr


def test_failed_sets_are_named(tmp_path):
    r, _ = measure(tmp_path, "kev-4b", "a.jsonl", "b.jsonl", bench_exit=1)
    assert "failed sets: a.jsonl b.jsonl" in r.stderr


PTY_RUN = """
import os, pty, sys, time
log, argv = sys.argv[1], sys.argv[2:]
pid, fd = pty.fork()
if pid == 0:
    os.execvp(argv[0], argv)
deadline = time.time() + 20
while not os.path.exists(log) and time.time() < deadline:
    time.sleep(0.1)
os.write(fd, b"\\x03")
while True:
    try:
        if not os.read(fd, 4096):
            break
    except OSError:
        break
print(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""


def test_ctrl_c_stops_the_whole_run(tmp_path):
    # From a terminal, Ctrl-C goes to the foreground process group; the run must stop there, not skip to the next set.
    # The pty is forked by a separate single-threaded python: forking one from pytest's threads is unsafe.
    port = free_port()
    (tmp_path / "ready").write_text("ok")
    bench = tmp_path / "bench.sh"
    bench.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/bench.log\nsleep 4\n')
    bench.chmod(0o755)
    pidfile = tmp_path / "server.pid"
    env = {**os.environ, "MEASURE_START": (f"echo $$ > {pidfile}; exec python3 -m http.server {port} --bind 127.0.0.1 "
                                           f"--directory {tmp_path}"),
           "MEASURE_READY_URL": f"http://127.0.0.1:{port}/ready", "MEASURE_BENCH": str(bench),
           "MEASURE_LOGS": str(tmp_path / "logs")}
    r = subprocess.run(["python3", "-c", PTY_RUN, str(tmp_path / "bench.log"),
                        "bash", "scripts/measure.sh", "kev-4b", "a.jsonl", "b.jsonl", "c.jsonl"],
                       cwd=ROOT, capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.strip() == "130", r.stderr
    assert (tmp_path / "bench.log").read_text().splitlines() == ["--engine kev-4b --questions a.jsonl"]
    assert wait_gone(int(pidfile.read_text()))
