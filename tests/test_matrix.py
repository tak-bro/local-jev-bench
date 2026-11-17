"""bench/matrix.py without measuring anything: which engine x set a pass still needs, and the pre-start checks."""

from __future__ import annotations

import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import matrix  # noqa: E402
import score  # noqa: E402

SINCE = datetime.fromisoformat("2026-11-20T00:00:00+00:00")


def write_log(path: Path, set_path: Path, started: str, footer: bool = True, sha: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"header": {"started": started, "set_sha256": sha or score.sha256(set_path)}},
             {"item": "i0", "call": "timed"}]
    if footer:
        lines.append({"footer": {"finished": started}})
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


@pytest.fixture
def a_set(tmp_path):
    s = tmp_path / "s.jsonl"
    s.write_text('{"id": "i0"}\n')
    return s


@pytest.mark.parametrize("started, footer, sha, want", [
    ("2026-11-21T00:00:00+00:00", True, None, True),
    ("2026-11-20T00:00:00+00:00", True, None, True),      # at the pass start counts
    ("2026-11-19T23:59:59+00:00", True, None, False),     # an earlier pass
    ("2026-11-21T00:00:00+00:00", False, None, False),    # cut short
    ("2026-11-21T00:00:00+00:00", True, "0" * 64, False),  # measured on another version of the set
])
def test_done(tmp_path, a_set, started, footer, sha, want):
    log = tmp_path / "runs" / "s" / "e.jsonl"
    write_log(log, a_set, started, footer, sha)
    assert matrix.done(log, a_set, SINCE) is want


def test_done_without_a_log_or_with_an_unreadable_one(tmp_path, a_set):
    log = tmp_path / "e.jsonl"
    assert not matrix.done(log, a_set, SINCE)
    log.write_text("")
    assert not matrix.done(log, a_set, SINCE)
    log.write_text("not json\n")
    assert not matrix.done(log, a_set, SINCE)


def test_plan_lists_only_what_is_left(tmp_path, a_set):
    other = tmp_path / "t.jsonl"
    other.write_text('{"id": "j0"}\n')
    runs = tmp_path / "runs"
    write_log(runs / "s" / "a.jsonl", a_set, "2026-11-21T00:00:00+00:00")
    write_log(runs / "t" / "a.jsonl", other, "2026-11-21T00:00:00+00:00")
    write_log(runs / "s" / "b.jsonl", a_set, "2026-11-21T00:00:00+00:00")
    assert matrix.plan(["a", "b", "c"], [a_set, other], runs, SINCE) == [("b", [other]), ("c", [a_set, other])]


def test_every_engine_is_in_the_matrix_and_the_slowest_go_last():
    assert sorted(matrix.ENGINES) == sorted(matrix.run.ENGINES)
    assert matrix.ENGINES[-len(matrix.LAST):] == list(matrix.LAST)


def test_pass_start_is_kept_until_a_new_pass(tmp_path):
    path = tmp_path / "logs" / "matrix.since"
    first = matrix.pass_start(path, new_pass=False)
    path.write_text("2026-01-01T00:00:00+00:00\n")
    assert matrix.pass_start(path, new_pass=False) == datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    assert matrix.pass_start(path, new_pass=True) >= first


@pytest.fixture
def clean_machine(monkeypatch):
    monkeypatch.setattr(matrix.run, "power", lambda: "AC Power; No thermal warning level has been recorded")
    monkeypatch.setattr(matrix.run, "memory_pressure", lambda: "normal")
    monkeypatch.setattr(matrix, "answers", lambda url: False)
    monkeypatch.setattr(matrix, "loaded_models", lambda url: [])


def test_preflight_passes_on_a_clean_machine(clean_machine):
    assert matrix.preflight() == []


def test_preflight_names_every_problem(clean_machine, monkeypatch):
    kev = matrix.run.ENGINES["kev-4b"]["url"]
    ollaya = matrix.run.OLLAYA_URLS["0.7"]
    monkeypatch.setattr(matrix.run, "power", lambda: "Battery Power")
    monkeypatch.setattr(matrix.run, "memory_pressure", lambda: "warn")
    monkeypatch.setattr(matrix, "answers", lambda url: url in (kev, ollaya))  # a daemon answering is fine
    monkeypatch.setattr(matrix, "loaded_models", lambda url: ["laya:en"] if url == ollaya else [])
    problems = matrix.preflight()
    assert problems == ["not on AC power: Battery Power", "memory pressure is warn",
                        f"{kev} already answers; stop that server first",
                        f"Ollaya at {ollaya} holds laya:en; `ollaya stop` them first"]
    assert "not on AC power" not in " ".join(matrix.preflight(allow_battery=True))


def test_answers_sees_a_listening_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert matrix.answers(f"http://127.0.0.1:{s.getsockname()[1]}")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert not matrix.answers(f"http://127.0.0.1:{port}")


def test_main_refuses_to_start_on_a_busy_machine(clean_machine, monkeypatch, tmp_path):
    monkeypatch.setattr(matrix.run, "memory_pressure", lambda: "critical")
    monkeypatch.setattr(sys, "argv", ["matrix.py", "--logs", str(tmp_path / "logs"), "--runs", str(tmp_path / "runs")])
    with pytest.raises(SystemExit, match="memory pressure is critical"):
        matrix.main()
    assert not (tmp_path / "logs" / "matrix.since").exists()  # no pass started


def test_main_refuses_unknown_engines(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["matrix.py", "--dry-run", "nope"])
    with pytest.raises(SystemExit) as e:
        matrix.main()
    assert e.value.code == 2
