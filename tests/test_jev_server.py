"""Offline checks for the JEV-9B adapter: a fake `decide` stands in for the model, so torch is not needed."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))
sys.path.insert(0, str(ROOT / "tests"))
import jev_server as srv  # noqa: E402
from test_offline import run  # noqa: E402

QUESTIONS = {
    "urgency": {"type": "noul", "instructions": "Is this urgent?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}


def test_prompts_follow_the_model_card_template():
    kind, prompt, labels = srv.to_prompt("s", "department", QUESTIONS["department"])
    assert kind == "choice" and labels == ["billing", "technical"]
    assert prompt == ("[kind] choice\n[state] s\n[question] Which team should handle this?\n[options]\n"
                      "A) billing: Charges, invoices, refunds\nB) technical: Bugs and outages\n[decision]:")
    kind, prompt, labels = srv.to_prompt("s", "urgency", QUESTIONS["urgency"])
    assert kind == "noul" and labels == [False, True] and prompt.endswith("[options]\nfalse\ntrue\n[decision]:")


def test_a_score_names_its_levels_and_lists_the_fixed_scale():
    # JEV answers on a fixed 0-5 scale: the prompt lists all six, the question says which levels are meant.
    kind, prompt, labels = srv.to_prompt("s", "frustration", QUESTIONS["frustration"])
    assert kind == "score" and labels == [0, 1, 2]
    assert "[question] How frustrated is the customer? Answer on a 0-2 scale: 0 = Calm, 1 = Frustrated, " \
           "2 = Very angry.\n" in prompt
    assert prompt.endswith("[options]\n0\n1\n2\n3\n4\n5\n[decision]:")


def test_answers_in_the_wire_format():
    seen = []

    def decide(kind, prompt, n):
        seen.append((kind, n))
        return {"noul": [0.2, 0.8], "choice": [0.7, 0.3], "score": [0.1, 0.3, 0.6]}[kind]

    status, body = srv.answer(decide, threading.Lock(), "JEV-9B@x", {"state": "s", "questions": QUESTIONS})
    assert status == 200 and body["model"] == "JEV-9B@x"
    a = body["answers"]
    assert a["urgency"] == {"type": "noul", "noul": 0.8}
    assert a["department"] == {"type": "choice", "choice": "billing",
                               "probabilities": {"billing": 0.7, "technical": 0.3}}
    assert a["frustration"]["score"] == pytest.approx(1.5)
    assert a["frustration"]["probabilities"] == {"0": 0.1, "1": 0.3, "2": 0.6}
    assert seen == [("noul", 2), ("choice", 2), ("score", 3)]


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "?", "criteria": {f"o{i}": "" for i in range(17)}},
    {"type": "score", "instructions": "?", "criteria": [f"l{i}" for i in range(7)]},
])
def test_more_options_than_slots_is_refused_not_truncated(q):
    status, body = srv.answer(lambda *a: pytest.fail("decided"), threading.Lock(), "n",
                              {"state": "s", "questions": {"q": q}})
    assert status == 422 and body["detail"]["code"] == "TOO_MANY_OPTIONS"


def test_sixteen_options_fit():
    q = {"type": "choice", "instructions": "?", "criteria": {f"o{i}": "" for i in range(16)}}
    _, prompt, labels = srv.to_prompt("s", "q", q)
    assert len(labels) == 16 and "P) o15\n[decision]:" in prompt


@pytest.mark.parametrize("body", [{"questions": QUESTIONS}, {"state": "s", "questions": {}},
                                  {"state": "s", "questions": {"q": {"type": "box"}}}])
def test_bad_requests_are_400(body):
    status, _ = srv.answer(lambda *a: pytest.fail("decided"), threading.Lock(), "n", body)
    assert status == 400


def test_a_json_state_is_sent_as_json():
    _, prompt, _ = srv.to_prompt(srv.as_text({"msg": "비"}), "u", QUESTIONS["urgency"])
    assert '[state] {"msg": "비"}\n' in prompt


def test_a_model_failure_is_a_500_not_a_dropped_connection():
    def decide(*a):
        raise RuntimeError("MPS out of memory")

    status, body = srv.answer(decide, threading.Lock(), "n", {"state": "s", "questions": QUESTIONS})
    assert status == 500 and "MPS out of memory" in body["detail"]


def test_the_bench_measures_it_over_http(monkeypatch):
    server = srv.make_server(lambda kind, prompt, n: [1 / n] * n, "JEV-9B@7ac669b", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setitem(run.ENGINES, "jev-9b", {**run.ENGINES["jev-9b"], "url": url})
    try:
        assert run.served("jev-9b") == "JEV-9B@7ac669b"
        answers, _ = run.ask("jev-9b", "s", QUESTIONS)
        assert answers["department"]["choice"] == "billing"  # a tie goes to the first option
    finally:
        server.shutdown()
