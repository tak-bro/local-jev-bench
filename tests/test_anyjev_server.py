"""Offline checks for the AnyJev adapter: AnyJev's own fake backend stands in for the LLM server."""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest
from anyjev import Decider
from anyjev.backends.fake import FakeBackend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))
sys.path.insert(0, str(ROOT / "tests"))
import anyjev_server as srv  # noqa: E402
from test_offline import FakeServer, run  # noqa: E402

QUESTIONS = {
    "urgency": {"type": "noul", "instructions": "Is this urgent?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}


def billing_backend(**kw):
    return FakeBackend(lambda state, option: 3.0 if "billing" in option or "Very angry" in option else 0.0, **kw)


@pytest.fixture
def adapter(monkeypatch):
    """Start the adapter over a decider and point the bench's anyjev engines at it."""
    servers = []

    def start(decider):
        server = srv.make_server(decider, port=0)
        servers.append(server)
        srv.serve_in_thread(server)
        url = f"http://127.0.0.1:{server.server_address[1]}"
        for name in ("anyjev-raw", "anyjev-l0"):
            monkeypatch.setitem(run.ENGINES, name, {**run.ENGINES[name], "url": url})
        return url

    yield start
    for server in servers:
        server.shutdown()


def post(url, body):
    import requests
    return requests.post(f"{url}/v1/systemone", json=body, timeout=30)


def test_answers_every_type_in_the_bench_shape(adapter):
    adapter(Decider(billing_backend()))
    answers, _ = run.ask("anyjev-l0", "Customer: charged twice!", QUESTIONS)  # check_shape runs inside ask
    assert answers["department"]["choice"] == "billing"
    assert set(answers["department"]["probabilities"]) == {"billing", "technical"}
    assert 0.0 <= answers["urgency"]["noul"] <= 1.0
    assert 0.0 <= answers["frustration"]["score"] <= 2.0  # level index, as the bench rounds it


class SpyDecider(Decider):
    def __init__(self, backend):
        super().__init__(backend)
        self.levels = []

    def decide(self, state, questions, level=None, require=None):
        self.levels.append(level)
        return super().decide(state, questions, level=level, require=require)


@pytest.mark.parametrize("engine, level", [("anyjev-raw", "raw"), ("anyjev-l0", "L0")])
def test_model_field_picks_the_level(adapter, engine, level):
    decider = SpyDecider(billing_backend())
    adapter(decider)
    run.ask(engine, "s", {"urgency": QUESTIONS["urgency"]})
    assert decider.levels == [level]


def test_unknown_model_is_rejected(adapter):
    decider = SpyDecider(billing_backend())
    url = adapter(decider)
    r = post(url, {"model": "anyjev-l1", "state": "s", "questions": {"urgency": QUESTIONS["urgency"]}})
    assert r.status_code == 400 and "anyjev-l1" in r.json()["detail"]
    assert decider.levels == []  # no default level was used


def test_unknown_question_type_is_rejected(adapter):
    url = adapter(Decider(billing_backend()))
    r = post(url, {"model": "anyjev-l0", "state": "s", "questions": {"x": {"type": "ranking", "instructions": "?"}}})
    assert r.status_code == 400 and "ranking" in r.json()["detail"]


def test_missing_state_is_rejected(adapter):
    url = adapter(Decider(billing_backend()))
    r = post(url, {"model": "anyjev-l0", "questions": {"urgency": QUESTIONS["urgency"]}})
    assert r.status_code == 400 and "state" in r.json()["detail"]


class RecordingBackend(FakeBackend):
    def __init__(self):
        super().__init__(lambda state, option: 0.0)
        self.prompts = []

    def next_token_logprobs(self, prompts, token_ids):
        self.prompts.extend(prompts)
        return super().next_token_logprobs(prompts, token_ids)


def test_choice_descriptions_reach_the_model(adapter):
    backend = RecordingBackend()
    adapter(Decider(backend))
    run.ask("anyjev-raw", "s", {"department": QUESTIONS["department"]})
    assert any("Charges, invoices, refunds" in p for p in backend.prompts)


class DownBackend(FakeBackend):
    def __init__(self, exc):
        super().__init__(lambda state, option: 0.0)
        self.exc = exc

    def next_token_logprobs(self, prompts, token_ids):
        raise self.exc


@pytest.mark.parametrize("exc", [urllib.error.URLError("refused"), srv.LabelMissing("no label tokens"), TimeoutError()])
def test_backend_failures_are_502(adapter, exc):
    url = adapter(Decider(DownBackend(exc)))
    r = post(url, {"model": "anyjev-l0", "state": "s", "questions": {"urgency": QUESTIONS["urgency"]}})
    assert r.status_code == 502
    with pytest.raises(run.EngineError, match="HTTP 502"):
        run.ask("anyjev-l0", "s", {"urgency": QUESTIONS["urgency"]})


def test_unexpected_backend_error_is_answered(adapter):
    # e.g. a 200 whose logprobs are null: the bench must see an HTTP error, not a dropped connection
    url = adapter(Decider(DownBackend(KeyError("top_logprobs"))))
    r = post(url, {"model": "anyjev-l0", "state": "s", "questions": {"urgency": QUESTIONS["urgency"]}})
    assert r.status_code == 500 and "top_logprobs" in r.json()["detail"]


class StubTokenizer:
    def decode(self, ids):
        return {1: "A", 2: "B"}[ids[0]]

    def convert_ids_to_tokens(self, tid):
        return {1: "A", 2: "B"}[tid]


def completions_server(top, seen=None):
    def reply(body):
        if seen is not None:
            seen.append(body)
        return 200, {"choices": [{"logprobs": {"top_logprobs": [top]}}]}
    return FakeServer({"/v1/completions": reply})


def strict_backend(monkeypatch, port):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name: StubTokenizer())
    return srv.StrictVLLMBackend(f"http://127.0.0.1:{port}", "qwen3-8b", workers=1, timeout=5)


def test_strict_backend_rejects_missing_label_tokens(monkeypatch):
    server = completions_server({"Hello": -0.1, "B": -2.0})  # "A" was not returned
    try:
        with pytest.raises(srv.LabelMissing):
            strict_backend(monkeypatch, server.port).next_token_logprobs(["p"], [[1, 2]])
    finally:
        server.close()


def test_strict_backend_asks_for_each_label_logprob(monkeypatch):
    # vllm-metal 0.30.0 reports raw logprobs whatever --logprobs-mode says, so top-K under allowed_token_ids can
    # leave a label out; logprob_token_ids returns exactly the labels.
    seen = []
    server = completions_server({"A": -8.0, "B": -9.5}, seen)
    try:
        [lp] = strict_backend(monkeypatch, server.port).next_token_logprobs(["p"], [[1, 2]])
    finally:
        server.close()
    assert seen[0]["logprob_token_ids"] == [1, 2] and seen[0].get("logprobs") is not None and seen[0]["max_tokens"] == 1
    assert list(lp) == [-8.0, -9.5]  # raw values pass through; the decider's softmax normalises them
