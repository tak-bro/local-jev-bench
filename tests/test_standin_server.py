"""The stand-in file server binds without resolving the host — macos-latest stalls the loopback lookup ~35 s."""

from __future__ import annotations

import http.client
import socket
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import standin_server  # noqa: E402


def get(server, path="/ready", timeout=5):
    """(status, body) — the body is read before the connection closes."""
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=timeout)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


@pytest.fixture
def served(monkeypatch, tmp_path):
    """A started stand-in server; getfqdn is patched to fail, so any bind-time lookup fails the test."""
    monkeypatch.setattr(socket, "getfqdn", lambda host: pytest.fail(f"server_bind resolved {host}"))
    (tmp_path / "ready").write_text("ok")
    server = standin_server.LocalHTTPServer(("127.0.0.1", 0), standin_server.handler(str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_bind_does_not_resolve_the_host(served):
    assert get(served)[0] == 200


def test_serves_the_directory(served):
    assert get(served) == (200, b"ok")
