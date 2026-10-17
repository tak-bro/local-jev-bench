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
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=timeout)
    try:
        conn.request("GET", path)
        return conn.getresponse()
    finally:
        conn.close()


def test_bind_does_not_resolve_the_host(monkeypatch, tmp_path):
    # getfqdn() between bind() and listen() leaves the port refusing connections for the lookup's whole
    # stall; the stand-ins must bind like the adapter does (serve/anyjev_server.py, LocalHTTPServer).
    monkeypatch.setattr(socket, "getfqdn", lambda host: pytest.fail(f"server_bind resolved {host}"))
    (tmp_path / "ready").write_text("ok")
    server = standin_server.LocalHTTPServer(("127.0.0.1", 0), standin_server.handler(str(tmp_path)))
    try:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        assert get(server).status == 200
    finally:
        server.shutdown()
        server.server_close()


def test_serves_the_directory(tmp_path):
    (tmp_path / "ready").write_text("ok")
    server = standin_server.LocalHTTPServer(("127.0.0.1", 0), standin_server.handler(str(tmp_path)))
    try:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        r = get(server)
        assert r.status == 200 and r.read() == b"ok"
    finally:
        server.shutdown()
        server.server_close()
