"""A stdlib file server whose bind() never resolves the host — the test stand-ins start this instead of
`python3 -m http.server`.

`http.server.HTTPServer.server_bind()` calls `socket.getfqdn(host)` between `bind()` and `listen()`. On a host
whose loopback reverse lookup stalls — macOS 15+/26 CI runners block it for ~35 s (actions/setup-python#1223)
— the port is bound but refuses every connection until the lookup returns, and every readiness wait sees a
server that is not up. Here `server_name` comes from the address; routes and behaviour are http.server's.

CLI: python3 tests/standin_server.py <port> <directory>
"""

from __future__ import annotations

import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer


class LocalHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that binds the address as given instead of resolving it."""

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def handler(directory: str):
    """SimpleHTTPRequestHandler over `directory` — the `--directory` of `python3 -m http.server`."""
    return functools.partial(SimpleHTTPRequestHandler, directory=directory)


def serve(port: int, directory: str) -> None:
    LocalHTTPServer(("127.0.0.1", port), handler(directory)).serve_forever()


def main() -> None:
    serve(int(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
