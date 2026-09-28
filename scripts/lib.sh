# Sourced by the serve scripts.

# Exit 1 when the port cannot be bound: a stale server would silently answer the smoke checks.
# Try an actual bind rather than looking for LISTEN: vLLM binds before it listens, and a hung one stays that way.
# The bind sets SO_REUSEADDR, as the Python adapters' HTTPServer does, so TIME_WAIT sockets left by a just-stopped
# server do not count. That also lets 127.0.0.1 bind beside a listener on 0.0.0.0, which would still answer, so a
# port that accepts a connection is refused too.
require_free_port() {
    if ! python3 - "$1" 2>/dev/null <<'PY'
import socket, sys
port = int(sys.argv[1])
with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", port)) == 0:
        sys.exit(1)
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port))
PY
    then
        echo "port $1 is already in use:" >&2
        lsof -nP -iTCP:"$1" >&2 || true
        exit 1
    fi
}
