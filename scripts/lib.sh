# Sourced by the serve scripts.

# Exit 1 when the port cannot be bound: a stale server would silently answer the smoke checks.
# Try an actual bind rather than looking for LISTEN: vLLM binds before it listens, and a hung one stays that way.
require_free_port() {
    if ! python3 -c 'import socket, sys; socket.socket().bind(("127.0.0.1", int(sys.argv[1])))' "$1" 2>/dev/null; then
        echo "port $1 is already in use:" >&2
        lsof -nP -iTCP:"$1" >&2 || true
        exit 1
    fi
}
