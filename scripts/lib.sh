# Sourced by the serve scripts.

# Exit 1 when something already listens on the port: a second server would fail to bind anyway,
# and a stale one would silently answer the smoke checks.
require_free_port() {
    if lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "port $1 is already in use:" >&2
        lsof -nP -iTCP:"$1" -sTCP:LISTEN >&2
        exit 1
    fi
}
