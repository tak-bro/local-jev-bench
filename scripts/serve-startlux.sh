#!/usr/bin/env bash
# Serve StartLux-Decision v1.1 4B Instant (CC BY-NC 4.0) with its own `startlux-decide serve`, whose /v1/systemone
# answers in TypeSafe's wire format, on localhost:${STARTLUX_PORT:-8060}, on the MLX backend (8-bit build). The name
# (folder plus repo revision) is what bench/run.py checks /health against before it measures.
#   uv tool install "startlux-decide[mlx] @ git+https://github.com/StartLuxLabs/StartLux-Decision@v1.1.0"
#   scripts/serve-startlux.sh
#   scripts/serve-startlux.sh --print-name  # the model name /health reports
set -euo pipefail

version=1.1.0
repo=startlux-models/StartLux-Decision-v1.1-4B-Instant-MLX-8bit rev=56f18acdc9502c44fdb80234c09a2276e3398916
name="${repo#*/}@${rev:0:7}"
if [ "${1:-}" = --print-name ]; then
    echo "$name"
    exit 0
fi

source "$(dirname "$0")/lib.sh"
port=${STARTLUX_PORT:-8060}
require_free_port "$port"

server=${STARTLUX:-startlux-decide}
installed=$(uv tool list 2>/dev/null | awk '$1 == "startlux-decide" {print $2}') || true
if [ "$installed" != "v$version" ]; then
    echo "startlux-decide is not $version: ${installed:-not installed}" >&2
    exit 1
fi

# Cached after the first download; the revision pins the exact files.
model=$(cd "$(dirname "$0")/.." && uv run hf download "$repo" --revision "$rev" --quiet)

exec "$server" serve --model "$model" --name "$name" --backend mlx --host 127.0.0.1 --port "$port"
