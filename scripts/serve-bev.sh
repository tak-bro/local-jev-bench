#!/usr/bin/env bash
# Serve bev-decider-0.4B (CC BY-NC 4.0; the first 20 layers of Qwen3-0.6B plus an order-invariant decision head) with
# its own `bev-decider serve`, whose /v1/systemone answers in TypeSafe's wire format, on
# localhost:${BEV_PORT:-8100}, on Metal (MPS, the package's bf16 default). /v1/models reports only the model name, so
# the weights' revision is pinned here.
#   uv tool install "bev-decider[serve]==0.2.1"
#   scripts/serve-bev.sh
set -euo pipefail

version=0.2.1
repo=avbiswas/bev-decider-0.4B rev=cf5138b5b465343d09c3b4fbd71eaca36d989d34

source "$(dirname "$0")/lib.sh"
port=${BEV_PORT:-8100}
require_free_port "$port"

server=${BEV:-bev-decider}
installed=$(uv tool list 2>/dev/null | awk '$1 == "bev-decider" {print $2}') || true
if [ "$installed" != "v$version" ]; then
    echo "bev-decider is not $version: ${installed:-not installed}" >&2
    exit 1
fi

# Cached after the first download; the revision pins the exact files.
model=$(cd "$(dirname "$0")/.." && uv run hf download "$repo" --revision "$rev" --quiet)

exec "$server" serve --model "$model" --host 127.0.0.1 --port "$port"
