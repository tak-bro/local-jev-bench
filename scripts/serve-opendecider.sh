#!/usr/bin/env bash
# Serve OpenDecider-small-td (4B, Q8_0 GGUF) through `opendecider serve`, whose /v1/systemone answers in TypeSafe's
# wire format, on localhost:${OPENDECIDER_PORT:-8050}. The GGUF runs in Ollama (Metal); opendecider builds the
# prompt and reads the option probabilities from Ollama's token log-probabilities.
#   uv tool install "opendecider[serve]==0.9.0"
#   ollama pull hf.co/manjunathshiva/opendecider-small-td-GGUF:Q8_0
#   scripts/serve-opendecider.sh
set -euo pipefail

version=0.9.0
model=ollama:hf.co/manjunathshiva/opendecider-small-td-GGUF:Q8_0

source "$(dirname "$0")/lib.sh"
port=${OPENDECIDER_PORT:-8050}
require_free_port "$port"

server=${OPENDECIDER:-opendecider}
installed=$("$server" serve --help > /dev/null 2>&1 && uv tool list 2>/dev/null | awk '$1 == "opendecider" {print $2}') || true
if [ "$installed" != "v$version" ]; then
    echo "opendecider is not $version: ${installed:-not installed}" >&2
    exit 1
fi
if [ "$(curl -s -o /dev/null -m 2 -w '%{http_code}' "${OLLAMA_URL:-http://127.0.0.1:11434}/api/version")" != 200 ]; then
    echo "Ollama does not answer on ${OLLAMA_URL:-http://127.0.0.1:11434}; start it first" >&2
    exit 1
fi

# One question per forward pass (--small-batch 1, the exact default) and a 300 s request timeout like the bench's.
exec "$server" serve --model "$model" --host 127.0.0.1 --port "$port" --small-batch 1 --request-timeout-s 300
