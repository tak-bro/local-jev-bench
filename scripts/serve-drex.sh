#!/usr/bin/env bash
# Serve Nace.AI's Drex v1.5 (9B, Q8_0 GGUF) with the llama-server of Nace's llama.cpp fork, whose /v1/systemone
# answers in TypeSafe's wire format, on localhost:${DREX_PORT:-8040}. The alias (GGUF file plus repo revision) is
# what bench/run.py checks /v1/models against before it measures.
#   scripts/serve-drex.sh
#   scripts/serve-drex.sh --print-alias  # the model name the bench sends and expects
# llama-server is built from the fork's drex-v1.5 branch (commit below) so that DREX_SERVER (default below) points
# at it; upstream llama.cpp does not load Drex's pointer head:
#   git clone --depth 1 -b drex-v1.5 https://github.com/nace-ai/llama.cpp ~/workspace/ext/nace-llama.cpp
#   cmake -S ~/workspace/ext/nace-llama.cpp -B ~/workspace/ext/nace-llama.cpp/build -DGGML_METAL=ON -DLLAMA_CURL=OFF \
#       -DCMAKE_BUILD_TYPE=Release
#   cmake --build ~/workspace/ext/nace-llama.cpp/build -j 12 --target llama-server
set -euo pipefail

commit=e8f7961
repo=nace-ai/drex-v1.5-Q8_0 file=drex-v1.5-Q8_0.gguf rev=37c2b59a5b8ecb95b19e6bfae2d4041e0de19808
alias="${file%.gguf}@${rev:0:7}"
if [ "${1:-}" = --print-alias ]; then
    echo "$alias"
    exit 0
fi

source "$(dirname "$0")/lib.sh"
port=${DREX_PORT:-8040}
require_free_port "$port"

server=${DREX_SERVER:-$HOME/workspace/ext/nace-llama.cpp/build/bin/llama-server}
version=$("$server" --version 2>&1) || true
if [[ $version != *"commit $commit"* ]]; then
    echo "$server is not the Nace fork at commit $commit: $version" >&2
    exit 1
fi

# Cached after the first download; the revision pins the exact file.
gguf=$(cd "$(dirname "$0")/.." && uv run hf download "$repo" "$file" --revision "$rev" --quiet)

# The model card's flags (embedding readout, no pooling), with one slot like every other engine here and -ngl 99 for
# Metal. 16,384 tokens per request.
exec "$server" -m "$gguf" --alias "$alias" \
    --host 127.0.0.1 --port "$port" \
    --embedding --pooling none -ngl 99 -np 1 -c 16384 -b 16384 -ub 2048
