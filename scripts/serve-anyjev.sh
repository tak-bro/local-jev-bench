#!/usr/bin/env bash
# Serve AnyJev's System One API on localhost:${ANYJEV_PORT:-8710}, reading label logprobs from the Qwen3-8B
# generate server that scripts/serve-llm.sh runs on :8092.
#   scripts/serve-llm.sh Qwen/Qwen3-8B 8092 qwen3-8b && scripts/serve-anyjev.sh
set -euo pipefail

source "$(dirname "$0")/lib.sh"
port=${ANYJEV_PORT:-8710}
require_free_port "$port"

cd "$(dirname "$0")/.."
exec uv run python serve/anyjev_server.py \
    --port "$port" \
    --llm-url "${ANYJEV_LLM_URL:-http://127.0.0.1:8092}" \
    --llm-model "${ANYJEV_LLM_MODEL:-qwen3-8b}" \
    --tokenizer "${ANYJEV_TOKENIZER:-Qwen/Qwen3-8B}"
