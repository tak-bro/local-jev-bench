#!/usr/bin/env bash
# Serve an embedding model through vllm-metal's pooling runner, bound to localhost.
#   scripts/serve-embed.sh mlx-community/Qwen3-Embedding-0.6B-8bit 8091 embed-small
#   scripts/serve-embed.sh Qwen/Qwen3-8B 8090 qwen3-8b
# clm-serve expects the second form: served name `qwen3-8b` on :8090 with last-token pooling.
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "usage: $0 <model> <port> <served-name>" >&2
    exit 2
fi
model=$1 port=$2 name=$3

source "$(dirname "$0")/lib.sh"
require_free_port "$port"

# vllm-metal's pooling docs run the server with V1 multiprocessing off.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
exec vllm serve "$model" \
    --served-model-name "$name" \
    --runner pooling \
    --max-model-len "${MAX_MODEL_LEN:-2048}" \
    --host 127.0.0.1 \
    --port "$port"
