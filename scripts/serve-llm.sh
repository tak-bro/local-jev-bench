#!/usr/bin/env bash
# Serve a generative model through vllm-metal (OpenAI-compatible chat), bound to localhost.
#   scripts/serve-llm.sh mlx-community/Qwen3-8B-4bit 8092 qwen3-8b-4bit
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "usage: $0 <model> <port> <served-name>" >&2
    exit 2
fi
model=$1 port=$2 name=$3

source "$(dirname "$0")/lib.sh"
require_free_port "$port"

# Same host settings as serve-embed.sh: loopback Gloo (the engine hangs resolving the .local hostname otherwise)
# and a memory fraction of the Metal wired limit that leaves the rest of the machine usable.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export GLOO_SOCKET_IFNAME=lo0 VLLM_HOST_IP=127.0.0.1
exec vllm serve "$model" \
    --served-model-name "$name" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.7}" \
    --host 127.0.0.1 \
    --port "$port"
