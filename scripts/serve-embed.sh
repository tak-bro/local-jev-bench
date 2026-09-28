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
# Without these, torch.distributed (Gloo) fails to resolve the .local hostname and the engine hangs at init.
export GLOO_SOCKET_IFNAME=lo0 VLLM_HOST_IP=127.0.0.1
# Pin last-token pooling with L2 normalisation (use_activation) instead of the architecture fallback:
# CLM's head was trained on exactly that.
# The default --gpu-memory-utilization (0.92) reserves nearly all unified memory for KV cache an embedding
# server barely uses; at 0.92 Qwen3-8B alone pushed macOS memory pressure to warn (M3 Max, 36 GB).
# vllm-metal applies the fraction to the Metal wired limit (28.1 GB here), so it must cover the ~16 GB of
# bf16 weights: 0.55 left a negative KV budget, 0.7 leaves about 3 GB.
exec vllm serve "$model" \
    --served-model-name "$name" \
    --runner pooling \
    --pooler-config '{"seq_pooling_type": "LAST", "use_activation": true}' \
    --max-model-len "${MAX_MODEL_LEN:-2048}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.7}" \
    --host 127.0.0.1 \
    --port "$port"
