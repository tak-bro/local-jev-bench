#!/usr/bin/env bash
# Serve AutoTrust's JEV-9B (Apache-2.0; Qwen3.5-9B plus a System 1 LoRA and a 24-slot decision head) through
# serve/jev_server.py, our /v1/systemone adapter over the model card's transformers + peft path, on
# localhost:${JEV_PORT:-8110}, on Metal (MPS, bf16). The adapter keeps its own uv script environment (torch,
# transformers, peft). The name (repo plus revision) is what bench/run.py checks /health against before it measures.
#   scripts/serve-jev.sh
#   scripts/serve-jev.sh --print-name  # the model name /health reports
set -euo pipefail

repo=autotrust/JEV-9B rev=7ac669b8fb476ba926491e3d1c39f9330d68dbed
name="${repo#*/}@${rev:0:7}"
if [ "${1:-}" = --print-name ]; then
    echo "$name"
    exit 0
fi

source "$(dirname "$0")/lib.sh"
port=${JEV_PORT:-8110}
require_free_port "$port"

root=$(cd "$(dirname "$0")/.." && pwd)
# Cached after the first download (about 18 GB); the revision pins the exact files. Only the top-level files and
# the System 1 adapter: the card's reports, videos and vision and vLLM adapters are not needed here. `hf download`
# takes one pattern per --include.
model=$(cd "$root" && uv run hf download "$repo" --revision "$rev" --quiet \
    --include "*.json" --include "*.safetensors" --include "*.jinja" --include "adapter/*" \
    --exclude "*/*/*" --exclude "reports/*" --exclude "vl/*" --exclude "videos/*" --exclude "adapter_vllm/*")

exec uv run --script "$root/serve/jev_server.py" --port "$port" --model "$model" --name "$name"
