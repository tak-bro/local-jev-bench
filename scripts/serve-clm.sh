#!/usr/bin/env bash
# Serve CLM's System One API on localhost:${CLM_PORT:-8700}, using the Qwen3-8B pooling server on :8090.
# The first run downloads the 75 MB reference head from Hugging Face.
set -euo pipefail

source "$(dirname "$0")/lib.sh"
port=${CLM_PORT:-8700}
require_free_port "$port"

cd "$(dirname "$0")/.."
# clm-serve binds 0.0.0.0 by default; keep it local.
exec uv run clm-serve \
    --host 127.0.0.1 \
    --port "$port" \
    --emb-url http://127.0.0.1:8090/v1/embeddings \
    --emb-model qwen3-8b \
    --max-tokens "${MAX_MODEL_LEN:-2048}"
