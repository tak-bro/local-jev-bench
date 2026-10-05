#!/usr/bin/env bash
# Serve Von's System One API (github.com/wfzyx/von, a 395M ModernBERT encoder) on localhost:${VON_PORT:-8030}.
# `von serve` speaks /v1/systemone itself; uvx runs the pinned von-sdk in its own environment, and the weights are
# pinned to one revision of huggingface.co/wfzyx/von.
#   scripts/serve-von.sh
# --noul-decision raw: by default Von pushes every noul probability outside 0.2-0.8 (0.8 + 0.1(p - 0.5)), which
# keeps the decision but not the calibration the other engines are scored on. --on-overflow refuse: a state over
# the window is an error, not a silently truncated input (bench/run.py already refuses over 2048 tokens).
set -euo pipefail

sdk=1.3.7
rev=498ceba33390b32cfefaab6422ec380318ba9b99

source "$(dirname "$0")/lib.sh"
port=${VON_PORT:-8030}
require_free_port "$port"

weights=$(cd "$(dirname "$0")/.." && uv run hf download wfzyx/von --revision "$rev" --quiet)
export VON_MODEL_ID=$weights VON_CHECKPOINT_DIR=$weights PYTHONUNBUFFERED=1
exec uvx --from "von-sdk==$sdk" von serve \
    --host 127.0.0.1 --port "$port" \
    --device "${VON_DEVICE:-mps}" \
    --noul-decision raw \
    --on-overflow refuse
