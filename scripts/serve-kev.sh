#!/usr/bin/env bash
# Serve Kev's System One API on localhost:${KEV_PORT:-8009} from a checkout of github.com/jaredpalmer/kev, which
# keeps its own uv environment (torch, mlx-lm). On Apple Silicon it runs on MLX.
#   git clone https://github.com/jaredpalmer/kev ~/workspace/tak-bro/kev && (cd ~/workspace/tak-bro/kev && uv sync --extra serve)
#   KEV_DIR=~/workspace/tak-bro/kev scripts/serve-kev.sh
#   KEV_RUN=jaredpalmer/kev-9b scripts/serve-kev.sh   # or kev-0.8b; measure it as --engine kev-9b
set -euo pipefail

source "$(dirname "$0")/lib.sh"
port=${KEV_PORT:-8009}
require_free_port "$port"

kev_dir=${KEV_DIR:-$HOME/workspace/tak-bro/kev}
if [ ! -f "$kev_dir/kev/serve.py" ]; then
    echo "no Kev checkout at $kev_dir (set KEV_DIR)" >&2
    exit 1
fi

cd "$kev_dir"
export PYTHONUNBUFFERED=1  # Kev prints its "serving ... via <backend>" line without flushing
exec uv run --extra serve python -m kev.serve \
    --run "${KEV_RUN:-jaredpalmer/kev-4b}" \
    --host 127.0.0.1 \
    --port "$port"
