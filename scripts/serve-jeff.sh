#!/usr/bin/env bash
# Serve Jeff's System One API on localhost:${JEFF_PORT:-8765} from a checkout of github.com/firelex/jeff, which keeps
# its own uv environment. MLX runs Jeff's Qwen models only, so the default is Jeff-Qwen3.5-2B.
#   git clone https://github.com/firelex/jeff ~/workspace/tak-bro/jeff && cd ~/workspace/tak-bro/jeff
#   git checkout f06788292874c21a5b5c41549ac220dd9e15da7f   # the commit measured here; JEFF_COMMIT overrides
#   uvx --from 'uv>=0.12.19' uv sync --no-default-groups --extra mac
#   uvx --from 'uv>=0.12.19' uv run --no-default-groups hf download mstrasser/Jeff-Qwen3.5-2B --local-dir checkpoints/jeff-2b
# Jeff's pyproject requires uv >= 0.12.19, hence uvx.
set -euo pipefail

source "$(dirname "$0")/lib.sh"
port=${JEFF_PORT:-8765}
require_free_port "$port"

jeff_dir=${JEFF_DIR:-$HOME/workspace/tak-bro/jeff}
checkpoint=${JEFF_CHECKPOINT:-checkpoints/jeff-2b}
if [ ! -f "$jeff_dir/$checkpoint/config.json" ]; then
    echo "no Jeff checkpoint at $jeff_dir/$checkpoint (set JEFF_DIR, JEFF_CHECKPOINT)" >&2
    exit 1
fi

want=${JEFF_COMMIT:-f06788292874c21a5b5c41549ac220dd9e15da7f}
head=$(git -C "$jeff_dir" rev-parse HEAD 2>/dev/null || echo none)
if [ "$head" != "$want" ]; then
    echo "Jeff checkout at $jeff_dir is at $head, not $want (git checkout it, or set JEFF_COMMIT)" >&2
    exit 1
fi

cd "$jeff_dir"
export JEFF_BACKEND=mlx JEFF_CHECKPOINT="$checkpoint" JEFF_HOST=127.0.0.1 PORT="$port"
exec uvx --from 'uv>=0.12.19' uv run --no-default-groups --extra mac jeff-serve
