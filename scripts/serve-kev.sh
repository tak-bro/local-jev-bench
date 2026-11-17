#!/usr/bin/env bash
# Serve Kev's System One API on localhost:${KEV_PORT:-8009} from a checkout of github.com/jaredpalmer/kev, which
# keeps its own uv environment (torch, mlx-lm). On Apple Silicon it runs on MLX.
#   git clone https://github.com/jaredpalmer/kev ~/workspace/tak-bro/kev && cd ~/workspace/tak-bro/kev
#   git checkout 0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91   # the commit measured here; KEV_COMMIT overrides
#   uv sync --extra serve
#   scripts/serve-kev.sh kev-9b             # or kev-0.8b, kev-4b (the default); measure it as --engine kev-9b
#   scripts/serve-kev.sh --print-run kev-9b # the run the bench expects /v1/models to report
# Kev resolves `repo@revision` from the Hub at serve time; the revision below pins each adapter. KEV_RUN overrides
# the run (a local run directory or another revision), which bench/run.py then refuses as another model.
set -euo pipefail

print_run=
if [ "${1:-}" = --print-run ]; then
    print_run=1
    shift
fi
engine=${1:-kev-4b}
case $engine in
    kev-0.8b) rev=bf75a6a8848ea6960ff2ed108d9ed44c2941174f ;;
    kev-4b) rev=6cfce5c2fa4b4bd64026336ab649c5ca78857d52 ;;
    kev-9b) rev=db029f08b290afd9fee4aa4bbcd9ae48602d1eb0 ;;
    *)
        echo "unknown engine: $engine (kev-0.8b, kev-4b or kev-9b)" >&2
        exit 2
        ;;
esac
run=${KEV_RUN:-jaredpalmer/$engine@$rev}
if [ -n "$print_run" ]; then
    echo "$run"
    exit 0
fi

source "$(dirname "$0")/lib.sh"
port=${KEV_PORT:-8009}
require_free_port "$port"

kev_dir=${KEV_DIR:-$HOME/workspace/tak-bro/kev}
if [ ! -f "$kev_dir/kev/serve.py" ]; then
    echo "no Kev checkout at $kev_dir (set KEV_DIR)" >&2
    exit 1
fi

want=${KEV_COMMIT:-0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91}
head=$(git -C "$kev_dir" rev-parse HEAD 2>/dev/null || echo none)
if [ "$head" != "$want" ]; then
    echo "Kev checkout at $kev_dir is at $head, not $want (git checkout it, or set KEV_COMMIT)" >&2
    exit 1
fi

cd "$kev_dir"
export PYTHONUNBUFFERED=1  # Kev prints its "serving ... via <backend>" line without flushing
exec uv run --extra serve python -m kev.serve \
    --run "$run" \
    --host 127.0.0.1 \
    --port "$port"
