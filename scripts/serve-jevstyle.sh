#!/usr/bin/env bash
# Serve a Jev-Style decision model (Apache-2.0) with its own `jev-style serve`, whose /v1/systemone answers in
# TypeSafe's wire format, on localhost:${JEVSTYLE_PORT:-8080}, on the MLX backend (bf16 weights). jev-style 0.6.3 pins
# each release's MLX repo revision itself (jev_style/models.py), so the version check pins the weights; /healthz
# reports the release's model id, which bench/run.py checks before it measures.
#   uv tool install "jev-style[mlx]==0.6.3"
#   scripts/serve-jevstyle.sh 0.8b               # Jev-Style-0.8B-Decision-v3 (MLX rev 7f14c9f)
#   scripts/serve-jevstyle.sh 2b                 # Jev-Style-2B-Decision-v4 (MLX rev a14bbe4), text only on MLX
#   scripts/serve-jevstyle.sh 0.8b --print-name  # the model id /healthz reports
set -euo pipefail

version=0.6.3
case "${1:-}" in
    0.8b) release=0.8b-v3 ;;
    2b) release=2b-v4 ;;
    *) echo "usage: $0 0.8b|2b [--print-name]" >&2; exit 2 ;;
esac
name="jev-style-$release"
name="${name%-*}-decision-${release##*-}"
if [ "${2:-}" = --print-name ]; then
    echo "$name"
    exit 0
fi

source "$(dirname "$0")/lib.sh"
port=${JEVSTYLE_PORT:-8080}
require_free_port "$port"

installed=$(uv tool list 2>/dev/null | awk '$1 == "jev-style" {print $2}') || true
if [ "$installed" != "v$version" ]; then
    echo "jev-style is not $version: ${installed:-not installed}" >&2
    exit 1
fi

# The first start downloads the pinned MLX weights into the Hugging Face cache; later starts reuse them.
exec "${JEVSTYLE:-jev-style}" serve --release "$release" --backend mlx --host 127.0.0.1 --port "$port"
