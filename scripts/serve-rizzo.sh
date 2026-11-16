#!/usr/bin/env bash
# Serve Rizzo Flow 4B (Apache-2.0; Spark-X2.5-4B plus its typed-decisions LoRA, merged, Q8_0 GGUF) with its own
# `rizzo serve`, whose /v1/systemone answers in TypeSafe's wire format, on localhost:${RIZZO_PORT:-8017}. rizzo-flow pins
# the GGUF (rizzoaiacademy/rizzo-flow rev 55633c8, sha256 checked on download) and its llama.cpp b11081 Metal runtime,
# both downloaded into ${RIZZO_HOME:-~/.local/opt/rizzo-flow} (`rizzo` reads them relative to the working directory).
#   uv tool install "rizzo-flow @ git+https://github.com/Rizzo-AI-Academy/rizzo-flow@b9ba007ee4d2928bbab5b1d8bfe9009c3696b6de"
#   scripts/serve-rizzo.sh
set -euo pipefail

source "$(dirname "$0")/lib.sh"
port=${RIZZO_PORT:-8017}
require_free_port "$port"

commit=b9ba007ee4d2928bbab5b1d8bfe9009c3696b6de
installed=$(uv tool list --show-version-specifiers 2>/dev/null | awk '$1 == "rizzo-flow"') || true
case "$installed" in
    *"@$commit"*) ;;
    *) echo "rizzo-flow is not ${commit:0:7}: ${installed:-not installed}" >&2; exit 1 ;;
esac

home=${RIZZO_HOME:-$HOME/.local/opt/rizzo-flow}
mkdir -p "$home"
cd "$home"
# Skips files already present and checked.
"${RIZZO:-rizzo}" download --size 4b --quant q8_0 >&2
exec "${RIZZO:-rizzo}" serve --size 4b --quant q8_0 --host 127.0.0.1 --port "$port"
