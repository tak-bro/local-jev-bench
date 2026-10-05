#!/usr/bin/env bash
# Serve a Clef GGUF with llama.cpp's llama-server, whose /v1/systemone answers in TypeSafe's wire format, on
# localhost:${LLAMA_PORT:-8020}. The alias (GGUF file plus repo revision) is what bench/run.py checks /v1/models
# against before it measures.
#   scripts/serve-llama.sh clef-flash                # ggml-org/Clef-Flash-GGUF Q8_0
#   scripts/serve-llama.sh --print-alias clef-flash  # the model name the bench sends and expects
# llama-server comes from the b11403 release, unpacked so that LLAMA_SERVER (default below) points at it:
# https://github.com/ggml-org/llama.cpp/releases/tag/b11403. /v1/systemone (ggml-org/llama.cpp#29818) and the clef
# architecture (#29831) were merged on 2026-10-02 and 10-03, after Homebrew's 0.5.0 (build 11146).
set -euo pipefail

build=11403
print_alias=
if [ "${1:-}" = --print-alias ]; then
    print_alias=1
    shift
fi
case ${1:-} in
    clef-flash) repo=ggml-org/Clef-Flash-GGUF file=Clef-Flash-Q8_0.gguf rev=4a7a08c09bc63baf043b62b5ba89dd67a0357d95 ;;
    *)
        echo "unknown engine: ${1:-} (clef-flash)" >&2
        exit 2
        ;;
esac
alias="${file%.gguf}@${rev:0:7}"
if [ -n "$print_alias" ]; then
    echo "$alias"
    exit 0
fi

source "$(dirname "$0")/lib.sh"
port=${LLAMA_PORT:-8020}
require_free_port "$port"

server=${LLAMA_SERVER:-$HOME/.local/opt/llama.cpp/b$build/llama-b$build/llama-server}
version=$("$server" --version 2>&1) || true
if [[ $version != *"build $build,"* ]]; then
    echo "$server is not llama.cpp build $build: $version" >&2
    exit 1
fi

# Cached after the first download; the revision pins the exact file.
gguf=$(cd "$(dirname "$0")/.." && uv run hf download "$repo" "$file" --revision "$rev" --quiet)

# Clef evaluates the whole prompt (state, every question and option) in one batch, so it must fit in --ubatch-size.
# One slot: the other engines are measured one call at a time too.
exec "$server" -m "$gguf" --alias "$alias" \
    --host 127.0.0.1 --port "$port" \
    -ngl 99 -np 1 -c 8192 -b 8192 -ub 8192
