#!/usr/bin/env bash
# Serve OneJev 1.6 4B (Apache-2.0) with OneJev's `qev serve` over llama.cpp: qev builds the prompts and reads the
# answer from the next-token logprobs of a llama-server child it starts, and answers /v1/systemone in TypeSafe's wire
# format on localhost:${ONEJEV_PORT:-8070}. The GGUF is mradermacher's Q8_0 at a pinned revision, text only
# (its mmproj is not downloaded); the tokenizer and calibration come from OmniJev/OneJev1.6-4B at a pinned revision. The
# name (GGUF file plus repo revision) is what bench/run.py checks /health against.
#   uv tool install --python 3.12 "qev @ git+https://github.com/OmniJev/OneJev@9d59064ed540f2b9bcbd2a3e5390b326ebf5fa15"
#   scripts/serve-onejev.sh
#   scripts/serve-onejev.sh --print-name  # the model name /health reports
# llama-server is llama.cpp b12282 (LLAMA_SERVER overrides it), unpacked like scripts/serve-llama.sh's b11403 from
# https://github.com/ggml-org/llama.cpp/releases/tag/b12282: b11403 refuses this GGUF (its header declares an MTP layer,
# block_count 33 and nextn_predict_layers 1, whose tensors the file does not ship; b12282 loads it).
set -euo pipefail

build=12282 qev_version=0.1.4
repo=mradermacher/OneJev1.6-4B-GGUF file=OneJev1.6-4B.Q8_0.gguf rev=2b605ac93b0fc86f50e4836ac1eb0d33f1311bac
tok_repo=OmniJev/OneJev1.6-4B tok_rev=a16abc067498ff3a15411bdb82f2517dbe43c565
name="${file%.gguf}@${rev:0:7}"
if [ "${1:-}" = --print-name ]; then
    echo "$name"
    exit 0
fi

source "$(dirname "$0")/lib.sh"
port=${ONEJEV_PORT:-8070}
require_free_port "$port"

installed=$(qev --help > /dev/null 2>&1 && uv tool list 2>/dev/null | awk '$1 == "qev" {print $2}') || true
if [ "$installed" != "v$qev_version" ]; then
    echo "qev is not $qev_version: ${installed:-not installed} (see the header of $0)" >&2
    exit 1
fi
llama=${LLAMA_SERVER:-$HOME/.local/opt/llama.cpp/b$build/llama-b$build/llama-server}
version=$("$llama" --version 2>&1) || true
if [[ $version != *"build $build,"* ]]; then
    echo "$llama is not llama.cpp build $build: $version" >&2
    exit 1
fi

# Cached after the first download; the revisions pin the exact files. The tokenizer folder skips the BF16 weights.
cd "$(dirname "$0")/.."
gguf=$(uv run hf download "$repo" "$file" --revision "$rev" --quiet)
tok=$(uv run hf download "$tok_repo" --revision "$tok_rev" --exclude '*.safetensors' --quiet)

exec qev serve --gguf "$gguf" --tokenizer "$tok" --llama-server "$llama" --name "$name" \
    --host 127.0.0.1 --port "$port"
