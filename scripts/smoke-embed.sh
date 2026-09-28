#!/usr/bin/env bash
# Ask a pooling server for one embedding and check its shape and norm.
#   scripts/smoke-embed.sh <port> [expected-dim]
# Exits 1 when the server is down, returns no vector, the dimension differs,
# or the vector is not L2-normalised (CLM's head was trained on normalised last-token embeddings).
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <port> [expected-dim]" >&2
    exit 2
fi
base="http://127.0.0.1:$1"

models=$(curl -fsS --max-time 10 "$base/v1/models") || { echo "no server on $base" >&2; exit 1; }
name=$(printf '%s' "$models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')

body=$(printf '{"model": "%s", "input": ["Customer: my invoice was charged twice."]}' "$name")
resp=$(curl -fsS --max-time 120 -H 'Content-Type: application/json' -d "$body" "$base/v1/embeddings") ||
    { echo "embedding request to $base failed" >&2; exit 1; }

printf '%s' "$resp" | python3 -c '
import json, math, sys
want = int(sys.argv[1]) if sys.argv[1] else None
vec = json.load(sys.stdin)["data"][0]["embedding"]
dim, norm = len(vec), math.sqrt(sum(x * x for x in vec))
print(f"model={sys.argv[2]} dim={dim} l2={norm:.4f}")
if dim == 0:
    sys.exit("empty embedding")
if want is not None and dim != want:
    sys.exit(f"dimension {dim} != expected {want}")
if not 0.99 <= norm <= 1.01:
    sys.exit(f"L2 norm {norm:.4f} is outside 0.99..1.01 (not normalised)")
' "${2:-}" "$name"
