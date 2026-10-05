#!/usr/bin/env bash
# Start one engine's servers, wait until they answer, benchmark it on each set, and stop them, even when a step fails.
# Engines that need most of the Metal memory are measured this way one at a time.
#   scripts/measure.sh kev-9b bench/data/transfer-v4.jsonl bench/data/typed-decisions.jsonl
#   scripts/measure.sh --reps 3 anyjev-l0 bench/questions.jsonl bench/questions_ko.jsonl
# The AnyJev adapter restarts before every set: its L0 batch prior accumulates over every call it serves.
# A server already answering on an engine's port is refused, except Ollaya's daemon: the request names the model, so
# a running daemon is used as it is and only that model is unloaded afterwards.
# Server output goes to logs/<name>.log. MEASURE_START and MEASURE_READY_URL replace the engine's servers with one
# command, MEASURE_SCRIPTS the directory of serve-*.sh, MEASURE_BENCH the bench command (tests, one-off servers).
set -euo pipefail

cd "$(dirname "$0")/.."
reps=()
if [ "${1:-}" = --reps ]; then
    reps=(--reps "$2")
    shift 2
fi
if [ $# -lt 2 ]; then
    echo "usage: $0 [--reps N] <engine> <set.jsonl>..." >&2
    exit 2
fi
engine=$1
shift
case $engine in
    ollaya | winnow | kev-0.8b | kev-4b | kev-9b | jeff | anyjev-raw | anyjev-l0 | clm | clef-flash | von) ;;
    *)
        echo "unknown engine: $engine" >&2
        exit 2
        ;;
esac

timeout=${MEASURE_READY_TIMEOUT:-600}
logs=${MEASURE_LOGS:-logs}
scripts=${MEASURE_SCRIPTS:-scripts}
read -ra bench <<< "${MEASURE_BENCH:-uv run python bench/run.py}"
mkdir -p "$logs"

pids=()
# stop <pid>: TERM the server's process group (the serve script and what it started), KILL it after 20 s.
stop() {
    local i
    kill -TERM -- "-$1" 2>/dev/null || return 0
    for ((i = 0; i < 200; i++)); do
        kill -0 -- "-$1" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 -- "-$1" 2>/dev/null; then
        echo "server $1 ignored TERM for 20s; killing it" >&2
        kill -KILL -- "-$1" 2>/dev/null || true
    fi
    wait "$1" 2>/dev/null || true
}
# shellcheck disable=SC2329  # called by the EXIT trap
stop_all() {
    local i
    unload  # while an Ollaya daemon this script started is still up: `ollaya stop` starts one if none answers
    for ((i = ${#pids[@]} - 1; i >= 0; i--)); do
        stop "${pids[i]}"
    done
    pids=()
}
# shellcheck disable=SC2329  # called by stop_all; Ollaya engines redefine it to unload their model
unload() { :; }
trap stop_all EXIT
trap 'exit 130' INT TERM

http_code() { curl -s -o /dev/null -m 2 -w '%{http_code}' "$1" || true; }

# start <name> <ready-url> <ok: 200|any> <command>...: run it in the background and wait until the URL answers
# (200: HTTP 200; any: any HTTP status, for a server with no GET route) while the process is still alive.
start() {
    local name=$1 url=$2 ok=$3 code pid deadline
    shift 3
    if [ "$(http_code "$url")" != 000 ]; then
        echo "$url already answers before $name starts; stop that server first" >&2
        exit 1
    fi
    set -m  # its own process group, so stopping it stops the children a serve script starts (uv run -> python)
    "$@" > "$logs/$name.log" 2>&1 &
    pid=$!
    set +m  # foreground commands stay in this group, so Ctrl-C reaches the trap
    pids+=("$pid")
    deadline=$((SECONDS + timeout))
    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name exited before it was ready; $logs/$name.log:" >&2
            tail -20 "$logs/$name.log" >&2
            exit 1
        fi
        code=$(http_code "$url")
        if [ "$code" = 200 ] || { [ "$ok" = any ] && [ "$code" != 000 ]; }; then
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "$name not ready after ${timeout}s; see $logs/$name.log" >&2
            exit 1
        fi
        sleep 1
    done
}

ollaya_url=${OLLAYA_URL:-http://127.0.0.1:11435}
per_set=()
if [ -n "${MEASURE_START:-}" ]; then
    start "$engine" "${MEASURE_READY_URL:?MEASURE_READY_URL with MEASURE_START}" 200 bash -c "$MEASURE_START"
else
    case $engine in
        ollaya | winnow)
            model=${OLLAYA_MODEL:-laya}
            [ "$engine" = winnow ] && model=${WINNOW_MODEL:-winnow:e4b}
            if [ "$(http_code "$ollaya_url/api/tags")" != 200 ]; then
                start ollaya "$ollaya_url/api/tags" 200 env OLLAYA_HOST="${ollaya_url#http://}" ollaya serve
            fi
            # shellcheck disable=SC2329  # called by stop_all
            unload() {
                if ! OLLAYA_HOST="${ollaya_url#http://}" ollaya stop "$model" > "$logs/unload.log" 2>&1; then
                    echo "could not unload $model from $ollaya_url; it may still hold memory:" >&2
                    cat "$logs/unload.log" >&2
                fi
            }
            ;;
        kev-0.8b | kev-4b | kev-9b)
            start "$engine" "${KEV_URL:-http://127.0.0.1:8009}/v1/models" 200 \
                env KEV_RUN="jaredpalmer/$engine" "$scripts/serve-kev.sh"
            ;;
        clef-flash)
            # /health answers 503 until the GGUF is loaded
            start "$engine" "${LLAMA_URL:-http://127.0.0.1:8020}/health" 200 "$scripts/serve-llama.sh" "$engine"
            ;;
        von)
            start von "${VON_URL:-http://127.0.0.1:8030}/health" 200 "$scripts/serve-von.sh"
            ;;
        jeff)
            start jeff "${JEFF_URL:-http://127.0.0.1:8765}/health" 200 "$scripts/serve-jeff.sh"
            ;;
        anyjev-raw | anyjev-l0)
            start llm "${ANYJEV_LLM_URL:-http://127.0.0.1:8092}/v1/models" 200 \
                "$scripts/serve-llm.sh" Qwen/Qwen3-8B 8092 qwen3-8b
            per_set=(anyjev "${ANYJEV_URL:-http://127.0.0.1:8710}/" any "$scripts/serve-anyjev.sh")
            ;;
        clm)
            start embed http://127.0.0.1:8090/v1/models 200 "$scripts/serve-embed.sh" Qwen/Qwen3-8B 8090 qwen3-8b
            start clm "${CLM_URL:-http://127.0.0.1:8700}/" any "$scripts/serve-clm.sh"
            ;;
    esac
fi

failed=()
for set in "$@"; do
    if [ ${#per_set[@]} -gt 0 ]; then
        start "${per_set[@]}"
    fi
    "${bench[@]}" --engine "$engine" --questions "$set" ${reps[@]+"${reps[@]}"} || failed+=("$set")
    if [ ${#per_set[@]} -gt 0 ]; then
        stop "${pids[${#pids[@]} - 1]}"
        unset 'pids[${#pids[@]}-1]'
    fi
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed sets: ${failed[*]}" >&2
    exit 1
fi
