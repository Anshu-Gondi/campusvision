#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Starting REAL stress runner..."

# ---------------- DEFAULTS ----------------
ITERATIONS=30
THREADS=12
PARALLEL_RUNS=3
SEED=$(date +%s)

TARGET="--all"   # default
TEST_FILTER=""

# ---------------- ARG PARSING ----------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --package|-p)
            TARGET="-p $2"
            shift 2
            ;;
        --test)
            TEST_FILTER="$2"
            shift 2
            ;;
        --iters)
            ITERATIONS="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

echo "Seed: $SEED"
echo "Target: $TARGET"
echo "Test filter: ${TEST_FILTER:-<none>}"

export RUST_BACKTRACE=1
export TEST_SEED=$SEED

# ---------------- RUN ----------------
for ((i=1; i<=ITERATIONS; i++))
do
    echo "=============================="
    echo "🔥 Iteration $i / $ITERATIONS"
    echo "=============================="

    if (( RANDOM % 2 )); then
        export FACE_DB_ROLE="writer"
    else
        export FACE_DB_ROLE="reader"
    fi

    echo "Role: $FACE_DB_ROLE"

    PIDS=()

    for ((p=1; p<=PARALLEL_RUNS; p++))
    do
        (
            echo "→ Runner $p started"

            cargo nextest run $TARGET \
                ${TEST_FILTER:+$TEST_FILTER} \
                --test-threads=$THREADS \
                --retries 1 \
                --failure-output final

        ) &

        PIDS+=($!)
    done

    # Random kill
    if (( RANDOM % 4 == 0 )); then
        sleep 1
        KILL_INDEX=$((RANDOM % ${#PIDS[@]}))
        echo "💀 Killing runner ${PIDS[$KILL_INDEX]}"
        kill -9 "${PIDS[$KILL_INDEX]}" || true
    fi

    # Wait
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "⚠️ Runner $pid failed"
    done

    # Random cleanup
    if (( RANDOM % 3 == 0 )); then
        echo "🧹 Simulating partial disk cleanup"
        rm -f ./face_database/data.bin || true
    fi

    sleep $((RANDOM % 2))
done

echo "✅ Stress test completed"