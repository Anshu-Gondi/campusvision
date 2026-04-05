#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Starting BENCH stress runner..."

ITERATIONS=10
PACKAGE="intelligence_core"
BENCH="simd"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iters)
            ITERATIONS="$2"
            shift 2
            ;;
        --bench)
            BENCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

echo "Iterations: $ITERATIONS"
echo "Bench: $BENCH"

for ((i=1; i<=ITERATIONS; i++))
do
    echo "=============================="
    echo "🔥 Bench Iteration $i"
    echo "=============================="

    # randomize env
    if (( RANDOM % 2 )); then
        export FACE_DB_ROLE="writer"
    else
        export FACE_DB_ROLE="reader"
    fi

    echo "Role: $FACE_DB_ROLE"

    cargo bench -p $PACKAGE --bench $BENCH -- \
        --warm-up-time 1 \
        --measurement-time 2 \
        --sample-size 10 \
        --noplot || echo "⚠️ Bench failed"

    sleep $((RANDOM % 2))
done

echo "✅ Bench stress completed"