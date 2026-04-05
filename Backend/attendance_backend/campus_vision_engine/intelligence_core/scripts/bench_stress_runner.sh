#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Starting BENCH stress runner..."

ITERATIONS=5
PACKAGE="intelligence_core"

for ((i=1; i<=ITERATIONS; i++))
do
    echo "=============================="
    echo "🔥 Bench Iteration $i"
    echo "=============================="

    # random role (simulate real system)
    if (( RANDOM % 2 )); then
        export FACE_DB_ROLE="writer"
    else
        export FACE_DB_ROLE="reader"
    fi

    echo "Role: $FACE_DB_ROLE"

    # random bench
    if (( RANDOM % 2 )); then
        BENCH="simd"
    else
        BENCH="search"
    fi

    echo "Running bench: $BENCH"

    cargo bench -p $PACKAGE --bench $BENCH -- \
        --warm-up-time 1 \
        --measurement-time 3 \
        --sample-size 20 \
        --noplot

    sleep $((RANDOM % 2))
done

echo "✅ Bench stress completed"