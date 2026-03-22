#!/bin/bash
# E2E test for cross-node distributed BAGEL deployment.
#
# Starts 3 processes on a single machine:
#   1. Mooncake Master (KV cache store coordination)
#   2. Worker (Stage 1 - DiT Diffusion, GPU 1)
#   3. Master (Stage 0 - Thinker LLM + API server, GPU 0)
#
# Then sends a test request and checks the response.
#
# Usage: bash scripts/test_cross_node_e2e.sh

set -euo pipefail

STAGE_CONFIG="vllm_omni/model_executor/stage_configs/bagel_multiconnector_localhost.yaml"
MODEL="ByteDance-Seed/BAGEL-7B-MoT"
LOGDIR="/tmp/cross_node_e2e"
mkdir -p "$LOGDIR"

cleanup() {
    echo "[E2E] Cleaning up..."
    kill $MASTER_PID $WORKER_PID $MOONCAKE_PID 2>/dev/null || true
    wait $MASTER_PID $WORKER_PID $MOONCAKE_PID 2>/dev/null || true
    echo "[E2E] Done."
}
trap cleanup EXIT

# ============================================================
# Step 1: Start Mooncake Master
# ============================================================
echo "[E2E] Starting Mooncake Master..."
mooncake_master \
    --rpc_port=50051 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=8080 \
    > "$LOGDIR/mooncake.log" 2>&1 &
MOONCAKE_PID=$!
echo "[E2E] Mooncake Master PID: $MOONCAKE_PID"

# Wait for mooncake to be ready (gRPC port)
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8080/metadata >/dev/null 2>&1; then
        echo "[E2E] Mooncake Master ready after ${i}s"
        break
    fi
    if ! kill -0 $MOONCAKE_PID 2>/dev/null; then
        echo "[E2E] ERROR: Mooncake Master died. Log:"
        tail -20 "$LOGDIR/mooncake.log"
        exit 1
    fi
    sleep 1
done

# ============================================================
# Step 2: Start Worker (Stage 1 - DiT Diffusion) on GPU 1
# ============================================================
echo "[E2E] Starting Worker (Stage 1) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -m vllm_omni.entrypoints.cli.main serve \
    "$MODEL" --omni \
    --stage-configs-path "$STAGE_CONFIG" \
    --stage-id 1 --headless \
    -oma 0.0.0.0 -omp 8091 \
    > "$LOGDIR/worker.log" 2>&1 &
WORKER_PID=$!
echo "[E2E] Worker PID: $WORKER_PID"

# Wait for worker ZMQ to bind
for i in $(seq 1 120); do
    if grep -q "listening on tcp" "$LOGDIR/worker.log" 2>/dev/null; then
        echo "[E2E] Worker ready after ${i}s"
        break
    fi
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "[E2E] ERROR: Worker died. Log:"
        tail -30 "$LOGDIR/worker.log"
        exit 1
    fi
    sleep 1
done

if ! grep -q "listening on tcp" "$LOGDIR/worker.log" 2>/dev/null; then
    echo "[E2E] ERROR: Worker not ready after 120s. Log:"
    tail -30 "$LOGDIR/worker.log"
    exit 1
fi

# ============================================================
# Step 3: Start Master (Stage 0 - Thinker LLM) on GPU 0
# ============================================================
echo "[E2E] Starting Master (Stage 0) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
    "$MODEL" --omni \
    --port 8000 \
    --stage-configs-path "$STAGE_CONFIG" \
    --stage-id 0 \
    -oma 127.0.0.1 -omp 8091 \
    > "$LOGDIR/master.log" 2>&1 &
MASTER_PID=$!
echo "[E2E] Master PID: $MASTER_PID"

# Wait for API server to be ready
for i in $(seq 1 180); do
    if grep -q "Application startup complete\|Uvicorn running\|Started server process" "$LOGDIR/master.log" 2>/dev/null; then
        echo "[E2E] Master API ready after ${i}s"
        break
    fi
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        echo "[E2E] ERROR: Master died. Log:"
        tail -40 "$LOGDIR/master.log"
        exit 1
    fi
    sleep 1
done

if ! grep -q "Application startup complete\|Uvicorn running\|Started server process" "$LOGDIR/master.log" 2>/dev/null; then
    echo "[E2E] ERROR: Master API not ready after 180s. Log:"
    tail -40 "$LOGDIR/master.log"
    exit 1
fi

# ============================================================
# Step 4: Send test request
# ============================================================
echo "[E2E] Sending test request..."
RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 300 \
    http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Generate an image of a cute cat"}]
    }')

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "[E2E] HTTP Status: $HTTP_CODE"
echo "[E2E] Response body (first 500 chars):"
echo "$BODY" | head -c 500
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo "[E2E] SUCCESS: Got 200 response from distributed BAGEL pipeline!"
else
    echo "[E2E] FAILED: Expected 200, got $HTTP_CODE"
    echo "[E2E] Master log tail:"
    tail -20 "$LOGDIR/master.log"
    echo "[E2E] Worker log tail:"
    tail -20 "$LOGDIR/worker.log"
    exit 1
fi
