#!/bin/bash

SCRIPT_DIR="/workspace/embedding"
PID_FILE="$SCRIPT_DIR/reranker.pid"
LOG_FILE="$SCRIPT_DIR/reranker.log"

# Stop if already running
if [ -f "$PID_FILE" ]; then
    echo "Reranker already running. Stopping first..."
    bash "$SCRIPT_DIR/stop-reranker.sh"
    sleep 2
fi

# Use venv
echo "Starting Python reranker server with vllm env..."
cd "$SCRIPT_DIR"
source /workspace/embedding/.venv/bin/activate

export HF_HOME=/workspace/huggingface_cache
export CUDA_VISIBLE_DEVICES=0

nohup python bge-reranker.py > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Reranker started with PID: $(cat $PID_FILE)"
echo "Log: $LOG_FILE"
