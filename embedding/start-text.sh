#!/bin/bash

cd /workspace/embedding

source /workspace/embedding/.venv/bin/activate

export HF_HOME=/workspace/huggingface_cache
export CUDA_VISIBLE_DEVICES=0

PID_FILE="text.pid"
LOG_FILE="text.log"

# Kill existing server
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing text embedding server (PID: $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
    rm "$PID_FILE"
fi

# Start server
echo "Starting text embedding server on port 8010..."
nohup python bge-m3.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 3

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Text embedding server started successfully on port 8010"
    echo "PID: $(cat "$PID_FILE")"
    tail -20 "$LOG_FILE"
else
    echo "Failed to start text embedding server"
    cat "$LOG_FILE"
    exit 1
fi
