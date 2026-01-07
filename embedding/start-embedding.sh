#!/bin/bash

cd /workspace/embedding

source /workspace/embedding/.venv/bin/activate

export HF_HOME=/workspace/huggingface_cache
export CUDA_VISIBLE_DEVICES=0

# Kill existing server
if [ -f embedding.pid ]; then
    OLD_PID=$(cat embedding.pid)
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "Stopping existing embedding server (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
    rm embedding.pid
fi

# Start server
echo "Starting embedding server..."
nohup python embedding.py > embedding.log 2>&1 &
echo $! > embedding.pid

sleep 3

if kill -0 $(cat embedding.pid) 2>/dev/null; then
    echo "Embedding server started successfully on port 8010"
    echo "PID: $(cat embedding.pid)"
    tail -20 embedding.log
else
    echo "Failed to start embedding server"
    cat embedding.log
    exit 1
fi
