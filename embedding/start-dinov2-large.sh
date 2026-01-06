#!/bin/bash
cd /workspace/embedding
source /workspace/embedding/.venv/bin/activate
export HF_HOME=/workspace/huggingface_cache
export CUDA_VISIBLE_DEVICES=0

# Kill existing server
if [ -f image2.pid ]; then
    OLD_PID=$(cat image2.pid)
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "Stopping existing DINOv2 server (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
    rm image2.pid
fi

# Start server
echo "Starting DINOv2 Large embedding server..."
nohup python dinov2-large.py > image2.log 2>&1 &
echo $! > image2.pid

sleep 3

if kill -0 $(cat image2.pid) 2>/dev/null; then
    echo "DINOv2 server started successfully on port 8012"
    echo "PID: $(cat image2.pid)"
    tail -20 image2.log
else
    echo "Failed to start DINOv2 server"
    cat image2.log
    exit 1
fi


