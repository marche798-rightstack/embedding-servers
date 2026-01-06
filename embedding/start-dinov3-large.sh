#!/bin/bash
cd /workspace/embedding
source /workspace/embedding/.venv/bin/activate
export HF_HOME=/workspace/huggingface_cache
#export HF_TOKEN=""
export CUDA_VISIBLE_DEVICES=0

# Kill existing server
if [ -f image.pid ]; then
    OLD_PID=$(cat image.pid)
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "Stopping existing DINOv3 server (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
    rm image.pid
fi

# Start server
echo "Starting DINOv3 Large embedding server..."
nohup python dinov3-large.py > image.log 2>&1 &
echo $! > image.pid

sleep 3

if kill -0 $(cat image.pid) 2>/dev/null; then
    echo "DINOv3 server started successfully on port 8012"
    echo "PID: $(cat image.pid)"
    tail -20 image.log
else
    echo "Failed to start DINOv3 server"
    cat image.log
    exit 1
fi


