#!/bin/bash
cd /workspace/embedding

if [ ! -f image2.pid ]; then
    echo "No PID file found. Server may not be running."
    exit 0
fi

PID=$(cat image2.pid)

if ! kill -0 $PID 2>/dev/null; then
    echo "Process $PID is not running."
    rm image2.pid
    exit 0
fi

echo "Stopping DINOv2 server (PID: $PID)..."
kill $PID

# Wait up to 10 seconds for graceful shutdown
for i in {1..10}; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "DINOv2 server stopped successfully."
        rm image2.pid
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "Force killing DINOv2 server..."
kill -9 $PID 2>/dev/null
rm image2.pid
echo "DINOv2 server force stopped."
