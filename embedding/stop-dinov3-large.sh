#!/bin/bash
cd /workspace/embedding

if [ ! -f image.pid ]; then
    echo "No PID file found. Server may not be running."
    exit 0
fi

PID=$(cat image.pid)

if ! kill -0 $PID 2>/dev/null; then
    echo "Process $PID is not running."
    rm image.pid
    exit 0
fi

echo "Stopping DINOv3 server (PID: $PID)..."
kill $PID

# Wait up to 10 seconds for graceful shutdown
for i in {1..10}; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "DINOv3 server stopped successfully."
        rm image.pid
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "Force killing DINOv3 server..."
kill -9 $PID 2>/dev/null
rm image.pid
echo "DINOv3 server force stopped."
