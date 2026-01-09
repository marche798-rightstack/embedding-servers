#!/bin/bash

cd /workspace/embedding

PID_FILE="image.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Server may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID is not running."
    rm "$PID_FILE"
    exit 0
fi

echo "Stopping image embedding server (PID: $PID)..."
kill "$PID"

# Wait up to 10 seconds for graceful shutdown
for i in {1..10}; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Image embedding server stopped successfully."
        rm "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "Force killing image embedding server..."
kill -9 "$PID" 2>/dev/null
rm "$PID_FILE"
echo "Image embedding server force stopped."
