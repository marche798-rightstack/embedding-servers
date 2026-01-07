#!/bin/bash

cd /workspace/embedding

if [ ! -f embedding.pid ]; then
    echo "No PID file found. Server may not be running."
    exit 0
fi

PID=$(cat embedding.pid)

if ! kill -0 $PID 2>/dev/null; then
    echo "Process $PID is not running."
    rm embedding.pid
    exit 0
fi

echo "Stopping embedding server (PID: $PID)..."
kill $PID

# Wait up to 10 seconds for graceful shutdown
for i in {1..10}; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "Embedding server stopped successfully."
        rm embedding.pid
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "Force killing embedding server..."
kill -9 $PID 2>/dev/null
rm embedding.pid
echo "Embedding server force stopped."
