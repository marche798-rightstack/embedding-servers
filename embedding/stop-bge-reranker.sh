#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/reranker.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "PID file not found. Reranker may not be running."
    exit 1
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "Stopping reranker (PID: $PID)..."
    kill "$PID"

    # Wait for graceful shutdown (max 10 seconds)
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "Reranker stopped successfully."
            rm "$PID_FILE"
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "Reranker did not stop gracefully. Force killing..."
    kill -9 "$PID"
    rm "$PID_FILE"
    echo "Reranker force stopped."
else
    echo "Process $PID not found. Removing stale PID file."
    rm "$PID_FILE"
fi

