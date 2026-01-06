#!/bin/bash

PID_FILE="/workspace/weaviate/weaviate.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping Weaviate (PID: $PID)..."
        kill $PID
        
        for i in {1..10}; do
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "Weaviate stopped successfully"
                rm "$PID_FILE"
                exit 0
            fi
            sleep 1
        done
        
        echo "Force killing Weaviate..."
        kill -9 $PID
        rm "$PID_FILE"
        echo "Weaviate force stopped"
    else
        echo "Weaviate is not running (stale PID file)"
        rm "$PID_FILE"
    fi
else
    echo "PID file not found. Trying to find process..."
    pkill -f "/workspace/weaviate/weaviate --host"
    echo "Weaviate stopped (if it was running)"
fi




