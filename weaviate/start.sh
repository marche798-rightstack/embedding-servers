#!/bin/bash

cd /workspace/weaviate

mkdir -p /workspace/weaviate/weaviate_data

export QUERY_DEFAULTS_LIMIT=25
export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
export DEFAULT_VECTORIZER_MODULE=none
export AUTHENTICATION_APIKEY_ENABLED=false
export AUTHENTICATION_APIKEY_ALLOWED_KEYS=1111
export AUTHENTICATION_APIKEY_USERS=rightstack
export ENABLE_PERSISTENCE=true
export PERSISTENCE_DATA_PATH=/workspace/weaviate/weaviate_data
export CLUSTER_HOSTNAME=node1
export GRPC_PORT=50051
export ENABLE_TOKENIZER_KAGOME_KR=true

if [ -f /workspace/weaviate/weaviate.pid ]; then
    OLD_PID=$(cat /workspace/weaviate/weaviate.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Weaviate is already running (PID: $OLD_PID)"
        exit 1
    else
        rm /workspace/weaviate/weaviate.pid
    fi
fi

nohup ./weaviate \
  --host 0.0.0.0 \
  --port 8020 \
  --scheme http \
  > /workspace/weaviate/weaviate.log 2>&1 &

echo $! > /workspace/weaviate/weaviate.pid

echo "Weaviate started in background. PID: $!"
echo "Check log: tail -f /workspace/weaviate/weaviate.log"


