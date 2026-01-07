#!/bin/bash
cd /workspace/weaviate

# ============================================
# Enter your RunPod public IP here
# ============================================
PUBLIC_IP="213.173.110.201"

# Create data directories
mkdir -p /workspace/weaviate/weaviate_data
mkdir -p /workspace/weaviate/certs

# Generate gRPC TLS certificate (first time only)
if [ ! -f /workspace/weaviate/certs/server.crt ]; then
    echo "Generating gRPC TLS certificate for IP: $PUBLIC_IP"
    
    # Create self-signed certificate
    openssl req -x509 -newkey rsa:4096 -nodes \
      -keyout /workspace/weaviate/certs/server.key \
      -out /workspace/weaviate/certs/server.crt \
      -days 365 \
      -subj "/CN=weaviate-grpc" \
      -addext "subjectAltName=IP:$PUBLIC_IP,IP:127.0.0.1,DNS:localhost"
    
    echo "Certificate generated successfully"
fi

# Weaviate environment variables
export QUERY_DEFAULTS_LIMIT=25
export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
export DEFAULT_VECTORIZER_MODULE=none
export AUTHENTICATION_APIKEY_ENABLED=false
export AUTHENTICATION_APIKEY_ALLOWED_KEYS=1111
export AUTHENTICATION_APIKEY_USERS=rightstack
export ENABLE_PERSISTENCE=true
export PERSISTENCE_DATA_PATH=/workspace/weaviate/weaviate_data
export CLUSTER_HOSTNAME=node1
export ENABLE_TOKENIZER_KAGOME_KR=true

# gRPC TLS configuration
export GRPC_PORT=50051
export GRPC_CERT_FILE=/workspace/weaviate/certs/server.crt
export GRPC_KEY_FILE=/workspace/weaviate/certs/server.key

# Check for existing process
if [ -f /workspace/weaviate/weaviate.pid ]; then
    OLD_PID=$(cat /workspace/weaviate/weaviate.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Weaviate is already running (PID: $OLD_PID)"
        exit 1
    else
        rm /workspace/weaviate/weaviate.pid
    fi
fi

# Start Weaviate
nohup ./weaviate \
  --host 0.0.0.0 \
  --port 8020 \
  --scheme http \
  > /workspace/weaviate/weaviate.log 2>&1 &

echo $! > /workspace/weaviate/weaviate.pid
echo "Weaviate started with gRPC TLS. PID: $!"
echo "gRPC listening on port 50051 with TLS (IP: $PUBLIC_IP)"
echo "Check log: tail -f /workspace/weaviate/weaviate.log"

