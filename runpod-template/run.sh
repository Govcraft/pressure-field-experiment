#!/bin/bash
#
# Startup script for Latin Square Experiment Pod
#
# Starts 5 vLLM servers (one per model) and runs experiments.
# Results are saved to /workspace/results (persistent network drive).

set -euo pipefail

echo "=== Latin Square Experiment Pod ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"

# Create results directory in /workspace (persistent volume)
mkdir -p /workspace/results

# Start vLLM servers for each model in the escalation chain
# Each model runs on a different port (8001-8005)
# The experiment uses HuggingFace model names which map to these ports

echo ""
echo "=== Starting vLLM Servers ==="
echo "Starting 5 vLLM instances for model escalation chain..."

# Model-to-port mapping:
# Port 8001: 0.5B, Port 8002: 1.5B, Port 8003: 3B, Port 8004: 7B, Port 8005: 14B

start_vllm_server() {
    local model=$1
    local port=$2
    local logfile="/workspace/vllm-${port}.log"

    echo "  Starting $model on port $port..."
    vllm serve /app/models/$model \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.18 \
        --max-model-len 2048 \
        --port $port \
        > "$logfile" 2>&1 &
    echo $!
}

# Start all servers (they share GPU memory, ~18% each = 90% total)
PID_05B=$(start_vllm_server "Qwen2.5-0.5B" 8001)
PID_15B=$(start_vllm_server "Qwen2.5-1.5B" 8002)
PID_3B=$(start_vllm_server "Qwen2.5-3B" 8003)
PID_7B=$(start_vllm_server "Qwen2.5-7B" 8004)
PID_14B=$(start_vllm_server "Qwen2.5-14B" 8005)

echo ""
echo "vLLM PIDs: 0.5B=$PID_05B, 1.5B=$PID_15B, 3B=$PID_3B, 7B=$PID_7B, 14B=$PID_14B"

# Wait for all servers to be ready
echo ""
echo "Waiting for all vLLM servers to be ready..."

wait_for_server() {
    local port=$1
    local name=$2
    for i in {1..180}; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "  $name (port $port) ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: $name (port $port) failed to start within 180s"
    echo "=== vLLM log for port $port ==="
    cat "/workspace/vllm-${port}.log"
    return 1
}

# Wait for all servers in parallel
wait_for_server 8001 "Qwen2.5-0.5B" &
WAIT_05B=$!
wait_for_server 8002 "Qwen2.5-1.5B" &
WAIT_15B=$!
wait_for_server 8003 "Qwen2.5-3B" &
WAIT_3B=$!
wait_for_server 8004 "Qwen2.5-7B" &
WAIT_7B=$!
wait_for_server 8005 "Qwen2.5-14B" &
WAIT_14B=$!

# Wait for all waits to complete
wait $WAIT_05B $WAIT_15B $WAIT_3B $WAIT_7B $WAIT_14B

echo ""
echo "All vLLM servers ready!"

# Show GPU memory usage
echo ""
echo "=== GPU Memory Usage ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || echo "nvidia-smi not available"

# Run experiments
echo ""
echo "=== Starting Experiments ==="
echo "Time: $(date)"
cd /app
export VLLM_HOST=http://localhost:8001  # Start with smallest model
export LATIN_EXPERIMENT=/app/latin-experiment
export OUTPUT_DIR=/workspace/results
export TRIALS=${TRIALS:-30}
export PARALLEL=${PARALLEL:-4}

echo "Configuration:"
echo "  TRIALS=$TRIALS"
echo "  PARALLEL=$PARALLEL"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo ""

./run-experiments.sh --parallel $PARALLEL --trials $TRIALS all

echo ""
echo "=== Experiments Complete ==="
echo "Time: $(date)"
echo "Results saved to: /workspace/results"
echo ""
ls -la /workspace/results/

# Exit - results persist in /workspace (network drive)
echo ""
echo "Pod can now be terminated. Results are saved to /workspace/results (persists after pod shutdown)."
exit 0
