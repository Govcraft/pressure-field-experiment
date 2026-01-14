#!/bin/bash
#
# Startup script for Latin Square Experiment Pod
#
# Downloads models to /workspace/models on first run (persists across restarts).
# Starts 3 vLLM servers (3B, 7B, 14B) and runs experiments.
# Results are saved to /workspace/results (persistent network drive).

set -euo pipefail

echo "=== Latin Square Experiment Pod ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"

# Create directories in /workspace (persistent volume)
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR" /workspace/results

# Persist vLLM compilation cache across pod restarts
# vLLM 0.13+ compiles CUDA graphs on first start, which is slow
# By storing in /workspace, subsequent pod starts reuse the cache
mkdir -p /workspace/.cache
export XDG_CACHE_HOME=/workspace/.cache
echo "vLLM cache directory: $XDG_CACHE_HOME/vllm"

# Download models if not already present
# Verifies tokenizer files exist (not just config.json)
download_model() {
    local model_name=$1
    local model_dir="$MODELS_DIR/$model_name"

    # Check for complete download: config.json AND tokenizer files
    # Qwen2 models need either tokenizer.json OR (vocab.json + merges.txt)
    if [[ -d "$model_dir" ]] && [[ -f "$model_dir/config.json" ]]; then
        if [[ -f "$model_dir/tokenizer.json" ]] || \
           ( [[ -f "$model_dir/vocab.json" ]] && [[ -f "$model_dir/merges.txt" ]] ); then
            echo "  $model_name already downloaded (tokenizer verified)"
            return 0
        else
            echo "  $model_name incomplete (missing tokenizer files), re-downloading..."
            rm -rf "$model_dir"
        fi
    fi

    echo "  Downloading $model_name..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/$model_name',
    local_dir='$model_dir',
    local_dir_use_symlinks=False  # Ensure actual files, not symlinks
)
print('  Done: $model_name')
"

    # Verify download
    if [[ ! -f "$model_dir/tokenizer.json" ]] && [[ ! -f "$model_dir/vocab.json" ]]; then
        echo "ERROR: $model_name download failed - tokenizer files missing!"
        ls -la "$model_dir/" || true
        return 1
    fi
}

echo ""
echo "=== Checking Models ==="
echo "Models directory: $MODELS_DIR"

# Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Force re-download if requested (useful for fixing corrupted downloads)
if [[ "${FORCE_REDOWNLOAD:-}" == "1" ]]; then
    echo "FORCE_REDOWNLOAD=1 detected, clearing all models..."
    rm -rf "$MODELS_DIR"/*
fi

# Only download models we use (3B, 7B, 14B) - skip 0.5B and 1.5B for efficiency
download_model "Qwen2.5-3B" || exit 1
download_model "Qwen2.5-7B" || exit 1
download_model "Qwen2.5-14B" || exit 1

echo ""
echo "All models ready!"

# Diagnostic: List all model directories with tokenizer files
echo ""
echo "=== Model Directory Summary ==="
for model in Qwen2.5-3B Qwen2.5-7B Qwen2.5-14B; do
    echo "$model:"
    ls -la "$MODELS_DIR/$model/"*token* "$MODELS_DIR/$model/"*vocab* 2>/dev/null | head -5 || echo "  WARNING: No tokenizer files found!"
done

# Start vLLM servers for each model in the escalation chain
# Each model runs on a different port (8003-8005)
# The experiment uses HuggingFace model names which map to these ports

echo ""
echo "=== Starting vLLM Servers ==="
echo "Starting 3 vLLM instances for model escalation chain (3B, 7B, 14B)..."

# Model-to-GPU and port mapping:
# GPU 0: 3B (port 8003, 25%) + 7B (port 8004, 45%) = 70%
# GPU 1: 14B (port 8005, 85%) - dedicated GPU for largest model

start_vllm_server() {
    local model=$1
    local port=$2
    local mem_util=$3
    local gpu=$4
    local logfile="/workspace/vllm-${port}.log"

    echo "  Starting $model on port $port (GPU $gpu, ${mem_util} mem)..."
    echo "  Model path: $MODELS_DIR/$model"

    # List tokenizer files for debugging
    echo "  Tokenizer files:" >> "$logfile"
    ls -la "$MODELS_DIR/$model/"*token* "$MODELS_DIR/$model/"*vocab* 2>/dev/null >> "$logfile" || true

    # Extract HuggingFace-style model name (e.g., "Qwen/Qwen2.5-0.5B" from "Qwen2.5-0.5B")
    local hf_model_name="Qwen/$model"

    CUDA_VISIBLE_DEVICES=$gpu vllm serve "$MODELS_DIR/$model" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$mem_util" \
        --max-model-len 2048 \
        --port $port \
        --tokenizer "$MODELS_DIR/$model" \
        --served-model-name "$hf_model_name" \
        > "$logfile" 2>&1 &
    echo $!
}

# Start servers SEQUENTIALLY to avoid initialization conflicts
# vLLM has issues when multiple instances start simultaneously

wait_for_server() {
    local port=$1
    local name=$2
    local timeout=${3:-180}
    for i in $(seq 1 $timeout); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "  $name (port $port) ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: $name (port $port) failed to start within ${timeout}s"
    echo "=== vLLM log for port $port ==="
    tail -50 "/workspace/vllm-${port}.log"
    return 1
}

echo ""
echo "Starting vLLM servers sequentially (to avoid init conflicts)..."

# GPU 0: 3B and 7B (smaller models share one GPU)
start_vllm_server "Qwen2.5-3B" 8003 0.25 0
wait_for_server 8003 "Qwen2.5-3B" 120 || exit 1

start_vllm_server "Qwen2.5-7B" 8004 0.45 0
wait_for_server 8004 "Qwen2.5-7B" 300 || exit 1

# GPU 1: 14B (largest model gets dedicated GPU)
start_vllm_server "Qwen2.5-14B" 8005 0.85 1
wait_for_server 8005 "Qwen2.5-14B" 600 || exit 1

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

# vLLM hosts for model escalation chain (one per model, in order)
# 3B -> 7B -> 14B (skip 0.5B, 1.5B for efficiency)
export VLLM_HOST=http://localhost:8003  # Start with 3B model
export VLLM_HOSTS="http://localhost:8003,http://localhost:8004,http://localhost:8005"

export LATIN_EXPERIMENT=/app/latin-experiment
export OUTPUT_DIR=/workspace/results
export TRIALS=${TRIALS:-30}
export PARALLEL=${PARALLEL:-10}

echo "Configuration:"
echo "  TRIALS=$TRIALS"
echo "  PARALLEL=$PARALLEL"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  VLLM_HOSTS=$VLLM_HOSTS"
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
