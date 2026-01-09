# Custom RunPod Pod Template for Latin Square Experiments
#
# Automatically runs all experiments when the pod starts.
# Results are saved to /workspace/results (persistent network drive).
#
# Usage:
#   docker build --platform linux/amd64 -t govcraft/latin-experiment:latest .
#   docker push govcraft/latin-experiment:latest

# Use RunPod PyTorch base image with CUDA 12.8
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1
ENV VLLM_HOST=http://localhost:8001
ENV LATIN_EXPERIMENT=/app/latin-experiment
ENV OUTPUT_DIR=/workspace/results
ENV TRIALS=30
ENV PARALLEL=4

WORKDIR /app

# Install system dependencies
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM and hf_transfer for fast downloads
RUN pip install --no-cache-dir vllm hf_transfer

# Pre-download all models in escalation chain during build
# This avoids download time when pod starts
# Models: 0.5B, 1.5B, 3B, 7B, 14B
ENV HF_HOME=/app/models
RUN python -c "from huggingface_hub import snapshot_download; \
    print('Downloading Qwen2.5-0.5B...'); \
    snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='/app/models/Qwen2.5-0.5B'); \
    print('Downloading Qwen2.5-1.5B...'); \
    snapshot_download('Qwen/Qwen2.5-1.5B', local_dir='/app/models/Qwen2.5-1.5B'); \
    print('Downloading Qwen2.5-3B...'); \
    snapshot_download('Qwen/Qwen2.5-3B', local_dir='/app/models/Qwen2.5-3B'); \
    print('Downloading Qwen2.5-7B...'); \
    snapshot_download('Qwen/Qwen2.5-7B', local_dir='/app/models/Qwen2.5-7B'); \
    print('Downloading Qwen2.5-14B...'); \
    snapshot_download('Qwen/Qwen2.5-14B', local_dir='/app/models/Qwen2.5-14B'); \
    print('All models downloaded!')"

# Copy experiment files
COPY latin-experiment /app/latin-experiment
COPY run-experiments.sh /app/run-experiments.sh
COPY run.sh /app/run.sh

RUN chmod +x /app/latin-experiment /app/run-experiments.sh /app/run.sh

# Startup script runs vLLM then experiments
CMD ["/app/run.sh"]
