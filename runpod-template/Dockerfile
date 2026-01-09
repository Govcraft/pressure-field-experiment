# Custom RunPod Pod Template for Latin Square Experiments
#
# Automatically runs all experiments when the pod starts.
# Models are downloaded to /workspace/models on first run (persistent).
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
ENV MODELS_DIR=/workspace/models
ENV TRIALS=30
ENV PARALLEL=4
# Set FORCE_REDOWNLOAD=1 to clear and re-download all models
ENV FORCE_REDOWNLOAD=0

WORKDIR /app

# Install system dependencies
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM and hf_transfer for fast downloads
RUN pip install --no-cache-dir vllm hf_transfer

# Copy experiment files
COPY latin-experiment /app/latin-experiment
COPY run-experiments.sh /app/run-experiments.sh
COPY run.sh /app/run.sh

RUN chmod +x /app/latin-experiment /app/run-experiments.sh /app/run.sh

# Startup script downloads models (if needed) then runs vLLM and experiments
CMD ["/app/run.sh"]
