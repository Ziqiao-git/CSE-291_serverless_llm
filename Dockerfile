###############################################################################
# Dockerfile: Pre-cache pythia-1b for vLLM and serve at runtime
###############################################################################
ARG BASE_CONTAINER=ghcr.io/ucsd-ets/scipy-ml-notebook:2025.2-stable
FROM ${BASE_CONTAINER}

USER root

# (A) Optional: install system packages
RUN apt-get update && \
    apt-get install -y htop && \
    rm -rf /var/lib/apt/lists/*

# (B) Create a folder to store the vLLM model cache
RUN mkdir -p /opt/vllm_models && chmod 777 /opt/vllm_models

# (C) Install vLLM
RUN pip install --no-cache-dir vllm

# (D) Pre-download the Pythia-1B model using vLLM's built-in download command
#     This fetches all necessary model files into /opt/vllm_models/EleutherAI__pythia-1b
RUN python -m vllm.entrypoints.download \
    --model EleutherAI/pythia-1b \
    --destination /opt/vllm_models \
    --trust-remote-code

# (E) Copy a simple startup script that runs the vLLM API server
#     (You can replace llm-serve.sh with your own script if desired)
COPY llm-serve.sh /app/llm-serve.sh
RUN chmod +x /app/llm-serve.sh

# (F) Default command: run your vLLM serve script
CMD ["/app/llm-serve.sh"]
