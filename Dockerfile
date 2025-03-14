ARG BASE_CONTAINER=ghcr.io/ucsd-ets/scipy-ml-notebook:2025.2-stable
FROM ${BASE_CONTAINER}

USER root

# (A) Optional system packages
RUN apt-get update && \
    apt-get install -y htop && \
    rm -rf /var/lib/apt/lists/*

# (B) Create a folder to store vLLM-cached models
RUN mkdir -p /opt/vllm_models && chmod 777 /opt/vllm_models

# (C) Install vLLM
RUN pip install --no-cache-dir vllm

# (D) Pre-download Pythia-1B using vLLM's "tools.download" script (not entrypoints.download!)
RUN python -m vllm.tools.download \
    --model EleutherAI/pythia-1b \
    --destination /opt/vllm_models \
    --trust-remote-code

# (E) Copy your startup script that launches vLLM
COPY llm-serve.sh /app/llm-serve.sh
RUN chmod +x /app/llm-serve.sh

# (F) Default command
CMD ["/app/llm-serve.sh"]
