###############################################################################
# Dockerfile: Pre-cache pythia-1b in /opt/huggingface, and partial-load at runtime
###############################################################################
ARG BASE_CONTAINER=ghcr.io/ucsd-ets/scipy-ml-notebook:2025.2-stable
FROM ${BASE_CONTAINER}

USER root

# (A) Install system packages if desired
RUN apt-get update && \
    apt-get install -y htop && \
    rm -rf /var/lib/apt/lists/*

# (B) Create a folder for Hugging Face cache, outside /home to avoid DSMLP overshadowing
RUN mkdir -p /opt/huggingface && chmod 777 /opt/huggingface
ENV HF_HOME=/opt/huggingface

# (C) Install necessary Python libraries
RUN pip install --no-cache-dir \
    accelerate \
    transformers \
    torch

# (D) Pre-download the pythia-1b model into /opt/huggingface
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('EleutherAI/pythia-1b'); \
AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b'); \
print('Pre-cached Pythia-1B in Docker image at /opt/huggingface')"

# (E) Copy your partial-load serving script
COPY llm-serve.py /app/llm-serve.py

# Default command: run your script
CMD [\"python\", \"/app/llm-serve.py\"]
