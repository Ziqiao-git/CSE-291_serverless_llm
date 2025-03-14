###############################################################################
# Dockerfile to pre-download "EleutherAI/pythia-1b" via Hugging Face, and serve with vLLM
###############################################################################
ARG BASE_CONTAINER=ghcr.io/ucsd-ets/scipy-ml-notebook:2025.2-stable
FROM ${BASE_CONTAINER}

USER root

# (A) Optional: Install system packages
RUN apt-get update && \
    apt-get install -y htop && \
    rm -rf /var/lib/apt/lists/*

# (B) Create a local directory for the Hugging Face cache
ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface
RUN mkdir -p /opt/huggingface && chmod 777 /opt/huggingface

# (C) Install necessary Python libraries, including vLLM and transformers
RUN pip install --no-cache-dir vllm transformers

# (D) Pre-download pythia-1b using standard huggingface/transformers calls
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer;\
AutoTokenizer.from_pretrained('EleutherAI/pythia-1b', cache_dir='/opt/huggingface');\
AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b', cache_dir='/opt/huggingface');\
print('Pre-downloaded pythia-1b into /opt/huggingface')\
"

# (E) Copy a script that launches vLLM’s OpenAI-compatible API server
COPY llm-serve.sh /app/llm-serve.sh
RUN chmod +x /app/llm-serve.sh

# (F) Default command: run your serve script
CMD ["/app/llm-serve.sh"]
