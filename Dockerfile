ARG BASE_CONTAINER=ghcr.io/ucsd-ets/scipy-ml-notebook:2025.2-stable
FROM ${BASE_CONTAINER}

USER root

# Create user zixi with a home directory
RUN useradd -m -s /bin/bash zixi

RUN apt-get update && \
    apt-get install -y htop && \
    rm -rf /var/lib/apt/lists/*

# Make huggingface cache folder in /home/zixi
RUN mkdir -p /home/zixi/.cache/huggingface/transformers && \
    chown -R zixi /home/zixi/.cache/huggingface

RUN pip install --no-cache-dir \
    accelerate \
    transformers \
    torch

ENV HF_HOME=/home/zixi/.cache/huggingface/transformers

USER zixi

RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer;\
AutoTokenizer.from_pretrained('EleutherAI/pythia-1b');\
AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b');\
print('Pre-cached Pythia-1B in Docker image')"

COPY llm-serve.py /home/zixi/llm-serve.py
