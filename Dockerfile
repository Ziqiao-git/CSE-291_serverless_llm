ARG BASE_CONTAINER=ghcr.io/ucsd-ets/scipy-ml-notebook:2025.2-stable
FROM ${BASE_CONTAINER}

USER root

# # RUN python -c "\
# # from transformers import AutoModelForCausalLM, AutoTokenizer;\
# # AutoTokenizer.from_pretrained('EleutherAI/pythia-1b');\
# # AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b');\
# # print('Pythia-1B downloaded into the container. Cache at:', '${TRANSFORMERS_CACHE}')"

# Install any system packages you might need
RUN apt-get update && \
    apt-get install -y htop && \
    rm -rf /var/lib/apt/lists/* 

# 3. Install python libraries, including accelerate, transformers, etc.
#    `--no-cache-dir` helps reduce final image size.
RUN mkdir -p /home/zixi/.cache/huggingface/transformers && \
    chown -R zixi /home/zixi/.cache/huggingface

RUN pip install --no-cache-dir \
    accelerate \
    transformers \
    torch


ENV TRANSFORMERS_CACHE=/home/zixi/.cache/huggingface/transformers

USER zixi

RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer;\
AutoTokenizer.from_pretrained('EleutherAI/pythia-1b');\
AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b');\
print('Pre-cached Pythia-1B in Docker image')"

COPY llm-serve.py /home/zixi/llm-serve.py

# # 3) install packages using notebook user
# RUN pip install --no-cache-dir vllm 


