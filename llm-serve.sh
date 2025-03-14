#!/usr/bin/env bash

echo "Launching vLLM server for pythia-1b from local Hugging Face cache..."

# The --model argument is "EleutherAI/pythia-1b", but the actual weights
# will load from /opt/huggingface because it's cached there.
python -m vllm.entrypoints.openai.api_server \
    --model EleutherAI/pythia-1b \
    --device cuda \
    --port 8000
