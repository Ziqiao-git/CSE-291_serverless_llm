#!/usr/bin/env bash
echo "Starting vLLM server with partial GPU usage..."

# If you want partial GPU usage, you can adjust:
#   --gpu-memory-utilization=0.5  (meaning vLLM tries to use ~50% of GPU memory)
#   or other advanced flags like --swap-space=4

python -m vllm.entrypoints.openai.api_server \
    --model /opt/vllm_models/EleutherAI__pythia-1b \
    --device cuda \
    --gpu-memory-utilization=0.7 \
    --port 8000
