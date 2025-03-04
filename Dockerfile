# -------------------------------------------------------------
#  Dockerfile Example for vLLM on DSMLP
# -------------------------------------------------------------
# Option 1: Start from a UCSD EdTech Services GPU image:
ARG BASE_CONTAINER=ghcr.io/ucsd-ets/datascience-notebook:2025.1-stable
FROM ${BASE_CONTAINER}

# Switch to root to install packages
USER root

# Install any system packages you might need
# (Below is just an example)
RUN apt-get update && \
    apt-get install -y htop && \
    rm -rf /var/lib/apt/lists/*

# Install vLLM and any additional Python libraries
# vLLM includes both CPU and GPU logic. Because scipy-ml
# already has PyTorch and other ML dependencies, you can
# typically just pip-install vLLM directly.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir vllm

# Optionally: If you want a specific Hugging Face transformer
# version, or any other libraries:
# RUN pip install --no-cache-dir transformers==4.x.x

# (Optional) Clean up to reduce image size
RUN conda clean -tipsy || true

# Switch back to the notebook user (UID=1000) to run code
USER $NB_UID

# By default, Jupyter is the entrypoint on datahub images.
# We can add an entrypoint to run vLLM if desired, but
# we typically just run the model from within the container.
