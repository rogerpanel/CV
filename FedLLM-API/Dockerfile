# FedLLM-API Docker Container
# Provides reproducible environment for all experiments

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy FedLLM-API code
COPY . /workspace/FedLLM-API/

# Set Python path
ENV PYTHONPATH="/workspace/FedLLM-API:${PYTHONPATH}"

# Create directories for data, logs, checkpoints
RUN mkdir -p /workspace/FedLLM-API/datasets \
             /workspace/FedLLM-API/logs \
             /workspace/FedLLM-API/checkpoints \
             /workspace/FedLLM-API/results

# Set working directory to FedLLM-API
WORKDIR /workspace/FedLLM-API

# Expose ports for TensorBoard and Flower server
EXPOSE 6006 8080

# Default command
CMD ["/bin/bash"]

# Usage:
# Build: docker build -t fedllm-api:latest .
# Run:   docker run --gpus all -it -v $(pwd):/workspace fedllm-api:latest
# With TensorBoard: docker run --gpus all -it -p 6006:6006 -v $(pwd):/workspace fedllm-api:latest
