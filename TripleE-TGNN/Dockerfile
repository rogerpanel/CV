# TripleE-TGNN Docker Container
# Provides reproducible environment with PyTorch Geometric for temporal graph learning

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH="/workspace/TripleE-TGNN:${PYTHONPATH}"

# Set working directory
WORKDIR /workspace

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
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==0.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and extensions
RUN pip3 install torch-geometric==2.3.1
RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Copy TripleE-TGNN code
COPY . /workspace/TripleE-TGNN/

# Create directories
RUN mkdir -p /workspace/TripleE-TGNN/datasets \
             /workspace/TripleE-TGNN/logs \
             /workspace/TripleE-TGNN/checkpoints \
             /workspace/TripleE-TGNN/results \
             /workspace/TripleE-TGNN/runs

# Set working directory to TripleE-TGNN
WORKDIR /workspace/TripleE-TGNN

# Expose ports for TensorBoard
EXPOSE 6006

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python3 -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

# Default command
CMD ["/bin/bash"]

# Usage:
# Build: docker build -t triplee-tgnn:latest .
# Run:   docker run --gpus all -it -v $(pwd):/workspace/TripleE-TGNN triplee-tgnn:latest
# With TensorBoard: docker run --gpus all -it -p 6006:6006 -v $(pwd):/workspace/TripleE-TGNN triplee-tgnn:latest
#
# Inside container:
# - Train: python training/train_triplee.py --config config/triplee_config.yaml
# - TensorBoard: tensorboard --logdir runs/ --host 0.0.0.0 --port 6006
