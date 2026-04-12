# PQ-IDPS Docker Container
# Provides reproducible environment for all experiments with quantum simulation support

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH="/workspace/PQ-IDPS:${PYTHONPATH}"

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
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PennyLane with optimizations
RUN pip3 install pennylane[all]==0.32.0

# Copy PQ-IDPS code
COPY . /workspace/PQ-IDPS/

# Create directories for data, logs, checkpoints, results
RUN mkdir -p /workspace/PQ-IDPS/datasets \
             /workspace/PQ-IDPS/logs \
             /workspace/PQ-IDPS/checkpoints \
             /workspace/PQ-IDPS/results \
             /workspace/PQ-IDPS/runs

# Set working directory to PQ-IDPS
WORKDIR /workspace/PQ-IDPS

# Expose ports for TensorBoard
EXPOSE 6006

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python3 -c "import pennylane as qml; print(f'PennyLane: {qml.__version__}')"

# Default command
CMD ["/bin/bash"]

# Usage:
# Build: docker build -t pq-idps:latest .
# Run:   docker run --gpus all -it -v $(pwd):/workspace/PQ-IDPS pq-idps:latest
# With TensorBoard: docker run --gpus all -it -p 6006:6006 -v $(pwd):/workspace/PQ-IDPS pq-idps:latest
#
# Inside container:
# - Train: python training/train_pq_idps.py --config config/pq_idps_config.yaml
# - TensorBoard: tensorboard --logdir runs/ --host 0.0.0.0 --port 6006
# - Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
