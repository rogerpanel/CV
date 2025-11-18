# Integrated AI-IDS Docker Image
# Production-ready containerized deployment

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Labels
LABEL maintainer="Roger Nick Anaedevha <ar006@campus.mephi.ru>"
LABEL description="Integrated AI-IDS: Unified Network Intrusion Detection System"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Install integrated-ai-ids
RUN pip3 install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /data/logs /data/models /config /var/log/ai-ids

# Download pre-trained models (optional - can be mounted as volume)
# RUN python3 -m integrated_ai_ids.utils.download_models --output /data/models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set permissions
RUN useradd -m -u 1000 aiids && \
    chown -R aiids:aiids /app /data /config /var/log/ai-ids

USER aiids

# Run API server
CMD ["python3", "-m", "integrated_ai_ids.api.rest_server", "--host", "0.0.0.0", "--port", "8000"]
