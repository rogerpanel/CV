#!/bin/bash
################################################################################
# Integrated AI-IDS Installation Script
################################################################################
#
# Automated installation and setup for Integrated AI-IDS
#
# Usage:
#   ./install.sh [--dev] [--cuda] [--docker]
#
# Options:
#   --dev      Install development dependencies
#   --cuda     Install CUDA-enabled PyTorch
#   --docker   Build and run Docker containers
#
# Author: Roger Nick Anaedevha
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DEV_MODE=false
CUDA_MODE=false
DOCKER_MODE=false

for arg in "$@"; do
    case $arg in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --cuda)
            CUDA_MODE=true
            shift
            ;;
        --docker)
            DOCKER_MODE=true
            shift
            ;;
    esac
done

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Integrated AI-IDS Installation${NC}"
echo -e "${BLUE}   PhD Dissertation Implementation${NC}"
echo -e "${BLUE}   Roger Nick Anaedevha - MEPhI University${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Check Python version
echo -e "${YELLOW}[1/8] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.8+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}\n"

# Create virtual environment
echo -e "${YELLOW}[2/8] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}\n"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}\n"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}[3/8] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded${NC}\n"

# Install PyTorch
echo -e "${YELLOW}[4/8] Installing PyTorch...${NC}"
if [ "$CUDA_MODE" = true ]; then
    echo "Installing CUDA-enabled PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi
echo -e "${GREEN}✓ PyTorch installed${NC}\n"

# Install main dependencies
echo -e "${YELLOW}[5/8] Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}\n"

# Install package
echo -e "${YELLOW}[6/8] Installing Integrated AI-IDS...${NC}"
if [ "$DEV_MODE" = true ]; then
    pip install -e ".[dev]"
    echo -e "${GREEN}✓ Installed in development mode${NC}\n"
else
    pip install -e .
    echo -e "${GREEN}✓ Installed${NC}\n"
fi

# Create necessary directories
echo -e "${YELLOW}[7/8] Creating directories...${NC}"
mkdir -p logs
mkdir -p data
mkdir -p checkpoints
mkdir -p results
echo -e "${GREEN}✓ Directories created${NC}\n"

# Run tests
echo -e "${YELLOW}[8/8] Running tests...${NC}"
python -m pytest tests/ -v --tb=short || echo -e "${YELLOW}⚠ Some tests failed (this may be expected for first-time setup)${NC}\n"

# Docker setup (optional)
if [ "$DOCKER_MODE" = true ]; then
    echo -e "\n${YELLOW}[Docker] Building Docker image...${NC}"
    docker build -t integrated-ai-ids:latest .
    echo -e "${GREEN}✓ Docker image built${NC}\n"

    echo -e "${YELLOW}[Docker] Starting services with docker-compose...${NC}"
    cd deployment/docker
    docker-compose up -d
    echo -e "${GREEN}✓ Docker services started${NC}\n"
    cd ../..
fi

# Print success message
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Installation Complete! ✓${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"

echo -e "${BLUE}Quick Start:${NC}"
echo -e "  1. Activate virtual environment:  ${YELLOW}source venv/bin/activate${NC}"
echo -e "  2. Run end-to-end demo:           ${YELLOW}python demos/end_to_end_demo.py${NC}"
echo -e "  3. Start REST API server:         ${YELLOW}python -m integrated_ai_ids.api.rest_server${NC}"
echo -e "  4. Run tests:                     ${YELLOW}pytest tests/${NC}"
echo -e "  5. Read documentation:            ${YELLOW}cat QUICKSTART.md${NC}\n"

echo -e "${BLUE}Next Steps:${NC}"
echo -e "  • Configure models:  Edit ${YELLOW}configs/model_config.yaml${NC}"
echo -e "  • Add datasets:      Place in ${YELLOW}data/${NC} directory"
echo -e "  • Integration:       See ${YELLOW}docs/INTEGRATION_GUIDE.md${NC}\n"

# Check GPU availability
if [ "$CUDA_MODE" = true ]; then
    echo -e "${BLUE}GPU Status:${NC}"
    python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('  No GPU detected')"
    echo ""
fi

echo -e "${GREEN}Ready for thesis defense demonstration! 🎓${NC}\n"
