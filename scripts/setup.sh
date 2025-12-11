#!/bin/bash
# Setup script for Conversational LLM Training
# Run this first to set up your environment

set -e

echo "=============================================="
echo "Setting up Conversational LLM Environment"
echo "=============================================="

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo ""
    echo "[1/4] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "[3/4] Installing PyTorch with CUDA support..."
echo "Detecting CUDA version..."

# Try to detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "nvcc not found, defaulting to CUDA 12.1"
    CUDA_VERSION="12.1"
fi

# Install PyTorch based on CUDA version
if [[ "$CUDA_VERSION" == "11."* ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12."* ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install other dependencies
echo ""
echo "[4/4] Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Process data: python src/data_processing.py"
echo "  3. Start training: python src/train.py --config configs/config.yaml"
echo ""
