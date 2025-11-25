#!/bin/bash
# Automated setup for WSL2 Ubuntu with CUDA

echo "===================================="
echo "SAM 3 Logo Detection - WSL2 Setup"
echo "===================================="
echo ""

# Check if running in WSL
if ! grep -q Microsoft /proc/version; then
    echo "ERROR: This script must run in WSL2!"
    exit 1
fi

# Check CUDA
echo "Checking NVIDIA GPU..."
nvidia-smi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: NVIDIA GPU not detected"
    echo "Follow WSL_SETUP.md to install CUDA drivers"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ NVIDIA GPU detected"
    nvidia-smi
fi

echo ""
echo "Step 1/5: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-venv python3.12-dev python3-pip build-essential

echo ""
echo "Step 2/5: Installing PyTorch with CUDA..."
pip3 install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "Step 3/5: Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "Step 4/5: Cloning and installing SAM 3..."
if [ ! -d "sam3" ]; then
    git clone https://github.com/facebookresearch/sam3.git
fi
cd sam3
pip3 install -e .
cd ..

echo ""
echo "Step 5/5: Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Please edit .env and add your Gemini API key"
fi

echo ""
echo "===================================="
echo "Setup Complete!"
echo "===================================="
echo ""
echo "Next steps:"
echo "1. Edit .env: nano .env"
echo "2. Add Gemini API key"
echo "3. Login to HuggingFace: huggingface-cli login"
echo "4. Run: python3 sam3_official_detector.py"
echo ""
echo "Your RTX 3090 is ready for SAM 3!"
echo ""
