#!/bin/bash
# Automated setup script for Linux/Mac (for reference)

echo "===================================="
echo "Logo Detection Setup"
echo "===================================="
echo ""

# Check Python
python3 --version || { echo "ERROR: Python 3.12+ required"; exit 1; }

echo "Step 1/5: Installing PyTorch with CUDA..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "Step 2/5: Installing core dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 3/5: Installing SAM 3..."
cd sam3
pip install -e .
cd ..

echo ""
echo "Step 4/5: Creating .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Please edit .env and add your Gemini API key"
fi

echo ""
echo "Step 5/5: HuggingFace authentication..."
echo ""
echo "IMPORTANT: You need to:"
echo "1. Request access to SAM 3: https://huggingface.co/facebook/sam3"
echo "2. Then run: huggingface-cli login"
echo ""

echo "===================================="
echo "Setup Complete!"
echo "===================================="
echo ""
echo "Next steps:"
echo "1. Request HuggingFace SAM 3 access"
echo "2. Login: huggingface-cli login"
echo "3. Run: python sam3_official_detector.py"
echo ""
