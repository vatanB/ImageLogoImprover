# Logo Detection and Enhancement with SAM 3

Automated logo detection and enhancement system using SAM 3 (text prompting) and Gemini 3 Pro Image.

## System Requirements

### Windows PC with NVIDIA GPU (Recommended)
- **GPU**: NVIDIA GPU with CUDA 12.6+ (RTX 3090 confirmed working)
- **Python**: 3.12+
- **CUDA**: 12.6 or higher
- **RAM**: 16GB+
- **VRAM**: 12GB+ (RTX 3090 has 24GB - perfect!)

## Quick Start (Windows + NVIDIA)

### 1. Install Prerequisites

```powershell
# Install Python 3.12
# Download from: https://www.python.org/downloads/

# Verify installation
python --version  # Should show 3.12.x

# Install CUDA 12.6
# Download from: https://developer.nvidia.com/cuda-downloads
```

### 2. Clone Repository

```powershell
git clone https://github.com/YOUR_USERNAME/ImageLogoImprover.git
cd ImageLogoImprover
```

### 3. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install PyTorch with CUDA

```powershell
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 5. Install SAM 3

```powershell
cd sam3
pip install -e .
cd ..
```

### 6. Install Other Dependencies

```powershell
pip install -r requirements.txt
```

### 7. Set Up API Keys

Create a `.env` file:
```env
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here
```

Get Gemini API key: https://aistudio.google.com/apikey

### 8. Authenticate with HuggingFace

```powershell
# Request access to SAM 3 checkpoint
# Visit: https://huggingface.co/facebook/sam3
# Click "Request Access"

# Once approved, login:
huggingface-cli login
# Paste your HF token from: https://huggingface.co/settings/tokens
```

### 9. Run Logo Detection

```powershell
python sam3_logo_detector.py
```

## Usage

### Detect Logos with SAM 3

```python
python sam3_logo_detector.py --image path/to/drift_kart.jpg --prompt "logo"
```

### Full Pipeline (Detect + Enhance)

```python
python logo_restoration_pipeline/main.py
```

This will:
1. **Detect** all logos using SAM 3 text prompting
2. **Crop** each logo region
3. **Enhance** with Gemini 3 Pro Image
4. **Paste** back enhanced logos
5. **Save** final result

## Project Structure

```
ImageLogoImprover/
├── sam3/                          # SAM 3 implementation
├── logo_restoration_pipeline/     # Main pipeline
│   ├── main.py                   # Orchestrates everything
│   ├── detector.py               # Logo detection (will use SAM 3)
│   └── generator.py              # Gemini enhancement
├── sam3_logo_detector.py         # SAM 3 wrapper
├── input/                        # Place images here
├── output/                       # Results saved here
├── assets/                       # Reference logos
│   └── bmw_logo.webp
├── requirements.txt
├── .env.example
└── README.md
```

## How It Works

### 1. SAM 3 Text-Prompted Detection
```python
# Simply prompt: "logo" or "BMW logo"
results = sam3.detect(image, prompt="logo")
# Returns: precise pixel-level masks for ALL logos
```

### 2. Gemini 3 Pro Image Enhancement
```python
# For each detected logo:
enhanced = gemini.enhance(cropped_logo, reference_logo)
# Returns: high-quality, sharp logo
```

### 3. Seamless Integration
```python
# Paste enhanced logos back
final_image = paste_enhanced(original, enhanced_logos)
```

## Why Windows + NVIDIA?

- **SAM 3**: Requires Triton (NVIDIA-only)
- **Speed**: GPU acceleration ~10x faster
- **Quality**: Better results with CUDA
- **Mac Support**: Limited (SAM 3 won't work)

## Troubleshooting

### CUDA Not Found
```powershell
nvidia-smi  # Check GPU is detected
nvcc --version  # Check CUDA installation
```

### SAM 3 Checkpoint Access Denied
1. Request access: https://huggingface.co/facebook/sam3
2. Wait for approval (usually instant)
3. Login: `huggingface-cli login`

### Out of Memory
- Reduce batch size
- Use smaller SAM 3 model (SAM 3-S instead of SAM 3-L)
- Close other GPU applications

## Expected Performance (RTX 3090)

- **Detection**: ~2 seconds per image
- **Enhancement**: ~3 seconds per logo  
- **Full Pipeline**: ~10 seconds for image with 4 logos

## License

SAM 3: Meta's SAM License
Gemini: Google's Terms of Service
