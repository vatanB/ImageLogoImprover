# Logo Detection with SAM 3 + Gemini Enhancement

Automated logo detection and enhancement using SAM 3 (text prompting) and Gemini 3 Pro Image.

## ⚠️ IMPORTANT: Windows Users - Use WSL2

**SAM 3 requires Linux.** Windows users must use WSL2 to access their NVIDIA GPU.

👉 **[Complete WSL2 Setup Guide](WSL_SETUP.md)** 👈

## Quick Start (WSL2 - Recommended)

```bash
# 1. Install WSL2 (PowerShell as Admin)
wsl --install

# 2. After reboot, in Ubuntu terminal:
git clone https://github.com/vatanB/ImageLogoImprover.git
cd ImageLogoImprover

# 3. Run automated setup (installs CUDA, SAM 3, everything)
./setup_wsl.sh

# 4. Configure
nano .env  # Add Gemini API key
huggingface-cli login  # Add HF token

# 5. Run!
python3 sam3_official_detector.py
```

## What This Does

1. **Detects** all logos using SAM 3 with text prompt "logo"
2. **Crops** each detected logo region
3. **Enhances** with Gemini 3 Pro Image for clarity
4. **Pastes** back enhanced logos into original image

## System Requirements

- **Windows**: WSL2 + NVIDIA GPU (RTX 3090 confirmed working)
- **Linux**: Native with NVIDIA GPU + CUDA 12.6
- **Mac**: Not supported (Triton unavailable)

## Performance (RTX 3090)

- Detection: ~1-2 seconds
- Enhancement per logo: ~2-3 seconds  
- Total for 4 logos: ~10 seconds

## Project Structure

```
ImageLogoImprover/
├── WSL_SETUP.md              # Complete WSL2 guide
├── setup_wsl.sh              # Automated WSL setup
├── sam3_official_detector.py # SAM 3 wrapper
├── logo_restoration_pipeline/ # Full pipeline
├── input/                    # Place images here
├── output/                   # Results saved here
└── assets/                   # Reference logos
```

## Troubleshooting

See [WSL_SETUP.md](WSL_SETUP.md#troubleshooting)

## Why WSL2?

SAM 3 requires Triton (NVIDIA library) which:
- ✅ Works on Linux
- ✅ Works in WSL2 with GPU passthrough
- ❌ Not available on Windows native
- ❌ Not available on Mac

WSL2 gives you full Linux environment with direct access to your RTX 3090!
