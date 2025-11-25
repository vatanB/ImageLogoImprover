# WSL2 + CUDA Setup for SAM 3 Logo Detection

## Complete WSL2 Setup Guide (Uses Your RTX 3090)

### Step 1: Install WSL2 with Ubuntu

Open PowerShell as Administrator and run:

```powershell
# Install WSL2
wsl --install

# Reboot when prompted
```

After reboot, Ubuntu will open automatically. Set up your username/password.

### Step 2: Install NVIDIA CUDA in WSL2

**In PowerShell (Windows side):**

```powershell
# Download NVIDIA CUDA Toolkit for WSL
# Visit: https://developer.nvidia.com/cuda-downloads
# Select: Linux > x86_64 > WSL-Ubuntu > 2.0 > deb (network)

# Or download directly:
curl -o cuda-wsl.exe https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
```

**In WSL Ubuntu terminal:**

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get -y install cuda-toolkit-12-6

# Verify CUDA works
nvidia-smi  # Should show your RTX 3090!
```

### Step 3: Install Python 3.12 in WSL

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
sudo apt install -y python3-pip git

# Set Python 3.12 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
```

### Step 4: Clone and Setup Project

```bash
# Clone repository
cd ~
git clone https://github.com/vatanB/ImageLogoImprover.git
cd ImageLogoImprover

# Run automated setup
chmod +x setup_wsl.sh
./setup_wsl.sh
```

### Step 5: Configure API Keys

```bash
# Edit .env file
nano .env
# Add: GOOGLE_GEMINI_API_KEY=your_key_here
# Save: Ctrl+X, Y, Enter

# Login to HuggingFace
huggingface-cli login
# Paste your token
```

### Step 6: Run Logo Detection!

```bash
python3 sam3_official_detector.py
```

## Accessing Windows Files from WSL

```bash
# Your Windows C: drive is at:
cd /mnt/c/

# Example: Access Desktop
cd /mnt/c/Users/vatan/Desktop/

# Copy results to Windows Desktop:
cp output/sam3_official_detection.jpg /mnt/c/Users/vatan/Desktop/
```

## Troubleshooting

### CUDA Not Working
```bash
# Check NVIDIA driver
nvidia-smi

# Should show: RTX 3090 with CUDA 12.6
```

### Python Issues
```bash
python3 --version  # Should show 3.12.x
```

### Out of Memory
```bash
# Free up VRAM
pkill -f python
nvidia-smi  # Check VRAM usage
```

## Performance

With RTX 3090 in WSL2:
- **Detection**: ~1-2 seconds per image
- **Enhancement**: ~2-3 seconds per logo
- **Total**: ~10 seconds for 4 logos

## Why WSL2?

- ✅ Uses your RTX 3090 (full CUDA support)
- ✅ SAM 3 works perfectly (Triton supported)
- ✅ Linux environment (all tools compatible)
- ✅ Access Windows files easily
- ✅ Native GPU performance
