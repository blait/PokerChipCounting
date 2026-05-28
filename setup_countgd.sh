#!/bin/bash
# CountGD Setup Script — Run on GPU server (separate from SAM3 instance)
#
# Prerequisites:
#   - Ubuntu 22.04+ with NVIDIA GPU (A10G/L40S/RTX 3090+)
#   - CUDA 11.8+ installed
#   - Python 3.9+
#
# Usage:
#   chmod +x setup_countgd.sh
#   ./setup_countgd.sh
#
# After setup, run the server:
#   source /opt/pytorch/bin/activate  # or your venv
#   cd /tmp && python3 countgd_server.py --countgd-repo /home/ubuntu/CountGD

set -e

echo "=== CountGD Setup ==="

INSTALL_DIR="/home/ubuntu/CountGD"
CHECKPOINT_DIR="${INSTALL_DIR}/checkpoints"

# Step 1: Clone CountGD repo
if [ -d "$INSTALL_DIR" ]; then
    echo "[1/6] CountGD repo already exists at $INSTALL_DIR, pulling latest..."
    cd "$INSTALL_DIR" && git pull
else
    echo "[1/6] Cloning CountGD..."
    cd /home/ubuntu
    git clone https://github.com/niki-amini-naieni/CountGD.git
fi

cd "$INSTALL_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Step 2: Install Python dependencies
echo "[2/6] Installing Python dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
pip install -q \
    transformers==4.39.1 \
    timm==0.9.16 \
    scipy \
    opencv-python \
    Pillow \
    pyyaml \
    supervision \
    groundingdino-py \
    omegaconf

# Also install FastAPI deps for server
pip install -q fastapi uvicorn python-multipart

# Step 3: Download BERT model
echo "[3/6] Downloading BERT base uncased..."
if [ -d "${CHECKPOINT_DIR}/bert-base-uncased" ]; then
    echo "  Already exists, skipping."
else
    python3 -c "
from transformers import BertModel, BertTokenizer
BertModel.from_pretrained('bert-base-uncased', cache_dir='${CHECKPOINT_DIR}/bert-base-uncased-cache')
BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='${CHECKPOINT_DIR}/bert-base-uncased-cache')
# Also save locally
import shutil, os
cache_dirs = [d for d in os.listdir('${CHECKPOINT_DIR}/bert-base-uncased-cache') if 'bert-base-uncased' in d.lower()]
print(f'Downloaded BERT to cache')
"
    # Alternative: use the download script if it exists
    if [ -f "download_bert.py" ]; then
        python3 download_bert.py || true
    fi
fi

# Step 4: Download GroundingDINO weights (Swin-B)
echo "[4/6] Downloading GroundingDINO Swin-B checkpoint..."
GDINO_CKPT="${CHECKPOINT_DIR}/groundingdino_swinb_cogcoor.pth"
if [ -f "$GDINO_CKPT" ]; then
    echo "  Already exists, skipping."
else
    wget -q --show-progress -O "$GDINO_CKPT" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
fi

# Step 5: Download CountGD checkpoint
echo "[5/6] Downloading CountGD checkpoint..."
COUNTGD_CKPT="${CHECKPOINT_DIR}/checkpoint_fsc147_best.pth"
if [ -f "$COUNTGD_CKPT" ]; then
    echo "  Already exists, skipping."
else
    echo "  NOTE: CountGD checkpoint must be downloaded manually."
    echo "  Visit: https://drive.google.com/drive/folders/1zXXhm3XH5S2VD2_LgQ8UJfPHnJlm0YhE"
    echo "  Download 'checkpoint_fsc147_best.pth' and place at:"
    echo "    $COUNTGD_CKPT"
    echo ""
    echo "  Or use gdown:"
    echo "    pip install gdown"
    echo "    gdown --fuzzy 'https://drive.google.com/file/d/<FILE_ID>/view' -O $COUNTGD_CKPT"
    echo ""
    # Try gdown if available
    if command -v gdown &> /dev/null; then
        echo "  Attempting gdown..."
        gdown --fuzzy "https://drive.google.com/drive/folders/1zXXhm3XH5S2VD2_LgQ8UJfPHnJlm0YhE" -O "$CHECKPOINT_DIR/" --folder || {
            echo "  gdown failed. Please download manually."
        }
    else
        pip install -q gdown
        gdown --fuzzy "https://drive.google.com/drive/folders/1zXXhm3XH5S2VD2_LgQ8UJfPHnJlm0YhE" -O "$CHECKPOINT_DIR/" --folder || {
            echo "  gdown failed. Please download manually."
        }
    fi
fi

# Step 6: Verify installation
echo "[6/6] Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, '${INSTALL_DIR}')
try:
    from models import build_model
    print('  CountGD models module: OK')
except ImportError as e:
    print(f'  WARNING: {e}')
    print('  You may need to install additional deps or check the repo structure.')

import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the CountGD server:"
echo "  cd /tmp"
echo "  cp /home/ubuntu/chipcounting/countgd_engine.py /tmp/"
echo "  cp /home/ubuntu/chipcounting/countgd_server.py /tmp/"
echo "  mkdir -p /tmp/countgd_static"
echo "  cp /home/ubuntu/chipcounting/static/countgd.html /tmp/countgd_static/"
echo "  python3 countgd_server.py \\"
echo "    --countgd-repo /home/ubuntu/CountGD \\"
echo "    --checkpoint /home/ubuntu/CountGD/checkpoints/checkpoint_fsc147_best.pth \\"
echo "    --port 8001"
echo ""
echo "Server will be available at http://<instance-ip>:8001"
