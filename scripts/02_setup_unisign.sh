#!/bin/bash
# ============================================
# Step 2: Conda + Uni-Sign + Weights Setup
# Run on Azure VM AFTER reboot (01 완료 후)
# ============================================

set -e

echo "=========================================="
echo "  BridgeCast AI - Uni-Sign Setup"
echo "=========================================="

# Check NVIDIA driver
echo "[0/6] Checking NVIDIA driver..."
if ! nvidia-smi &>/dev/null; then
    echo "❌ NVIDIA driver not found! Run 01_install_nvidia.sh first."
    exit 1
fi
echo "✅ NVIDIA driver OK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install Miniconda (if not exists)
echo "[1/6] Installing Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    $HOME/miniconda3/bin/conda init bash
    echo "✅ Miniconda installed"
else
    echo "✅ Miniconda already installed"
fi

# Source conda
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

# Create conda environment
echo "[2/6] Creating conda environment (unisign, python 3.9)..."
if conda env list | grep -q "unisign"; then
    echo "✅ Environment 'unisign' already exists"
else
    conda create -n unisign python=3.9 -y
fi
conda activate unisign

# Install PyTorch + dependencies
echo "[3/6] Installing PyTorch (CUDA 12.1)..."
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

echo "[4/6] Installing Uni-Sign dependencies..."
pip install transformers deepspeed rtmlib onnxruntime-gpu huggingface_hub

# Clone Uni-Sign
echo "[5/6] Cloning Uni-Sign repo..."
UNISIGN_DIR="$HOME/Uni-Sign"
if [ ! -d "$UNISIGN_DIR" ]; then
    git clone https://github.com/ZechengLi19/Uni-Sign.git "$UNISIGN_DIR"
else
    echo "✅ Uni-Sign repo already cloned"
fi

# Download weights
echo "[6/6] Downloading Uni-Sign weights from HuggingFace..."
python -c "
from huggingface_hub import snapshot_download
import os

weights_dir = os.path.expanduser('~/Uni-Sign/weights')
if not os.path.exists(weights_dir) or len(os.listdir(weights_dir)) < 3:
    print('Downloading weights... (this may take a few minutes)')
    snapshot_download('ZechengLi19/Uni-Sign', local_dir=weights_dir)
    print('✅ Weights downloaded')
else:
    print('✅ Weights already downloaded')
"

# Install Uni-Sign requirements (if exists)
if [ -f "$UNISIGN_DIR/requirements.txt" ]; then
    pip install -r "$UNISIGN_DIR/requirements.txt" 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "  ✅ Setup Complete!"
echo "=========================================="
echo ""
echo "테스트 방법:"
echo "  conda activate unisign"
echo "  cd ~/Uni-Sign"
echo "  python ./demo/online_inference.py \\"
echo "    --online_video <ASL_VIDEO.mp4> \\"
echo "    --finetune ./weights/wlasl_pose_only_islr.pth"
echo ""
echo "API 서버 실행:"
echo "  cd ~/azure_som"
echo "  ./scripts/03_run_api_server.sh"
