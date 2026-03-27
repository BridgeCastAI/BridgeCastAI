#!/bin/bash
# ============================================
# Step 1: NVIDIA Driver Install
# Run on Azure VM (Ubuntu 22.04 + T4)
# ⚠️ 실행 후 재부팅 필요! 재부팅 후 02 스크립트 실행
# ============================================

set -e

echo "=========================================="
echo "  BridgeCast AI - NVIDIA Driver Setup"
echo "=========================================="

# Update system
echo "[1/3] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
echo "[2/3] Installing NVIDIA drivers..."
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install

# Verify (will work after reboot)
echo "[3/3] Driver installed. Rebooting in 5 seconds..."
echo ""
echo "⚠️  재부팅 후 다시 SSH 접속해서 확인:"
echo "    ssh azureuser@<VM_IP>"
echo "    nvidia-smi    # T4 16GB 보이면 성공!"
echo ""
echo "그 다음:"
echo "    cd azure_som"
echo "    ./scripts/02_setup_unisign.sh"
echo ""

sleep 5
sudo reboot
