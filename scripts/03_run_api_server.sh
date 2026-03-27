#!/bin/bash
# ============================================
# Step 3: Run Uni-Sign API Server
# Run on Azure VM AFTER 02 완료 후
# ============================================

set -e

echo "=========================================="
echo "  BridgeCast AI - API Server Start"
echo "=========================================="

# Source conda
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate unisign

# Install API server dependencies
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

pip install -q fastapi uvicorn python-multipart

# Copy server to home for easy access
cp "$REPO_DIR/server/api_server.py" "$HOME/api_server.py" 2>/dev/null || true

echo ""
echo "🚀 Starting API server on port 8000..."
echo ""
echo "⚠️  Azure Portal에서 포트 8000 열었는지 확인!"
echo "   VM → Networking → Add inbound rule → Port 8000"
echo ""
echo "로컬에서 테스트:"
echo "   curl http://<VM_IP>:8000/health"
echo "   python client/test_api.py --host <VM_IP>"
echo ""

cd "$HOME"
python api_server.py --host 0.0.0.0 --port 8000
