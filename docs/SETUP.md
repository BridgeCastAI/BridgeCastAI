# BridgeCast AI — Setup Guide

## Prerequisites

- Python 3.9+
- Azure subscription with services provisioned
- Mac or Windows

## 1. Clone & Configure

```bash
git clone https://github.com/BridgeCastAI/BridgeCastAI.git
cd BridgeCastAI/server
cp .env.example .env
# Fill in your Azure credentials in .env
```

## 2. Install & Run

```bash
pip install -r requirements.txt
python meeting_api.py --host 0.0.0.0 --port 8000
```

## 3. Verify

```bash
curl http://localhost:8000/health
# Returns JSON with status of all 15 Azure services
```

## 4. Open Frontend

Open directly in your browser — no build step required:

| Page | File | Description |
|------|------|------------|
| Landing Page | `client/landing.html` | Dark/Light mode, i18n (EN/KO/ZH-TW), glassmorphism UI |
| Meeting Room | `client/meeting-room.html` | STT, transcript, PDF export, avatar, Content Safety badge |
| Avatar Demo | `client/avatar.html` | 3D signing avatar with 139 signs, demo phrases |

## 5. GPU VM Setup (Sign Language Recognition)

```bash
# SSH into Azure GPU VM (NC4as_T4_v3)
ssh azureuser@<VM_PUBLIC_IP>

# Run setup scripts in order
./scripts/01_install_nvidia.sh && sudo reboot
# Reconnect after reboot
./scripts/02_setup_unisign.sh
./scripts/03_run_api_server.sh
```

## 6. Infrastructure Deployment (Azure Bicep)

```bash
az deployment group create \
  --resource-group bridgecast-rg \
  --template-file infra/main.bicep \
  --parameters projectName=bridgecast location=eastus
```

This provisions all 15 Azure services in a single command.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Not connected to server" banner | Start the server: `python meeting_api.py` |
| Avatar not appearing | Ensure Three.js CDN loads (needs internet) |
| STT not working | Check `AZURE_SPEECH_KEY` in `.env` |
| GPU VM connection refused | Open port 8000 in NSG rules |
| WebRTC video not connecting | Both users must be on same `?room=` parameter |
