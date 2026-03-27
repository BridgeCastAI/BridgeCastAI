"""
BridgeCast AI — Azure App Service Configuration
Deployment config + health monitoring for the web application.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# App Service configuration
APP_SERVICE_CONFIG = {
    "name": "bridgecast-app",
    "resource_group": "bridgecast-rg",
    "plan": "bridgecast-plan",
    "runtime": "PYTHON:3.9",
    "region": "eastus",
    "sku": "B1",  # Basic tier (sufficient for hackathon)
}

# Startup command for App Service
STARTUP_COMMAND = "gunicorn meeting_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"


def get_app_settings() -> dict:
    """
    Generate App Service application settings from environment variables.
    These get configured in Azure Portal > App Service > Configuration.
    """
    return {
        # Azure Speech
        "AZURE_SPEECH_KEY": os.getenv("AZURE_SPEECH_KEY", ""),
        "AZURE_SPEECH_REGION": os.getenv("AZURE_SPEECH_REGION", "eastus"),
        # Azure OpenAI
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY", ""),
        "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        # Azure Translator
        "AZURE_TRANSLATOR_KEY": os.getenv("AZURE_TRANSLATOR_KEY", ""),
        "AZURE_TRANSLATOR_ENDPOINT": os.getenv("AZURE_TRANSLATOR_ENDPOINT", ""),
        "AZURE_TRANSLATOR_REGION": os.getenv("AZURE_TRANSLATOR_REGION", "eastus"),
        # Azure Content Safety
        "AZURE_CONTENT_SAFETY_ENDPOINT": os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT", ""),
        "AZURE_CONTENT_SAFETY_KEY": os.getenv("AZURE_CONTENT_SAFETY_KEY", ""),
        # Azure Cosmos DB
        "AZURE_COSMOS_ENDPOINT": os.getenv("AZURE_COSMOS_ENDPOINT", ""),
        "AZURE_COSMOS_KEY": os.getenv("AZURE_COSMOS_KEY", ""),
        # Azure Communication Services
        "AZURE_COMMUNICATION_CONNECTION_STRING": os.getenv("AZURE_COMMUNICATION_CONNECTION_STRING", ""),
        # Azure Key Vault
        "AZURE_KEYVAULT_URL": os.getenv("AZURE_KEYVAULT_URL", ""),
        # Azure Functions
        "AZURE_FUNCTIONS_URL": os.getenv("AZURE_FUNCTIONS_URL", ""),
        # Azure Application Insights
        "AZURE_APPINSIGHTS_CONNECTION_STRING": os.getenv("AZURE_APPINSIGHTS_CONNECTION_STRING", ""),
        # GPU VM (sign recognition API)
        "GPU_VM_URL": os.getenv("GPU_VM_URL", "http://localhost:8000"),
        # App settings
        "SCM_DO_BUILD_DURING_DEPLOYMENT": "true",
        "WEBSITES_PORT": "8000",
    }


def generate_deployment_script() -> str:
    """Generate Azure CLI deployment script for App Service."""
    cfg = APP_SERVICE_CONFIG
    settings = get_app_settings()

    settings_args = " ".join(
        f'{k}="{v}"' for k, v in settings.items() if v
    )

    return f"""#!/bin/bash
# BridgeCast AI — Azure App Service Deployment Script
# Run this from the azure_som/server/ directory

set -e

RESOURCE_GROUP="{cfg['resource_group']}"
APP_NAME="{cfg['name']}"
PLAN_NAME="{cfg['plan']}"
REGION="{cfg['region']}"
SKU="{cfg['sku']}"

echo "=== Creating App Service Plan ==="
az appservice plan create \\
    --name $PLAN_NAME \\
    --resource-group $RESOURCE_GROUP \\
    --sku $SKU \\
    --is-linux

echo "=== Creating Web App ==="
az webapp create \\
    --resource-group $RESOURCE_GROUP \\
    --plan $PLAN_NAME \\
    --name $APP_NAME \\
    --runtime "{cfg['runtime']}"

echo "=== Configuring App Settings ==="
az webapp config appsettings set \\
    --resource-group $RESOURCE_GROUP \\
    --name $APP_NAME \\
    --settings {settings_args}

echo "=== Setting Startup Command ==="
az webapp config set \\
    --resource-group $RESOURCE_GROUP \\
    --name $APP_NAME \\
    --startup-file "{STARTUP_COMMAND}"

echo "=== Deploying Code ==="
az webapp deploy \\
    --resource-group $RESOURCE_GROUP \\
    --name $APP_NAME \\
    --src-path ../server/ \\
    --type zip

echo ""
echo "=== Deployment Complete! ==="
echo "URL: https://$APP_NAME.azurewebsites.net"
echo "Health: https://$APP_NAME.azurewebsites.net/health"
"""


def generate_requirements_for_appservice() -> str:
    """Generate requirements.txt optimized for App Service deployment."""
    return """fastapi==0.109.0
uvicorn==0.27.0
gunicorn==21.2.0
python-multipart==0.0.6
python-dotenv==1.0.0
requests>=2.31.0
websockets>=12.0
azure-cognitiveservices-speech==1.35.0
openai>=1.12.0
azure-cosmos>=4.5.1
azure-communication-rooms>=1.1.0
azure-communication-identity>=1.5.0
azure-identity>=1.15.0
azure-keyvault-secrets>=4.7.0
azure-monitor-opentelemetry>=1.2.0
"""


def get_health_status() -> dict:
    """Get comprehensive health status for the application."""
    services = {}

    # Check each Azure service configuration
    checks = {
        "speech": bool(os.getenv("AZURE_SPEECH_KEY")),
        "openai": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
        "translator": bool(os.getenv("AZURE_TRANSLATOR_KEY")),
        "content_safety": bool(os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")),
        "cosmos_db": bool(os.getenv("AZURE_COSMOS_ENDPOINT")),
        "communication": bool(os.getenv("AZURE_COMMUNICATION_CONNECTION_STRING")),
        "keyvault": bool(os.getenv("AZURE_KEYVAULT_URL")),
        "functions": bool(os.getenv("AZURE_FUNCTIONS_URL")),
        "app_insights": bool(os.getenv("AZURE_APPINSIGHTS_CONNECTION_STRING")),
        "gpu_vm": bool(os.getenv("GPU_VM_URL")),
    }

    configured = sum(1 for v in checks.values() if v)
    total = len(checks)

    return {
        "status": "ok" if configured > 3 else "partial",
        "services_configured": f"{configured}/{total}",
        "services": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }
