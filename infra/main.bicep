// ============================================================================
// BridgeCast AI — Bicep IaC entry point
//
// Provisions all 15 Azure services in a single deployment. Each service lives
// in its own module under ./modules/ for isolation and reusability.
//
// Usage:
//   az deployment group create \
//     --resource-group bridgecast-rg \
//     --template-file infra/main.bicep \
//     --parameters projectName=bridgecast location=eastus
//
// NOTE: Module ordering here is purely for human readability — Bicep resolves
// the dependency graph automatically, so reordering won't break anything.
// ============================================================================

// We assume the resource group already exists. Creating it separately keeps
// this template simpler and avoids needing subscription-level permissions.
targetScope = 'resourceGroup'

@description('Project name prefix for all resources')
param projectName string = 'bridgecast'

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Azure OpenAI model deployment name')
param openAiDeploymentName string = 'gpt-4o'

@description('GPU VM admin username')
param vmAdminUsername string = 'azureuser'

@description('GPU VM admin SSH public key')
@secure()
param vmAdminSshKey string = ''

// ============================================================================
// Variables
// ============================================================================

// uniqueString() is deterministic per RG + project name, so each team member
// deploying to the same subscription gets collision-free resource names.
var uniqueSuffix = uniqueString(resourceGroup().id, projectName)
var baseName = '${projectName}-${uniqueSuffix}'

// ============================================================================
// 1/15 — Azure OpenAI Service
// Real-time GPT-4o for meeting summarisation and sign-language context
// ============================================================================

module openAi 'modules/openai.bicep' = {
  name: 'openai'
  params: {
    name: '${projectName}-openai-${uniqueSuffix}'
    location: location
    deploymentName: openAiDeploymentName
  }
}

// ============================================================================
// 2/15 — Azure Speech Service
// STT + TTS for spoken-language participants
// ============================================================================

module speech 'modules/speech.bicep' = {
  name: 'speech'
  params: {
    name: '${projectName}-speech-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 3/15 — Azure AI Vision
// Video frame analysis for sign-language pose extraction
// ============================================================================

module vision 'modules/vision.bicep' = {
  name: 'vision'
  params: {
    name: '${projectName}-vision-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 4/15 — Azure Content Safety
// RAI: screens generated text and translations for harmful content
// ============================================================================

module contentSafety 'modules/contentSafety.bicep' = {
  name: 'contentSafety'
  params: {
    name: '${projectName}-safety-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 5/15 — Azure AI Language (PII Detection & Sentiment)
// SECURITY: PII redaction runs before any text is persisted to Cosmos DB
// ============================================================================

module language 'modules/language.bicep' = {
  name: 'language'
  params: {
    name: '${projectName}-language-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 6/15 — Azure Translator
// location='global' is required by Azure for this resource type
// ============================================================================

module translator 'modules/translator.bicep' = {
  name: 'translator'
  params: {
    name: '${projectName}-translator-${uniqueSuffix}'
    location: 'global'
  }
}

// ============================================================================
// 7/15 — Azure Cosmos DB
// Meeting transcripts, user profiles, analytics — NoSQL for flexible schema
// ============================================================================

module cosmosDb 'modules/cosmosdb.bicep' = {
  name: 'cosmosDb'
  params: {
    name: '${projectName}-cosmos-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 8/15 — Azure Communication Services
// Video calling backbone. location='global' — Azure requires it for ACS.
// ============================================================================

module communication 'modules/communication.bicep' = {
  name: 'communication'
  params: {
    name: '${projectName}-acs-${uniqueSuffix}'
    location: 'global'
  }
}

// ============================================================================
// 9/15 — Azure SignalR Service
// WebSocket fan-out for real-time captions and sign overlays
// ============================================================================

module signalR 'modules/signalr.bicep' = {
  name: 'signalR'
  params: {
    name: '${projectName}-signalr-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 10/15 — Azure Blob Storage
// Meeting recordings, exported PDFs, avatar assets
// ============================================================================

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    name: '${projectName}store${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 11/15 — Azure Key Vault
// SECURITY: all service keys and connection strings live here, not in env vars
// ============================================================================

module keyVault 'modules/keyvault.bicep' = {
  name: 'keyVault'
  params: {
    name: '${projectName}-kv-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 12/15 — Azure Application Insights + Log Analytics
// Backs the monitor_service.py telemetry pipeline
// ============================================================================

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring'
  params: {
    name: '${projectName}-insights-${uniqueSuffix}'
    logAnalyticsName: '${projectName}-logs-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 13/15 — Azure Functions
// Lightweight triggers: post-meeting PDF export, scheduled cleanup, etc.
// ============================================================================

module functions 'modules/functions.bicep' = {
  name: 'functions'
  params: {
    name: '${projectName}-func-${uniqueSuffix}'
    location: location
    storageAccountName: storage.outputs.name
  }
}

// ============================================================================
// 14/15 — Azure App Service
// Hosts the FastAPI backend (server/) and static client files
// ============================================================================

module appService 'modules/appservice.bicep' = {
  name: 'appService'
  params: {
    name: '${projectName}-app-${uniqueSuffix}'
    location: location
  }
}

// ============================================================================
// 15/15 — Azure GPU VM (NC4as_T4_v3) — Uni-Sign Inference
// NC4as_T4_v3 is the cheapest NVIDIA T4 SKU on Azure, and the T4 handles
// Uni-Sign inference at ~30 FPS which is plenty for real-time meetings.
// ============================================================================

module gpuVm 'modules/gpuvm.bicep' = {
  name: 'gpuVm'
  params: {
    name: '${projectName}-gpu-${uniqueSuffix}'
    location: location
    adminUsername: vmAdminUsername
    adminSshKey: vmAdminSshKey
  }
}

// ============================================================================
// Outputs — used by CI/CD and by server/.env generation scripts
// ============================================================================

output openAiEndpoint string = openAi.outputs.endpoint
output speechEndpoint string = speech.outputs.endpoint
output cosmosEndpoint string = cosmosDb.outputs.endpoint
output signalREndpoint string = signalR.outputs.endpoint
output storageAccountName string = storage.outputs.name
output appInsightsKey string = monitoring.outputs.instrumentationKey
output appServiceUrl string = appService.outputs.url
output gpuVmPublicIp string = gpuVm.outputs.publicIp
