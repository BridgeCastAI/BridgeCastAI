# BridgeCast AI — API Reference

**Base URL:** `http://localhost:8000`

## Health & System

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `GET` | `/health` | Health check (all 15 services) | All |
| `GET` | `/analytics/dashboard` | Aggregated dashboard analytics | Azure Monitor |
| `GET` | `/analytics/meeting/{id}` | Meeting analytics | Azure Monitor |
| `GET` | `/rai-report` | RAI fairness & error report | RAI Toolbox |

## Real-Time Communication

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `WS` | `/ws/meeting/{room_id}` | WebRTC signaling + real-time message broadcast | WebSocket P2P |
| `WS` | `/ws/stt` | WebSocket real-time speech-to-text | Azure Speech |
| `POST` | `/signalr/negotiate` | SignalR client connection negotiation | Azure SignalR |
| `POST` | `/signalr/broadcast` | Broadcast message to all/group | Azure SignalR |

## Sign Language Recognition

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `POST` | `/predict` | Uni-Sign language recognition (video) | GPU VM |
| `POST` | `/predict/ksl` | Korean Sign Language recognition | GPU VM |
| `GET` | `/ksl/status` | KSL model status & capabilities | GPU VM |

## Sign Language Avatar

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `POST` | `/avatar/sign` | Text → ASL/KSL/TSL sign animation | Azure OpenAI |
| `POST` | `/avatar/generate` | Generate avatar animation with timing | Azure OpenAI |
| `GET` | `/avatar/vocabulary` | Supported sign vocabulary (full details) | Azure OpenAI |
| `GET` | `/avatar/signs` | List available sign names | Azure OpenAI |

## Speech & Translation

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `POST` | `/tts` | Text-to-speech synthesis | Azure Speech |
| `POST` | `/translate` | Multi-language translation | Azure Translator |
| `POST` | `/translate/batch` | Batch translation (multiple texts) | Azure Translator |

## Content Safety & Privacy (RAI)

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `POST` | `/safety/check` | Content safety check | Azure Content Safety |
| `GET` | `/safety/stats` | Content safety statistics | Azure Content Safety |
| `POST` | `/pii/detect` | Detect & redact PII from text | Azure AI Language |
| `POST` | `/pii/redact` | Redact PII (returns clean text only) | Azure AI Language |
| `POST` | `/sentiment` | Sentiment analysis | Azure AI Language |

## Meetings & Storage

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `POST` | `/meeting/notes` | AI meeting notes generation | Azure OpenAI |
| `POST` | `/meeting/save` | Save meeting to Cosmos DB | Azure Cosmos DB |
| `GET` | `/meeting/{id}` | Retrieve meeting data | Azure Cosmos DB |
| `GET` | `/meetings` | List all meetings | Azure Cosmos DB |
| `DELETE` | `/meeting/{id}` | Delete meeting and associated files | Azure Cosmos DB |
| `POST` | `/meeting/export-pdf` | Export meeting transcript as PDF | Azure Blob Storage |
| `POST` | `/storage/upload-recording` | Upload recording to Blob | Azure Blob Storage |
| `POST` | `/storage/upload-pdf` | Upload PDF to Blob | Azure Blob Storage |
| `GET` | `/storage/meeting-files/{id}` | List meeting files | Azure Blob Storage |

## Video Rooms (ACS)

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `POST` | `/room/create` | Create ACS video room | Azure Communication Services |
| `POST` | `/room/join` | Join room and get access token | Azure Communication Services |
| `GET` | `/room/{id}` | Get room details | Azure Communication Services |
| `GET` | `/room/{id}/token` | Get participant access token | Azure Communication Services |
| `GET` | `/room/{id}/participants` | List room participants | Azure Communication Services |

## Serverless Functions

| Method | Endpoint | Description | Azure Service |
|--------|----------|-------------|---------------|
| `POST` | `/functions/summarize` | Serverless meeting summary | Azure Functions |
| `POST` | `/functions/emergency-alert` | Emergency keyword detection | Azure Functions |
| `POST` | `/functions/accessibility-report` | Accessibility metrics report | Azure Functions |
