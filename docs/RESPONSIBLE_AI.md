# Responsible AI Report — BridgeCast AI

This document details how BridgeCast AI implements Microsoft's six Responsible AI principles and the specific Azure tools and practices we use.

---

## 1. Fairness

### Principle
AI systems should treat all people fairly and avoid affecting similarly situated groups in different ways.

### Our Implementation

- **Equal modality treatment**: Sign language and speech are two equally valid input/output channels. Neither is marked as "primary" or "assistive" in the UI or architecture.
- **Modular sign language support**: Each sign language (ASL, KSL, BSL, TSL, etc.) gets its own recognition model. This prevents linguistic bias from a one-size-fits-all model.
- **Vocabulary transparency**: The avatar clearly supports 114 signs. When a word is not in the vocabulary, it is fingerspelled — never silently dropped.

### Azure Tools Used
- **Azure OpenAI** — ASL gloss generation with explicit vocabulary bounds
- **Uni-Sign** (research model) — trained on WLASL dataset with documented demographics

### Evidence in Code
- `avatar_service.py`: `get_vocabulary()` returns the full list of supported signs
- `meeting_api.py`: `/avatar/vocabulary` endpoint exposes vocabulary for audit
- `avatar_service.py`: `unknown_signs` count returned in every response

---

## 2. Reliability & Safety

### Principle
AI systems should perform reliably and safely under expected and unexpected conditions.

### Our Implementation

- **Confidence scoring**: Every sign recognition result includes a confidence percentage. Low-confidence results (< 60%) are visually flagged in the UI.
- **Graceful degradation**: If Azure services are unavailable, the system falls back to local processing where possible.
- **Emergency detection**: `functions_service.py` detects urgent keywords (fire, earthquake, evacuate, emergency, help, danger, 911) and triggers immediate visual alerts for Deaf participants.
- **Content Safety**: All text output passes through Azure Content Safety API before display.

### Azure Tools Used
- **Azure Content Safety** — real-time text moderation for hate, violence, self-harm, sexual content
- **Azure Monitor / Application Insights** — tracks errors, latency, and service health
- **Azure Functions** — serverless emergency alert processing

### Evidence in Code
- `content_safety_service.py`: `check_text_safety()` returns category scores and blocked status
- `meeting_api.py`: `/safety/check` endpoint for on-demand content checking
- `monitor_service.py`: `track_event()`, `track_metric()` for all critical operations
- `functions_service.py`: `trigger_emergency_alert()` with keyword detection
- `meeting_api.py`: `/health` endpoint reports status of all 13 services

---

## 3. Privacy & Security

### Principle
AI systems should be secure and respect privacy.

### Our Implementation

- **No permanent video/audio storage**: Webcam and microphone streams are processed in real-time and immediately discarded. No video frames are saved to disk or cloud by default.
- **User-controlled data retention**: Meeting transcripts and recordings are stored only when the user explicitly opts in.
- **Secret management**: All API keys, connection strings, and credentials are stored in Azure Key Vault — never hardcoded or committed to source control.
- **Encryption**: All data encrypted in transit (TLS 1.2+) and at rest (Azure Storage Service Encryption).
- **Minimal data collection**: We collect only what is necessary for the meeting to function.

### Azure Tools Used
- **Azure Key Vault** — centralized secret storage with access policies
- **Azure Blob Storage** — encrypted at-rest storage with time-limited SAS tokens for sharing
- **Azure Cosmos DB** — encrypted document storage with per-meeting access control

### Evidence in Code
- `keyvault_service.py`: `load_all_secrets()` bootstraps credentials from Key Vault at startup
- `blob_service.py`: `_generate_sas_url()` creates time-limited (24-72hr) download links
- `.env.example`: All credentials are environment variables, never hardcoded
- `.gitignore`: `.env`, `*.pem`, `*.key` files excluded from version control

---

## 4. Inclusiveness

### Principle
AI systems should empower everyone and engage people.

### Our Implementation

- **Bidirectional communication**: Deaf users can both receive information (via captions + signing avatar) AND contribute (via sign language → speech). This is the core innovation — moving from one-way accessibility to true participation.
- **Speaker attribution**: Meeting notes attribute contributions to Deaf participants by name — "Somi (sign): ..." — ensuring their input is formally recorded.
- **Accessibility Dashboard**: Real-time visualization of participation balance between sign and voice users, providing quantitative evidence of inclusiveness.
- **Multilingual UI**: Interface available in English, Korean, and Traditional Chinese.
- **Dark/Light mode**: Respects user visual preferences.

### Azure Tools Used
- **Azure Speech Services** — STT/TTS in multiple languages
- **Azure Translator** — real-time translation for multilingual meetings
- **Azure Communication Services** — accessible video rooms

### Evidence in Code
- `meeting-room.html`: Transcript entries tagged with `🎤 Voice` or `🤟 Sign` badges
- `landing.html`: `data-i18n` and `data-i18n-html` attributes on 175+ elements
- `avatar_service.py`: 114 ASL signs for meeting scenarios
- `meeting-room.html`: Accessibility Dashboard with participation balance metrics

---

## 5. Transparency

### Principle
AI systems should be understandable. People should be aware of system capabilities and limitations.

### Our Implementation

- **Confidence scores visible**: Sign recognition confidence percentage is displayed in the meeting room UI — users always know how certain the AI is.
- **Model identification**: The UI shows which model is active ("Uni-Sign (WLASL)") so users know what's processing their input.
- **Vocabulary disclosure**: The `/avatar/vocabulary` endpoint and UI disclose exactly which signs the avatar can perform.
- **Transparency Note**: A comprehensive [TRANSPARENCY_NOTE.md](TRANSPARENCY_NOTE.md) documents capabilities, limitations, failure modes, and deployment recommendations.
- **Open source**: Full source code is available on GitHub for audit.

### Azure Tools Used
- **Azure Monitor** — all AI operations logged with latency and accuracy metrics
- **Azure OpenAI** — prompts are documented in code with clear system instructions

### Evidence in Code
- `meeting-room.html`: `signAvgConfidence` and `signConfidenceBar` UI elements
- `meeting-room.html`: `signLatency` display showing real-time processing speed
- `api_server.py`: `/predict` returns `latency_ms` with every response
- `avatar_service.py`: Response includes `known_signs`, `unknown_signs`, `sign_count`
- `TRANSPARENCY_NOTE.md`: Full documentation of system behavior

---

## 6. Accountability

### Principle
People should be accountable for AI systems.

### Our Implementation

- **Human oversight recommended**: The Transparency Note explicitly states that BridgeCast AI should supplement, not replace, human interpreters in critical settings.
- **Audit trail**: Azure Monitor / Application Insights tracks all API calls, latency, errors, and user interactions.
- **Meeting analytics**: The `/analytics/meeting/{id}` endpoint provides post-meeting analysis of system performance.
- **Error reporting**: Users can flag incorrect recognitions. Meeting notes are explicitly marked as AI-generated and should be reviewed by a human.
- **Team accountability**: Team BridgeCast (Somi, Ollie, TSAIHSUAN) is accountable for the system's behavior and actively monitors its outputs.

### Azure Tools Used
- **Azure Monitor / Application Insights** — comprehensive telemetry and alerting
- **Azure Cosmos DB** — persistent meeting records for audit

### Evidence in Code
- `monitor_service.py`: `track_sign_recognition()`, `track_stt_recognition()` log every AI decision
- `meeting_api.py`: HTTP middleware tracks all requests with status codes and latency
- `meeting_api.py`: `/analytics/meeting/{id}` endpoint for post-meeting review
- `rai_assessment.py`: **Microsoft Responsible AI Toolbox** integration — fairness assessment across skin tones (Fitzpatrick I-VI), signing styles, and lighting conditions; error analysis for worst-performing cohorts; interpretability report with feature importance
- `meeting_api.py`: `GET /rai-report` endpoint exposes the full RAI assessment as JSON

---

## Impact Assessment

This section documents the potential impacts of BridgeCast AI on stakeholders, following the Microsoft Responsible AI Impact Assessment framework.

### Stakeholder Analysis

| Stakeholder | Positive Impact | Potential Risk | Mitigation |
|------------|----------------|---------------|-----------|
| **Deaf/HoH participants** | Full meeting participation for the first time; voice representation via TTS; formal meeting attribution | Incorrect sign recognition may misrepresent their message | Confidence scores shown; human review of notes recommended |
| **Hearing participants** | Natural communication with Deaf colleagues without learning sign language | May over-rely on AI instead of learning basic signs or Deaf culture | Onboarding materials encourage cultural awareness |
| **Organizations** | Legal compliance with accessibility mandates; inclusive culture | May use BridgeCast as a checkbox instead of genuine inclusion | Transparency Note recommends systemic inclusion, not just tool deployment |
| **Human interpreters** | Can focus on high-stakes settings; AI handles routine meetings | Job displacement concern for routine interpretation | Explicitly positioned as supplement, not replacement |

### Risk Assessment

| Risk | Likelihood | Severity | Mitigation |
|------|-----------|----------|-----------|
| Misrecognition of sign causing miscommunication | Medium | Medium | Confidence scores, "uncertain" labels, human review |
| Privacy breach of video/audio data | Low | High | No permanent storage default, Key Vault, encryption |
| Bias against non-ASL sign languages | Medium | Medium | Modular architecture, clearly labeled active model |
| Emergency alert false positive/negative | Low | High | Keyword-based detection with human escalation path |
| Content Safety over-filtering Deaf cultural expressions | Low | Medium | Adjustable thresholds, user override capability |

### Intended vs. Unintended Uses

| Use Case | Intended? | Notes |
|----------|----------|-------|
| Workplace team meetings | ✅ Yes | Primary use case |
| Education and classrooms | ✅ Yes | Expansion scenario |
| Medical consultations | ⚠️ Caution | Should not replace certified medical interpreter |
| Legal proceedings | ❌ Not intended | Requires certified court interpreter |
| Surveillance of Deaf individuals | ❌ Prohibited | Explicitly stated in Transparency Note |

---

## WCAG 2.1 Accessibility Compliance

BridgeCast AI follows WCAG 2.1 Level AA guidelines for its web interfaces:

| WCAG Criterion | Implementation |
|---------------|----------------|
| **1.1.1 Non-text Content** | All icons have `aria-hidden="true"` with adjacent text labels; avatar animations accompanied by text captions |
| **1.3.1 Info and Relationships** | Semantic HTML (`<header>`, `<main>`, `<footer>`, `<nav>`, `<aside>`) used throughout |
| **1.4.3 Contrast (Minimum)** | Dark mode and light mode both tested for 4.5:1 contrast ratio on text |
| **1.4.11 Non-text Contrast** | UI controls (buttons, toggles) meet 3:1 contrast ratio |
| **2.1.1 Keyboard** | All interactive elements are focusable; modal dialogs trap focus |
| **2.4.1 Bypass Blocks** | Navigation links provide skip-to-content functionality |
| **2.4.4 Link Purpose** | All links and buttons have descriptive text or tooltips |
| **3.1.1 Language of Page** | `<html lang>` attribute updates dynamically with language toggle |
| **3.1.2 Language of Parts** | Language toggle changes `lang` to `en`, `ko`, or `zh-TW` |
| **4.1.2 Name, Role, Value** | Buttons have accessible names via text content or `data-i18n` |

---

## Error Handling & Degradation Scenarios

| Scenario | System Behavior | User Experience |
|----------|----------------|-----------------|
| **Azure Speech service down** | Falls back to browser Web Speech API if available; logs error to Monitor | Banner: "Speech service degraded — using browser fallback" |
| **GPU VM unreachable** | Sign recognition disabled; STT/avatar still work | Sign detection panel shows "Offline"; camera capture paused |
| **Azure OpenAI timeout** | Meeting notes generation retries once, then returns partial results | "Summary generation delayed — please retry" toast |
| **Network disconnected** | WebSocket reconnects automatically (3 retries); offline indicator shown | Red "Not connected" banner with retry button |
| **Content Safety API down** | Text passes through unfiltered; event logged for audit | Normal operation but safety bypass logged |
| **Cosmos DB unavailable** | Meeting data stored in browser localStorage as fallback | "Saving locally — will sync when connection restores" |
| **Low sign recognition confidence** | Yellow warning indicator; text marked as "uncertain" | User sees confidence percentage and can dismiss |

---

## Summary Table

| Principle | Azure Services Used | Implementation Status |
|-----------|-------------------|----------------------|
| **Fairness** | OpenAI, Uni-Sign | ✅ Modular design, vocabulary transparency |
| **Reliability & Safety** | Content Safety, Monitor, Functions | ✅ Confidence scores, emergency alerts, content filtering |
| **Privacy & Security** | Key Vault, Blob Storage, Cosmos DB | ✅ No permanent storage, encryption, secret management |
| **Inclusiveness** | Speech, Translator, Communication Services | ✅ Bidirectional communication, multilingual, speaker attribution |
| **Transparency** | Monitor, OpenAI | ✅ Confidence display, Transparency Note, open source |
| **Accountability** | Monitor, Cosmos DB | ✅ Audit trail, analytics, human oversight recommended |

---

*This report was prepared by Team BridgeCast for the Microsoft Innovation Challenge (March 2026).*
