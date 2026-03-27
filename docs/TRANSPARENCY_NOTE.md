# Transparency Note — BridgeCast AI

> This document follows Microsoft's [Transparency Note guidelines](https://learn.microsoft.com/en-us/azure/ai-services/responsible-use-of-ai-overview) to clearly communicate the purpose, capabilities, and limitations of BridgeCast AI.

---

## What is BridgeCast AI?

BridgeCast AI is a real-time bidirectional communication platform that bridges sign language and spoken language in video meetings. It enables Deaf and hard-of-hearing participants to communicate naturally alongside hearing colleagues — using sign language as input and receiving information through a signing avatar, captions, and text.

### Intended Use

- **Primary**: Real-time accessible meetings between Deaf/HoH and hearing participants
- **Scenario**: Enterprise video conferencing, team meetings, one-on-ones
- **Users**: Organizations seeking to create inclusive meeting environments

### Not Intended For

- Medical diagnosis or emergency communication where accuracy is life-critical
- Legal proceedings requiring certified interpretation
- Replacement for qualified human sign language interpreters in high-stakes settings
- Surveillance or monitoring of Deaf individuals without consent

---

## How Does It Work?

### Sign Language → Speech (Sign-to-Text-to-Speech)

1. **Video capture**: User's webcam captures sign language at 30fps
2. **Pose estimation**: RTMPose extracts 133 body/hand/face keypoints
3. **Feature extraction**: ST-GCN processes temporal sequences of keypoints
4. **Recognition**: mT5 decoder generates natural language text from sign features
5. **Speech synthesis**: Azure TTS converts recognized text to spoken audio

**Confidence scoring**: Each recognition result includes a confidence score (0-100%). Users see this score in the UI, so they can judge reliability themselves.

### Speech → Sign Avatar (Text-to-Sign)

1. **Speech recognition**: Azure STT converts spoken audio to text in real-time
2. **Gloss conversion**: Azure OpenAI converts English text to ASL gloss sequence
3. **Animation mapping**: Each gloss token maps to pre-defined sign animation data
4. **Avatar rendering**: Three.js 3D avatar performs the sign sequence

**Known vocabulary**: The avatar currently supports 114 ASL signs. Text containing unsupported signs is fingerspelled letter-by-letter, and users are notified when this occurs.

---

## Capabilities and Limitations

### What It Does Well

| Capability | Details |
|-----------|---------|
| Real-time STT | < 500ms latency via Azure Speech Services |
| Common ASL signs | 114 pre-mapped signs covering meeting scenarios |
| Emotion detection | Sentiment tags on transcribed utterances |
| Speaker attribution | Identifies who said what (voice and sign) |
| Content moderation | Azure Content Safety filters harmful content |

### Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **ASL-centric** | Currently optimized for American Sign Language; other sign languages (BSL, KSL, etc.) have lower accuracy | Modular architecture allows adding new sign language models. UI clearly states which model is active. |
| **Vocabulary size** | Avatar knows 114 signs; complex sentences may be partially fingerspelled | Unknown words are fingerspelled. Confidence score drops visibly to alert users. |
| **Lighting sensitivity** | Sign recognition accuracy decreases in poor lighting or with occluded hands | UI displays confidence score so users can assess reliability. Recommends adequate lighting in onboarding. |
| **Single signer** | Currently processes one signer at a time per camera feed | Meeting room assigns one camera per Deaf participant. |
| **Cultural nuance** | Regional sign language variations (e.g., Southern ASL vs. standard ASL) may not be captured | Modular model architecture supports future regional variants. |
| **Emotion inference** | Sentiment detection may not accurately reflect Deaf cultural expression norms | Emotion tags are suggestions, not definitive. Users can correct or dismiss them. |
| **Avatar expressiveness** | 3D avatar cannot fully replicate the nuance of a human signer | Avatar is a supplement, not a replacement. Text captions are always available alongside. |

### Failure Modes

- **Low confidence recognition**: When the model is uncertain (< 60% confidence), the UI displays a yellow warning indicator and the recognized text is marked as "uncertain"
- **Unrecognized signs**: Signs outside the training vocabulary return "unknown" — the system does NOT guess
- **Network interruption**: If connection to Azure services drops, the system falls back to local processing where possible and clearly indicates degraded mode

---

## Fairness Considerations

### Who Benefits

- Deaf and hard-of-hearing professionals in workplace meetings
- Hearing colleagues who want to communicate naturally with Deaf team members
- Organizations committed to accessibility and inclusion

### Who Might Be Disadvantaged

- **Users of less-common sign languages**: The current model is trained primarily on ASL (WLASL dataset). Users of other sign languages will experience lower accuracy.
- **Users with non-standard signing**: Individuals with motor differences or non-standard signing styles may see reduced recognition accuracy.
- **Users with darker skin tones in poor lighting**: Pose estimation accuracy can vary; we recommend adequate, even lighting.

### What We've Done

- **Modular design**: Each sign language gets its own recognition model, avoiding a one-size-fits-all approach that would privilege one language over others
- **Equal modality treatment**: Sign and speech are treated as equally valid input/output — the UI does not present either as "primary" or "assistive"
- **Confidence transparency**: All AI outputs include confidence scores visible to the user
- **Testing diversity**: Designed to be tested across varied skin tones, signing styles, and lighting conditions

---

## Privacy and Security

### Data Collection

| Data Type | Collected? | Stored? | Duration | Purpose |
|-----------|-----------|---------|----------|---------|
| Video (webcam) | Yes, in-session | No (default) | Real-time only | Sign language recognition |
| Audio (microphone) | Yes, in-session | No (default) | Real-time only | Speech-to-text |
| Transcript text | Yes | Optional (Cosmos DB) | User-controlled | Meeting notes |
| Meeting recordings | Optional | Optional (Blob Storage) | User-controlled | Review and sharing |

### Key Principles

- **No permanent video/audio storage by default** — streams are processed in real-time and discarded
- **User controls data retention** — recording and transcript storage are opt-in
- **Encryption** — all data encrypted in transit (TLS) and at rest (Azure Storage encryption)
- **Secret management** — all API keys and credentials stored in Azure Key Vault, never in code
- **Content Safety** — Azure Content Safety API filters all text output in real-time for harmful content (hate speech, violence, self-harm, sexual content)

### Compliance Considerations

- Video streams are processed by Azure Speech and Azure AI services under [Microsoft's AI service terms](https://learn.microsoft.com/en-us/legal/cognitive-services/speech-service/speech-to-text/data-privacy-security)
- Meeting data stored in Cosmos DB and Blob Storage follows Azure's regional data residency policies
- Organizations should conduct their own DPIA (Data Protection Impact Assessment) before production deployment

---

## Evaluation and Testing

### How We Measure Quality

| Metric | How Measured | Target |
|--------|-------------|--------|
| Sign recognition accuracy | Correct predictions / total predictions on WLASL test set | > 70% top-5 accuracy |
| STT accuracy | Word Error Rate via Azure Speech | < 15% WER |
| Avatar response latency | Time from text input to first sign animation frame | < 300ms |
| End-to-end latency | Time from sign input to TTS audio output | < 2 seconds |
| Content Safety precision | False positive rate on benign content | < 5% |

### What We Cannot Guarantee

- **100% recognition accuracy** — sign language recognition is an active research area. Errors will occur.
- **Perfect emotion detection** — sentiment analysis is probabilistic, not deterministic.
- **Cultural appropriateness** — the system may not capture all cultural nuances of Deaf communication.
- **Interpreter-level quality** — BridgeCast AI is a communication tool, not a certified interpreter.

---

## Recommendations for Deployment

1. **Always provide human interpreter option** — BridgeCast AI should supplement, not replace, human interpreters in critical settings
2. **Inform all participants** — all meeting participants should know that AI is being used for sign recognition and transcription
3. **Monitor confidence scores** — if recognition confidence consistently drops below 60%, check lighting, camera angle, and network quality
4. **Review meeting notes** — AI-generated summaries should be reviewed by a human before being used for official decisions
5. **Provide feedback channels** — users should be able to report recognition errors to improve the system
6. **Test with your user population** — recognition accuracy varies by signing style; test with your actual users before relying on it

---

## Contact

**Team BridgeCast** — Somi, Ollie, TSAIHSUAN

For questions about this Transparency Note or the responsible use of BridgeCast AI, please open an issue on our [GitHub repository](https://github.com/BridgeCastAI/azure_som).

---

*This Transparency Note was last updated on March 23, 2026.*
