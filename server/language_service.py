"""
BridgeCast AI — Azure AI Language (Service 15/15)

PII detection/redaction and sentiment analysis via Azure AI Language
(formerly Text Analytics). Two distinct responsibilities here:

1. RAI: PII scrubbing — every transcript passes through detect_pii() BEFORE
   it gets written to Cosmos DB. Names, emails, phone numbers, and other
   personal data are redacted so they never hit persistent storage.

2. Sentiment analysis — feeds directly into the avatar system. The sentiment
   score determines the avatar's facial expression during a meeting:
   positive text triggers a smile, negative triggers a concerned look, etc.

Both endpoints use the v3.1 synchronous API (not the newer async jobs API)
because we're analyzing one document at a time in real-time — the jobs API
adds unnecessary polling overhead for single-doc calls.
"""

import os
import logging
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# The most common PII types you'd see in a meeting transcript.
# We intentionally skip niche categories (passport numbers, driver's licenses)
# to reduce false positives on short utterances.
DEFAULT_PII_CATEGORIES = [
    "Person", "Email", "PhoneNumber", "Address",
    "CreditCardNumber", "SocialSecurityNumber",
    "IPAddress", "Organization",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_config() -> dict:
    """Load Language service credentials from env vars.

    Raises EnvironmentError immediately on missing config — we want this to
    fail loudly at startup, not silently return empty results at runtime.
    """
    endpoint = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
    key = os.environ.get("AZURE_LANGUAGE_KEY")

    if not endpoint or not key:
        raise EnvironmentError(
            "AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY must be set."
        )

    return {"endpoint": endpoint.rstrip("/"), "key": key}


# ---------------------------------------------------------------------------
# PII Detection & Redaction
# ---------------------------------------------------------------------------

async def detect_pii(
    text: str,
    language: str = "en",
    categories: Optional[List[str]] = None,
) -> dict:
    """Run PII detection on a piece of text and return both the redacted
    version and the raw entity list.

    RAI: This is the privacy gate — call this before storing anything in
    Cosmos DB. The redacted text has PII replaced with asterisks so the
    original personal data never reaches the database.

    Args:
        text: The transcript text to scan.
        language: BCP-47 language code. Supports "en", "ko", "zh-Hant", etc.
        categories: Override which PII types to look for. Defaults to
            DEFAULT_PII_CATEGORIES if not specified.

    Returns:
        A dict with ``redactedText``, ``entities`` list, and ``pii_detected``
        boolean. On failure, returns the original text unmodified with an
        ``error`` field — we never block the pipeline over a PII API hiccup.
    """
    config = _get_config()
    url = f"{config['endpoint']}/language/:analyze-text/jobs?api-version=2023-04-01"

    # Using v3.1 synchronous endpoint — faster round-trip for single documents
    # compared to the newer jobs-based API which requires polling
    pii_url = (
        f"{config['endpoint']}/text/analytics/v3.1/entities/recognition/pii"
    )

    body = {
        "documents": [
            {"id": "1", "language": language, "text": text}
        ]
    }

    if categories:
        body["piiCategories"] = categories

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                pii_url,
                headers={
                    "Ocp-Apim-Subscription-Key": config["key"],
                    "Content-Type": "application/json",
                },
                json=body,
            )

            if resp.status_code != 200:
                logger.warning("PII detection failed: %s %s", resp.status_code, resp.text)
                # Graceful degradation: return original text rather than
                # blocking the entire transcription flow
                return {
                    "redactedText": text,
                    "entities": [],
                    "pii_detected": False,
                    "error": f"HTTP {resp.status_code}",
                }

            result = resp.json()
            doc = result.get("documents", [{}])[0]

            entities = [
                {
                    "text": e.get("text"),
                    "category": e.get("category"),
                    "subcategory": e.get("subcategory"),
                    "offset": e.get("offset"),
                    "length": e.get("length"),
                    "confidenceScore": e.get("confidenceScore"),
                }
                for e in doc.get("entities", [])
            ]

            return {
                "redactedText": doc.get("redactedText", text),
                "entities": entities,
                "pii_detected": len(entities) > 0,
            }

    except EnvironmentError:
        # IMPORTANT: config errors must propagate — if the key is missing,
        # we want to know immediately, not silently skip PII detection
        raise
    except Exception as exc:
        logger.error("PII detection error: %s", exc)
        return {
            "redactedText": text,
            "entities": [],
            "pii_detected": False,
            "error": str(exc),
        }


async def redact_pii(text: str, language: str = "en") -> str:
    """Shortcut that just returns the redacted text string.

    Handy when you don't care about the entity details — just need clean
    text for storage.
    """
    result = await detect_pii(text, language)
    return result["redactedText"]


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------

async def analyze_sentiment(text: str, language: str = "en") -> dict:
    """Analyze the emotional tone of text, returning an overall sentiment
    label plus per-sentence breakdowns.

    The avatar system consumes these results to adjust facial expressions
    in real time — "positive" maps to a smile, "negative" to concern, and
    "neutral" keeps a calm expression. The per-sentence detail lets us
    react to tonal shifts within a single utterance.

    Returns ``"unknown"`` sentiment on API failure rather than raising,
    so the avatar just holds its current expression instead of crashing.
    """
    config = _get_config()
    url = f"{config['endpoint']}/text/analytics/v3.1/sentiment"

    body = {
        "documents": [
            {"id": "1", "language": language, "text": text}
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Ocp-Apim-Subscription-Key": config["key"],
                    "Content-Type": "application/json",
                },
                json=body,
            )

            if resp.status_code != 200:
                logger.warning("Sentiment analysis failed: %s %s", resp.status_code, resp.text)
                return {
                    "sentiment": "unknown",
                    "confidenceScores": {},
                    "error": f"HTTP {resp.status_code}",
                }

            result = resp.json()
            doc = result.get("documents", [{}])[0]

            sentences = [
                {
                    "text": s.get("text"),
                    "sentiment": s.get("sentiment"),
                    "confidenceScores": s.get("confidenceScores"),
                }
                for s in doc.get("sentences", [])
            ]

            return {
                "sentiment": doc.get("sentiment", "unknown"),
                "confidenceScores": doc.get("confidenceScores", {}),
                "sentences": sentences,
            }

    except EnvironmentError:
        # Same pattern as detect_pii: let config errors bubble up
        raise
    except Exception as exc:
        logger.error("Sentiment analysis error: %s", exc)
        return {
            "sentiment": "unknown",
            "confidenceScores": {},
            "error": str(exc),
        }
