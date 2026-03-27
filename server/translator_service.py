"""
BridgeCast AI — Azure Translator Service
Provides text translation and language detection using the Azure Translator REST API.
"""

import os
import logging
import uuid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Supported language codes
SUPPORTED_LANGUAGES = {"en", "ko", "zh-Hant", "ja"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_config() -> dict:
    """Return Translator API configuration from environment variables."""
    key = os.environ.get("AZURE_TRANSLATOR_KEY")
    endpoint = os.environ.get("AZURE_TRANSLATOR_ENDPOINT")
    region = os.environ.get("AZURE_TRANSLATOR_REGION", "eastus")

    if not key or not endpoint:
        raise EnvironmentError(
            "AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_ENDPOINT must be set."
        )

    return {"key": key, "endpoint": endpoint.rstrip("/"), "region": region}


def _build_headers(config: dict) -> dict:
    """Build the standard headers for Azure Translator requests."""
    return {
        "Ocp-Apim-Subscription-Key": config["key"],
        "Ocp-Apim-Subscription-Region": config["region"],
        "Content-Type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def translate_text(
    text: str,
    from_lang: Optional[str],
    to_lang: str,
) -> dict:
    """Translate *text* from one language to another.

    Parameters
    ----------
    text : str
        The text to translate.
    from_lang : str or None
        Source language code (e.g. "en"). Pass None for auto-detection.
    to_lang : str
        Target language code (e.g. "ko").

    Returns
    -------
    dict
        {
            "translated_text": "...",
            "detected_language": "en" | None,
            "from": "en",
            "to": "ko",
        }
    """
    if to_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported target language '{to_lang}'. "
            f"Choose from {SUPPORTED_LANGUAGES}"
        )

    if from_lang and from_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported source language '{from_lang}'. "
            f"Choose from {SUPPORTED_LANGUAGES}"
        )

    config = _get_config()
    url = f"{config['endpoint']}/translate"
    params = {"api-version": "3.0", "to": to_lang}
    if from_lang:
        params["from"] = from_lang

    body = [{"Text": text}]

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            url,
            params=params,
            headers=_build_headers(config),
            json=body,
        )
        response.raise_for_status()

    result = response.json()[0]
    translation = result["translations"][0]
    detected = result.get("detectedLanguage", {}).get("language")

    logger.info(
        "Translated [%s -> %s]: '%s' -> '%s'",
        from_lang or detected or "auto",
        to_lang,
        text[:60],
        translation["text"][:60],
    )

    return {
        "translated_text": translation["text"],
        "detected_language": detected,
        "from": from_lang or detected,
        "to": to_lang,
    }


async def detect_language(text: str) -> dict:
    """Detect the language of the given text.

    Returns
    -------
    dict
        {
            "language": "en",
            "confidence": 0.98,
            "alternatives": [...]
        }
    """
    config = _get_config()
    url = f"{config['endpoint']}/detect"
    params = {"api-version": "3.0"}
    body = [{"Text": text}]

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            url,
            params=params,
            headers=_build_headers(config),
            json=body,
        )
        response.raise_for_status()

    result = response.json()[0]

    logger.info(
        "Detected language: %s (score=%.2f) for text: '%s'",
        result["language"],
        result["score"],
        text[:60],
    )

    return {
        "language": result["language"],
        "confidence": result["score"],
        "alternatives": result.get("alternatives", []),
    }
