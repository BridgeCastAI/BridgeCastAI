"""
BridgeCast AI — Azure Content Safety Service

Every user message passes through this service before entering the
transcript or being broadcast. The meeting UI shows a "Safe" badge on
each checked message for RAI transparency.

Uses Azure Content Safety REST API with FourSeverityLevels output.
Severity scale: 0 (safe) → 6 (severe).
Categories: Hate, SelfHarm, Sexual, Violence.
"""

import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Azure recommends threshold=4 for production — catches clearly harmful
# content without triggering too many false positives during conversation.
DEFAULT_THRESHOLD = 4

_API_VERSION = "2024-09-01"

# Azure returns PascalCase category names; we normalize to snake_case
# so the rest of our Python code stays consistent.
_CATEGORY_MAP = {
    "Hate": "hate",
    "SelfHarm": "self_harm",
    "Sexual": "sexual",
    "Violence": "violence",
}
_ALL_CATEGORIES = tuple(_CATEGORY_MAP.values())


def _get_config() -> dict:
    """Load Content Safety credentials from environment."""
    endpoint = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT")
    key = os.environ.get("AZURE_CONTENT_SAFETY_KEY")
    if not endpoint or not key:
        raise EnvironmentError(
            "AZURE_CONTENT_SAFETY_ENDPOINT and AZURE_CONTENT_SAFETY_KEY must be set."
        )
    return {"endpoint": endpoint.rstrip("/"), "key": key}


async def check_text_safety(text: str, threshold: Optional[int] = None) -> dict:
    """Check text for harmful content via Azure Content Safety.

    Returns a dict with:
      - safe (bool): True if nothing exceeded the threshold
      - categories: per-category severity scores
      - flagged_categories: list of category names that exceeded threshold

    Empty/whitespace text short-circuits to safe — saves an API round-trip.
    """
    if not text or not text.strip():
        return {
            "safe": True,
            "categories": {cat: {"severity": 0} for cat in _ALL_CATEGORIES},
            "flagged_categories": [],
        }

    threshold = threshold or DEFAULT_THRESHOLD
    config = _get_config()

    # FourSeverityLevels returns 0/2/4/6 — enough granularity for
    # real-time moderation without over-blocking normal conversation.
    body = {
        "text": text,
        "categories": list(_CATEGORY_MAP.keys()),
        "outputType": "FourSeverityLevels",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{config['endpoint']}/contentsafety/text:analyze",
            params={"api-version": _API_VERSION},
            headers={
                "Ocp-Apim-Subscription-Key": config["key"],
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()

    result = resp.json()

    categories = {}
    flagged = []
    for item in result.get("categoriesAnalysis", []):
        cat = _CATEGORY_MAP.get(item["category"], item["category"])
        sev = item.get("severity", 0)
        categories[cat] = {"severity": sev}
        if sev >= threshold:
            flagged.append(cat)

    # Backfill any category the API omitted (happens when severity is 0)
    for cat in _ALL_CATEGORIES:
        categories.setdefault(cat, {"severity": 0})

    is_safe = len(flagged) == 0

    # Truncate logged text to protect user privacy
    logger.info("safety: safe=%s flagged=%s text='%s...'", is_safe, flagged, text[:60])

    return {
        "safe": is_safe,
        "categories": categories,
        "flagged_categories": flagged,
    }
