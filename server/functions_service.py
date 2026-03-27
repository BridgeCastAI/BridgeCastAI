"""
BridgeCast AI — Azure Functions Integration
Serverless orchestration: meeting end → trigger summarization, alerts, etc.
Uses httpx.AsyncClient to avoid blocking the FastAPI event loop.
"""

import json
import os
import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

FUNCTIONS_BASE_URL = os.getenv("AZURE_FUNCTIONS_URL", "")


async def trigger_meeting_summary(meeting_id: str, transcript: list) -> dict:
    """Trigger summary generation via Azure Functions, with local fallback."""
    if not FUNCTIONS_BASE_URL:
        logger.warning("AZURE_FUNCTIONS_URL not set, running locally")
        from openai_service import generate_meeting_notes
        return generate_meeting_notes(transcript)

    url = f"{FUNCTIONS_BASE_URL}/api/summarize-meeting"
    payload = {
        "meeting_id": meeting_id,
        "transcript": transcript,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.error("Failed to trigger meeting summary function: %s", e)
        from openai_service import generate_meeting_notes
        return generate_meeting_notes(transcript)


async def trigger_emergency_alert(meeting_id: str, text: str, speaker: str) -> dict:
    """Send emergency keyword detection request to Azure Functions."""
    if not FUNCTIONS_BASE_URL:
        logger.warning("AZURE_FUNCTIONS_URL not set, skipping alert function")
        return {"alert": False}

    url = f"{FUNCTIONS_BASE_URL}/api/emergency-alert"
    payload = {
        "meeting_id": meeting_id,
        "text": text,
        "speaker": speaker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.error("Failed to trigger emergency alert function: %s", e)
        return {"alert": False, "error": str(e)}


async def trigger_accessibility_report(meeting_id: str, metrics: dict) -> dict:
    """Request accessibility report from Azure Functions, with local fallback."""
    if not FUNCTIONS_BASE_URL:
        logger.warning("AZURE_FUNCTIONS_URL not set, generating report locally")
        return _generate_local_report(metrics)

    url = f"{FUNCTIONS_BASE_URL}/api/accessibility-report"
    payload = {
        "meeting_id": meeting_id,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.error("Failed to trigger accessibility report function: %s", e)
        return _generate_local_report(metrics)


def _generate_local_report(metrics: dict) -> dict:
    """Generate accessibility report locally as fallback.

    Returns a structured report with participation balance, bidirectional
    success rates, and average latency metrics. All values default to 0
    when source metrics are unavailable.
    """
    total_utterances = metrics.get("total_utterances", 0)
    sign_utterances = metrics.get("sign_utterances", 0)
    speech_utterances = metrics.get("speech_utterances", 0)

    return {
        "participation_balance": {
            "sign_users": sign_utterances,
            "speech_users": speech_utterances,
            "total": total_utterances,
            "balance_score": min(sign_utterances, speech_utterances) / max(total_utterances, 1),
        },
        "bidirectional_rate": {
            "sign_to_speech_success": metrics.get("sign_to_speech_success", 0),
            "speech_to_text_success": metrics.get("speech_to_text_success", 0),
        },
        "avg_latency_ms": {
            "sign_recognition": metrics.get("avg_sign_latency_ms", 0),
            "stt": metrics.get("avg_stt_latency_ms", 0),
            "tts": metrics.get("avg_tts_latency_ms", 0),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local_fallback",
    }


# --- Azure Functions deployment helpers ---

FUNCTION_APP_CODE = {
    "summarize-meeting": '''
import azure.functions as func
import json
import os
from openai import AzureOpenAI

def main(req: func.HttpRequest) -> func.HttpResponse:
    body = req.get_json()
    transcript = body.get("transcript", [])

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2024-02-15-preview",
    )

    transcript_text = "\\n".join(
        f"[{e.get('timestamp','')}] {e.get('speaker','Unknown')}: {e.get('text','')}"
        for e in transcript
    )

    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        messages=[
            {"role": "system", "content": "Generate structured meeting notes in JSON."},
            {"role": "user", "content": f"Transcript:\\n{transcript_text}"},
        ],
        response_format={"type": "json_object"},
    )

    return func.HttpResponse(response.choices[0].message.content, mimetype="application/json")
''',
    "emergency-alert": '''
import azure.functions as func
import json

EMERGENCY_KEYWORDS = ["fire", "earthquake", "evacuate", "emergency", "help", "danger", "911"]

def main(req: func.HttpRequest) -> func.HttpResponse:
    body = req.get_json()
    text = body.get("text", "").lower()

    is_emergency = any(kw in text for kw in EMERGENCY_KEYWORDS)

    return func.HttpResponse(
        json.dumps({"alert": is_emergency, "text": text}),
        mimetype="application/json"
    )
''',
}


def get_function_deployment_code(function_name: str) -> str:
    """Return the Azure Function source for the given function name."""
    return FUNCTION_APP_CODE.get(function_name, "")
