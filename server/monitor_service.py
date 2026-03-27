"""
BridgeCast AI — Azure Monitor / Application Insights integration (service 8 of 15).

Provides telemetry tracking for sign-language recognition, STT, and meeting
analytics. Talks to Application Insights via the OpenCensus Azure exporter
(the recommended Python SDK — the older `applicationinsights` package is
deprecated and doesn't support connection strings).

The in-memory event store here is intentionally simple: good enough for dev
and demo, but production would swap it out for queries against Log Analytics
via its REST API.

Relationships:
    - meeting_api.py calls track_sign_recognition / track_stt_recognition
      on every inference result
    - The /analytics endpoint in meeting_api.py delegates to
      get_meeting_analytics() below
    - Bicep module #12 (monitoring.bicep) provisions the App Insights
      resource this service connects to
"""

import os
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telemetry client (lazy singleton)
# ---------------------------------------------------------------------------

_exporter = None
_tracer = None


def _get_exporter():
    """Lazily create the OpenCensus metrics exporter.

    We defer the import and init until first use so the rest of the server
    can start up even if App Insights isn't configured yet (e.g. local dev).
    """
    global _exporter
    if _exporter is not None:
        return _exporter

    conn_str = os.environ.get("AZURE_APPINSIGHTS_CONNECTION_STRING")
    if not conn_str:
        raise EnvironmentError(
            "AZURE_APPINSIGHTS_CONNECTION_STRING must be set."
        )

    from opencensus.ext.azure import metrics_exporter
    _exporter = metrics_exporter.new_metrics_exporter(
        connection_string=conn_str,
    )
    logger.info("Application Insights metrics exporter initialised.")
    return _exporter


def _get_azure_logger() -> logging.Logger:
    """Return a Python logger wired to Application Insights for custom events.

    Uses a named logger ("bridgecast.telemetry") so we only attach the
    AzureLogHandler once, no matter how many times this is called.
    """
    conn_str = os.environ.get("AZURE_APPINSIGHTS_CONNECTION_STRING")
    if not conn_str:
        raise EnvironmentError(
            "AZURE_APPINSIGHTS_CONNECTION_STRING must be set."
        )

    from opencensus.ext.azure.log_exporter import AzureLogHandler

    ai_logger = logging.getLogger("bridgecast.telemetry")
    # NOTE: only add the handler once — duplicate handlers = duplicate telemetry bills
    if not ai_logger.handlers:
        handler = AzureLogHandler(connection_string=conn_str)
        ai_logger.addHandler(handler)
        ai_logger.setLevel(logging.INFO)

    return ai_logger


# ---------------------------------------------------------------------------
# In-memory analytics store (bounded ring buffer)
# ---------------------------------------------------------------------------

# ~200 bytes/event * 10K ≈ 2 MB — keeps memory predictable for long sessions
_MAX_EVENT_STORE_SIZE = 10_000
_event_store: List[Dict[str, Any]] = []


def _append_event(event: Dict[str, Any]) -> None:
    """Add an event to the ring buffer.

    When the buffer is full we trim the oldest 20% in one shot rather than
    evicting one-by-one on each append. This gives us amortized O(1) inserts
    instead of O(n) list shifts on every call.
    """
    _event_store.append(event)
    if len(_event_store) > _MAX_EVENT_STORE_SIZE:
        trim_count = _MAX_EVENT_STORE_SIZE // 5
        del _event_store[:trim_count]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def track_event(name: str, properties: Optional[Dict[str, Any]] = None) -> None:
    """Send a custom event to Application Insights.

    Args:
        name: Event name — e.g. "MeetingStarted", "SignRecognized".
        properties: Arbitrary key-value pairs attached to the event.
    """
    properties = properties or {}
    properties["event_name"] = name
    properties["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Always persist locally so analytics queries work even without App Insights
    _append_event({"name": name, **properties})

    try:
        ai_logger = _get_azure_logger()
        ai_logger.info(name, extra={"custom_dimensions": properties})
        logger.debug("Tracked event: %s %s", name, properties)
    except EnvironmentError:
        # Graceful degradation: local-only mode when App Insights isn't set up
        logger.warning("App Insights not configured — event logged locally only: %s", name)
    except Exception as exc:
        logger.warning("Failed to send event to App Insights: %s", exc)


def track_metric(name: str, value: float) -> None:
    """Send a numeric metric to Application Insights.

    Args:
        name: Metric name — e.g. "stt_latency_ms", "sign_confidence".
        value: The measured value.
    """
    properties = {
        "metric_name": name,
        "value": value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_event({"name": f"metric:{name}", **properties})

    try:
        ai_logger = _get_azure_logger()
        ai_logger.info(
            "Metric: %s = %s",
            name,
            value,
            extra={"custom_dimensions": {"metric_name": name, "value": str(value)}},
        )
        logger.debug("Tracked metric: %s = %s", name, value)
    except EnvironmentError:
        logger.warning("App Insights not configured — metric logged locally only: %s", name)
    except Exception as exc:
        logger.warning("Failed to send metric to App Insights: %s", exc)


def track_sign_recognition(
    sign_text: str,
    confidence: float,
    latency_ms: float,
) -> None:
    """Domain helper: emit one event + two metrics for a sign recognition result.

    Bundles the event and its associated latency/confidence metrics into a
    single call so callers don't have to remember to track all three.
    """
    track_event("SignRecognition", {
        "sign_text": sign_text,
        "confidence": confidence,
        "latency_ms": latency_ms,
    })
    track_metric("sign_recognition_latency_ms", latency_ms)
    track_metric("sign_recognition_confidence", confidence)


def track_stt_recognition(text: str, latency_ms: float) -> None:
    """Domain helper: emit one event + one metric for an STT result.

    Truncates the transcript to 200 chars to avoid bloating telemetry payloads.
    """
    track_event("STTRecognition", {
        "text": text[:200],
        "latency_ms": latency_ms,
    })
    track_metric("stt_latency_ms", latency_ms)


def get_meeting_analytics(meeting_id: str) -> Dict[str, Any]:
    """Aggregate participation and latency stats for a given meeting.

    Scans the in-memory event store — fast and simple for demos.
    TODO: In production, query Log Analytics workspace via REST so we
    aren't limited to events from this process's memory.
    """
    meeting_events = [
        e for e in _event_store
        if e.get("meeting_id") == meeting_id
    ]

    sign_events = [e for e in meeting_events if e.get("name") == "SignRecognition"]
    stt_events = [e for e in meeting_events if e.get("name") == "STTRecognition"]

    avg_sign_latency = 0.0
    if sign_events:
        avg_sign_latency = sum(e.get("latency_ms", 0) for e in sign_events) / len(sign_events)

    avg_stt_latency = 0.0
    if stt_events:
        avg_stt_latency = sum(e.get("latency_ms", 0) for e in stt_events) / len(stt_events)

    return {
        "meeting_id": meeting_id,
        "total_events": len(meeting_events),
        "sign_recognitions": len(sign_events),
        "stt_recognitions": len(stt_events),
        "avg_sign_latency_ms": round(avg_sign_latency, 2),
        "avg_stt_latency_ms": round(avg_stt_latency, 2),
    }
