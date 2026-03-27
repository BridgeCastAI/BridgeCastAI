"""
BridgeCast AI — Azure SignalR Service (Service 14/15)

Real-time bidirectional messaging layer for live meeting sessions.
Integrates with Azure SignalR Service in **Serverless mode**, which means
there's no persistent server-side hub — we talk to SignalR entirely through
its REST API and hand out self-signed JWTs so clients can connect directly.

This module is used by meeting_api.py to push transcriptions, sign-language
recognition results, safety badges, and emergency alerts to connected clients
in real time. Cosmos DB stores the data; SignalR delivers it instantly.

Why raw REST instead of the Azure SDK?
    The official Python SDK doesn't expose a convenient REST client for
    serverless mode. Parsing the connection string and hitting the REST API
    ourselves is actually simpler and gives us full control over token TTL,
    audience claims, etc.
"""

import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_config() -> dict:
    """Parse the SignalR connection string into endpoint + access key.

    Azure connection strings follow the format:
        Endpoint=https://xxx.service.signalr.net;AccessKey=base64...;Version=1.0;

    We split on semicolons ourselves because the SDK doesn't do this for
    serverless REST calls.
    """
    connection_string = os.environ.get("AZURE_SIGNALR_CONNECTION_STRING")
    if not connection_string:
        raise EnvironmentError(
            "AZURE_SIGNALR_CONNECTION_STRING must be set."
        )

    # NOTE: partition("=") only splits on the FIRST "=", which matters
    # because the base64 AccessKey itself contains "=" padding chars
    parts = {}
    for segment in connection_string.split(";"):
        segment = segment.strip()
        if "=" in segment:
            key, _, value = segment.partition("=")
            parts[key.strip()] = value.strip()

    endpoint = parts.get("Endpoint", "")
    access_key = parts.get("AccessKey", "")

    if not endpoint or not access_key:
        raise EnvironmentError(
            "AZURE_SIGNALR_CONNECTION_STRING must contain Endpoint and AccessKey."
        )

    return {"endpoint": endpoint.rstrip("/"), "access_key": access_key}


def _generate_token(endpoint: str, access_key: str, hub: str, user_id: Optional[str] = None, ttl: int = 3600) -> str:
    """Mint a short-lived JWT for SignalR client negotiation.

    SignalR Serverless mode requires self-signed JWTs with a very specific
    audience format: ``{endpoint}/client/?hub={hub}``. The service validates
    this exact string, so even a trailing slash mismatch will cause 401s.

    We build the JWT manually (header.payload.signature) rather than pulling
    in PyJWT — fewer dependencies for a straightforward HS256 token.
    """
    import time
    import hmac
    import hashlib
    import base64
    import json as _json

    # IMPORTANT: audience must match this exact format or SignalR rejects the token
    audience = f"{endpoint}/client/?hub={hub}"
    now = int(time.time())
    exp = now + ttl

    header = base64.urlsafe_b64encode(
        _json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()

    payload_data = {"aud": audience, "iat": now, "exp": exp}
    if user_id:
        payload_data["sub"] = user_id

    payload = base64.urlsafe_b64encode(
        _json.dumps(payload_data).encode()
    ).rstrip(b"=").decode()

    signature_input = f"{header}.{payload}"
    sig = hmac.new(
        access_key.encode(), signature_input.encode(), hashlib.sha256
    ).digest()
    signature = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()

    return f"{header}.{payload}.{signature}"


# ---------------------------------------------------------------------------
# Hub name — all BridgeCast connections share a single hub.
# Groups within the hub separate individual meetings.
# ---------------------------------------------------------------------------

HUB_NAME = "bridgecast"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def negotiate(user_id: Optional[str] = None) -> dict:
    """Build the negotiation payload that clients need to open a WebSocket.

    Returns a dict with ``url`` (wss:// endpoint) and ``accessToken`` (JWT).
    The client JS SDK calls this once on page load, then upgrades to WS.
    """
    config = _get_config()
    token = _generate_token(
        config["endpoint"], config["access_key"], HUB_NAME, user_id
    )
    ws_url = config["endpoint"].replace("https://", "wss://").replace("http://", "ws://")

    return {
        "url": f"{ws_url}/client/?hub={HUB_NAME}",
        "accessToken": token,
    }


async def send_to_all(message: dict, hub: Optional[str] = None) -> dict:
    """Broadcast to every connected client across the entire hub.

    This is the nuclear option — use it for things like emergency safety
    alerts that genuinely need to reach everyone, not for per-meeting data.
    For meeting-specific messages (transcriptions, sign recognition),
    use send_to_group() instead.
    """
    config = _get_config()
    hub = hub or HUB_NAME
    token = _generate_token(config["endpoint"], config["access_key"], hub)

    url = f"{config['endpoint']}/api/v1/hubs/{hub}/:send?api-version=2024-01-01"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"target": "newMessage", "arguments": [message]},
            )

            # NOTE: SignalR returns 202 Accepted (not 200) because delivery
            # is async — the service queues the message for fanout
            if resp.status_code in (200, 202):
                logger.info("SignalR broadcast sent to hub '%s'", hub)
                return {"success": True, "detail": "Message broadcast to all clients"}
            else:
                logger.warning("SignalR broadcast failed: %s %s", resp.status_code, resp.text)
                return {"success": False, "detail": f"HTTP {resp.status_code}: {resp.text}"}

    except Exception as exc:
        logger.error("SignalR send_to_all error: %s", exc)
        return {"success": False, "detail": str(exc)}


async def send_to_user(user_id: str, message: dict, hub: Optional[str] = None) -> dict:
    """Send a message to a specific user by their SignalR user ID.

    Useful for direct notifications — e.g., telling a participant their
    safety badge was updated or their avatar config changed.
    """
    config = _get_config()
    hub = hub or HUB_NAME
    token = _generate_token(config["endpoint"], config["access_key"], hub)

    url = f"{config['endpoint']}/api/v1/hubs/{hub}/users/{user_id}/:send?api-version=2024-01-01"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"target": "newMessage", "arguments": [message]},
            )

            # 202 = queued for delivery, same as send_to_all
            if resp.status_code in (200, 202):
                return {"success": True, "detail": f"Message sent to user {user_id}"}
            else:
                return {"success": False, "detail": f"HTTP {resp.status_code}: {resp.text}"}

    except Exception as exc:
        logger.error("SignalR send_to_user error: %s", exc)
        return {"success": False, "detail": str(exc)}


async def send_to_group(group: str, message: dict, hub: Optional[str] = None) -> dict:
    """Send a message to everyone in a specific group (= one meeting room).

    This is the workhorse for live meetings. Each meeting room ID maps to a
    SignalR group, so transcription results, sign-language recognition, and
    sentiment updates only go to the people actually in that meeting.
    """
    config = _get_config()
    hub = hub or HUB_NAME
    token = _generate_token(config["endpoint"], config["access_key"], hub)

    url = f"{config['endpoint']}/api/v1/hubs/{hub}/groups/{group}/:send?api-version=2024-01-01"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"target": "newMessage", "arguments": [message]},
            )

            if resp.status_code in (200, 202):
                return {"success": True, "detail": f"Message sent to group {group}"}
            else:
                return {"success": False, "detail": f"HTTP {resp.status_code}: {resp.text}"}

    except Exception as exc:
        logger.error("SignalR send_to_group error: %s", exc)
        return {"success": False, "detail": str(exc)}
