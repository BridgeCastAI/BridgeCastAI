"""
BridgeCast AI — Azure Communication Services (ACS) integration.
Service #4 of 15.

Handles video meeting room lifecycle (create/join/list) and participant
token issuance.  The Rooms API is what lets us spin up isolated,
short-lived meeting spaces — each room gets its own roster, its own
permissions, and its own expiry window.

Depends on:
    - Azure Communication Services resource (connection string in env)
    - Cosmos DB service (upstream — meeting metadata is persisted there)
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from azure.communication.rooms import (
    RoomsClient,
    RoomParticipant,
    ParticipantRole,
)
from azure.communication.identity import (
    CommunicationIdentityClient,
    CommunicationUserIdentifier,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_connection_string() -> str:
    """Pull the ACS connection string from env, or fail loudly."""
    conn = os.environ.get("AZURE_COMMUNICATION_CONNECTION_STRING")
    if not conn:
        raise EnvironmentError(
            "AZURE_COMMUNICATION_CONNECTION_STRING must be set."
        )
    return conn


def _get_rooms_client() -> RoomsClient:
    """Instantiate a Rooms client from the shared connection string."""
    return RoomsClient.from_connection_string(_get_connection_string())


def _get_identity_client() -> CommunicationIdentityClient:
    """Instantiate an Identity client for user/token management."""
    return CommunicationIdentityClient.from_connection_string(
        _get_connection_string()
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_meeting_room(
    valid_from: Optional[datetime] = None,
    valid_until: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Spin up a new ACS Room with a bounded validity window.

    Args:
        valid_from:  When the room becomes joinable.  Defaults to now.
        valid_until: When the room auto-expires.  Defaults to 1 hour out.

    Returns:
        Dict with room_id, created_at, valid_from, valid_until.
    """
    now = datetime.now(timezone.utc)
    valid_from = valid_from or now
    # NOTE: 1-hour default is a sweet spot — long enough for a real meeting,
    # short enough that forgotten rooms don't linger during demos.
    valid_until = valid_until or (now + timedelta(hours=1))

    rooms_client = _get_rooms_client()
    room = rooms_client.create_room(
        valid_from=valid_from,
        valid_until=valid_until,
    )

    logger.info("ACS room created: %s", room.id)

    return {
        "room_id": room.id,
        "created_at": now.isoformat(),
        "valid_from": valid_from.isoformat(),
        "valid_until": valid_until.isoformat(),
    }


async def get_room_token(
    room_id: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create (or reuse) an ACS identity and return a VOIP token for the room.

    If no user_id is supplied we mint a fresh identity — handy for
    guest/anonymous join scenarios.

    Args:
        room_id: The ACS room to join.
        user_id: Existing ACS user ID, or None to create one on the fly.

    Returns:
        Dict with user_id, token, expires_on, room_id.
    """
    identity_client = _get_identity_client()

    if user_id:
        user = CommunicationUserIdentifier(user_id)
    else:
        user = identity_client.create_user()

    # IMPORTANT: The "voip" scope is mandatory for Rooms API participation —
    # it grants the audio/video capabilities the room expects.
    token_response = identity_client.get_token(user, scopes=["voip"])

    # Add to the room as ATTENDEE.  We don't differentiate presenter vs
    # consumer because BridgeCast treats every participant equally —
    # everyone can speak, hear, and see sign-language avatars.
    rooms_client = _get_rooms_client()
    participant = RoomParticipant(
        communication_identifier=user,
        role=ParticipantRole.ATTENDEE,
    )

    try:
        rooms_client.add_or_update_participants(
            room_id=room_id, participants=[participant]
        )
    except Exception as exc:
        logger.warning(
            "Could not add participant to room %s: %s", room_id, exc
        )

    # NOTE: Defensive extraction — the ACS SDK returns different object
    # shapes depending on whether the user was freshly created or
    # reconstituted from an existing ID string.  .properties["id"] is the
    # canonical path, but we fall back to str() just in case.
    user_id_str = user.properties.get("id", str(user)) if hasattr(user, "properties") else str(user)

    logger.info("Token issued for user=%s in room=%s", user_id_str, room_id)

    return {
        "user_id": user_id_str,
        "token": token_response.token,
        "expires_on": token_response.expires_on.isoformat(),
        "room_id": room_id,
    }


async def list_participants(room_id: str) -> List[Dict[str, Any]]:
    """Return every participant currently rostered in the given room.

    Returns:
        List of dicts, each with user_id and role.
    """
    rooms_client = _get_rooms_client()
    participants = rooms_client.list_participants(room_id=room_id)

    result = []
    for p in participants:
        # Same defensive id extraction as get_room_token — see note there.
        uid = (
            p.communication_identifier.properties.get("id", "unknown")
            if hasattr(p.communication_identifier, "properties")
            else str(p.communication_identifier)
        )
        result.append({
            "user_id": uid,
            "role": str(p.role),
        })

    logger.info("Room %s has %d participants", room_id, len(result))
    return result
