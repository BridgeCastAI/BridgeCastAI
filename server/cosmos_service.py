"""
BridgeCast AI — Azure Cosmos DB persistence layer.
Service #5 of 15.

Stores meeting records (transcripts, notes, metadata) and per-user
preferences in Cosmos DB's NoSQL API.  Two containers live under a single
database:

    bridgecast-db/
        meetings   — one doc per meeting, partitioned by user_id
        users      — preferences doc per user, also partitioned by user_id

The /user_id partition key is deliberate: most queries are scoped to a
single user ("show me MY meetings"), so we get cheap single-partition
reads instead of expensive cross-partition fan-out.

Relies on:
    - Azure Cosmos DB account (endpoint + key in env)
    - Called by meeting_api.py for CRUD operations
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from azure.cosmos import CosmosClient as AzureCosmosClient
from azure.cosmos import PartitionKey, exceptions as cosmos_exceptions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database / container names
# ---------------------------------------------------------------------------
DATABASE_NAME = "bridgecast-db"
MEETINGS_CONTAINER = "meetings"
USERS_CONTAINER = "users"


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------

class CosmosService:
    """Wraps the Cosmos DB SDK with lazy init and high-level CRUD helpers.

    DB and container references are created on first actual use (not at
    import time) so that importing this module doesn't open a connection
    to Azure — important because not every API route touches Cosmos.
    """

    def __init__(self):
        endpoint = os.environ.get("AZURE_COSMOS_ENDPOINT")
        key = os.environ.get("AZURE_COSMOS_KEY")

        if not endpoint or not key:
            raise EnvironmentError(
                "AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY must be set."
            )

        self._client = AzureCosmosClient(endpoint, credential=key)
        # Lazy — populated by _ensure_db() on first call
        self._db = None
        self._meetings = None
        self._users = None

    # --- Lazy initialisation ------------------------------------------------

    def _ensure_db(self):
        """Provision the database and containers if they don't exist yet.

        Uses create_*_if_not_exists so this is safe to call repeatedly —
        after the first run it's essentially a no-op dict lookup.
        """
        if self._db is not None:
            return

        self._db = self._client.create_database_if_not_exists(id=DATABASE_NAME)

        # NOTE: 400 RU/s is the minimum provisioned throughput on Cosmos.
        # More than enough for hackathon demo traffic; easy to scale later.
        self._meetings = self._db.create_container_if_not_exists(
            id=MEETINGS_CONTAINER,
            partition_key=PartitionKey(path="/user_id"),
            offer_throughput=400,
        )

        self._users = self._db.create_container_if_not_exists(
            id=USERS_CONTAINER,
            partition_key=PartitionKey(path="/user_id"),
            offer_throughput=400,
        )

        logger.info("Cosmos DB initialised: database=%s", DATABASE_NAME)

    # --- Meeting CRUD -------------------------------------------------------

    def save_meeting(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a meeting record (transcript, notes, etc.).

        Args:
            meeting_data: Must contain 'user_id' (partition key).
                          'id' and 'created_at' are auto-generated if absent.

        Returns:
            The upserted document as returned by Cosmos.
        """
        self._ensure_db()

        if "user_id" not in meeting_data:
            raise ValueError("meeting_data must contain 'user_id'.")

        # uuid4().hex gives us a dash-free string — shorter and URL-friendly
        # compared to the standard 8-4-4-4-12 format.
        meeting_data.setdefault("id", uuid.uuid4().hex)
        meeting_data.setdefault(
            "created_at", datetime.now(timezone.utc).isoformat()
        )
        meeting_data.setdefault("type", "meeting")

        result = self._meetings.upsert_item(meeting_data)
        logger.info("Meeting saved: id=%s, user=%s", result["id"], result["user_id"])
        return result

    def get_meeting(self, meeting_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Point-read a single meeting document.

        Args:
            meeting_id: Document id.
            user_id: Partition key — Cosmos needs this for an efficient
                     direct lookup instead of a cross-partition scan.

        Returns:
            The meeting dict, or None if it doesn't exist.
        """
        self._ensure_db()

        try:
            item = self._meetings.read_item(item=meeting_id, partition_key=user_id)
            return item
        except cosmos_exceptions.CosmosResourceNotFoundError:
            logger.warning("Meeting not found: id=%s, user=%s", meeting_id, user_id)
            return None

    def list_meetings(self, user_id: str) -> List[Dict[str, Any]]:
        """Fetch all meetings for a user, newest first.

        Because the partition key is /user_id, this stays within a single
        partition — no cross-partition fan-out, no extra RU cost.
        """
        self._ensure_db()

        query = (
            "SELECT * FROM c WHERE c.user_id = @uid ORDER BY c.created_at DESC"
        )
        params = [{"name": "@uid", "value": user_id}]

        items = list(
            self._meetings.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=False,
            )
        )
        logger.info("Listed %d meetings for user=%s", len(items), user_id)
        return items

    def delete_meeting(self, meeting_id: str, user_id: str) -> None:
        """Remove a meeting document. Silently succeeds if already gone."""
        self._ensure_db()
        try:
            self._meetings.delete_item(item=meeting_id, partition_key=user_id)
            logger.info("Meeting deleted: id=%s, user=%s", meeting_id, user_id)
        except cosmos_exceptions.CosmosResourceNotFoundError:
            logger.warning("Meeting not found for delete: id=%s", meeting_id)

    # --- User preferences ---------------------------------------------------

    def save_user_preferences(
        self, user_id: str, prefs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upsert user preferences (sign language, captions, theme, etc.).

        Args:
            user_id: The user whose preferences to save.
            prefs: Arbitrary preference fields.  Common ones include
                   preferred_sign_language, caption_size, caption_language,
                   and theme.
        """
        self._ensure_db()

        doc = {
            "id": f"prefs-{user_id}",
            "user_id": user_id,
            "type": "preferences",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **prefs,
        }

        result = self._users.upsert_item(doc)
        logger.info("User preferences saved: user=%s", user_id)
        return result

    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load saved preferences for a user, or None if they haven't set any."""
        self._ensure_db()

        try:
            item = self._users.read_item(
                item=f"prefs-{user_id}", partition_key=user_id
            )
            return item
        except cosmos_exceptions.CosmosResourceNotFoundError:
            logger.info("No preferences found for user=%s", user_id)
            return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[CosmosService] = None


def get_cosmos_service() -> CosmosService:
    """Return the shared CosmosService instance (created on first call).

    One client per process is Azure best practice — it lets the SDK
    reuse TCP connections and connection pools across requests.
    """
    global _instance
    if _instance is None:
        _instance = CosmosService()
    return _instance
