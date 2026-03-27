"""
BridgeCast AI — Azure Blob Storage Service

Stores meeting recordings (webm), exported PDF notes, and sign language
reference clips. Each meeting's files live under a {meeting_id}/ prefix
for easy bulk listing and cleanup.

Containers:
  - meeting-recordings: video recordings from ACS
  - meeting-notes: exported PDF transcripts
  - sign-clips: captured sign language clips for model training
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
    BlobSasPermissions,
)

logger = logging.getLogger(__name__)

_blob_client: Optional[BlobServiceClient] = None

CONTAINER_MEETINGS = "meeting-recordings"
CONTAINER_NOTES = "meeting-notes"
CONTAINER_SIGNS = "sign-clips"


def _get_client() -> BlobServiceClient:
    """Lazy singleton — creates the BlobServiceClient once and reuses it."""
    global _blob_client
    if _blob_client is None:
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING must be set.")
        _blob_client = BlobServiceClient.from_connection_string(conn_str)
    return _blob_client


def _ensure_container(name: str) -> None:
    """Create a container if it doesn't exist yet.
    The 409 Conflict from Azure is expected when it already exists."""
    try:
        _get_client().create_container(name)
        logger.info("Created blob container: %s", name)
    except Exception:
        pass  # container already exists — expected on every call after first


def _blob_path(meeting_id: str, suffix: str) -> str:
    """Build a timestamped blob name under the meeting prefix.
    Format: {meeting_id}/20260328_143012_{suffix}"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{meeting_id}/{ts}_{suffix}"


# ---------------------------------------------------------------------------
# Upload operations
# ---------------------------------------------------------------------------

async def upload_meeting_recording(
    meeting_id: str, video_bytes: bytes, content_type: str = "video/webm",
) -> dict:
    """Upload a meeting recording. Returns the blob URL + a 24h SAS link."""
    _ensure_container(CONTAINER_MEETINGS)
    blob_name = _blob_path(meeting_id, "recording.webm")
    blob = _get_client().get_blob_client(CONTAINER_MEETINGS, blob_name)

    # overwrite=True so re-uploads don't fail if the user retries
    blob.upload_blob(
        video_bytes, overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )

    logger.info("Uploaded recording: %s (%d bytes)", blob_name, len(video_bytes))
    return {
        "blob_name": blob_name,
        "url": blob.url,
        "download_url": _generate_sas_url(CONTAINER_MEETINGS, blob_name, hours=24),
        "size_bytes": len(video_bytes),
    }


async def upload_meeting_pdf(
    meeting_id: str, pdf_bytes: bytes, filename: str = "meeting_notes.pdf",
) -> dict:
    """Upload an exported PDF. SAS link valid for 72h (longer because
    users often share these after the meeting)."""
    _ensure_container(CONTAINER_NOTES)
    blob_name = _blob_path(meeting_id, filename)
    blob = _get_client().get_blob_client(CONTAINER_NOTES, blob_name)

    blob.upload_blob(
        pdf_bytes, overwrite=True,
        content_settings=ContentSettings(content_type="application/pdf"),
    )

    logger.info("Uploaded PDF: %s (%d bytes)", blob_name, len(pdf_bytes))
    return {
        "blob_name": blob_name,
        "url": blob.url,
        "share_url": _generate_sas_url(CONTAINER_NOTES, blob_name, hours=72),
        "size_bytes": len(pdf_bytes),
    }


async def upload_sign_clip(
    meeting_id: str, clip_bytes: bytes, sign_label: str,
    content_type: str = "video/webm",
) -> dict:
    """Upload a short sign language clip, tagged with its gloss label.
    These clips can be used later to fine-tune recognition models."""
    _ensure_container(CONTAINER_SIGNS)
    blob_name = _blob_path(meeting_id, f"{sign_label}.webm")
    blob = _get_client().get_blob_client(CONTAINER_SIGNS, blob_name)

    blob.upload_blob(
        clip_bytes, overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )

    logger.info("Uploaded sign clip: %s (%s)", blob_name, sign_label)
    return {"blob_name": blob_name, "url": blob.url, "sign_label": sign_label}


# ---------------------------------------------------------------------------
# List / Delete
# ---------------------------------------------------------------------------

async def list_meeting_files(meeting_id: str) -> dict:
    """List all blobs for a meeting across all three containers.
    Returns separate arrays for recordings, notes, and sign_clips."""
    client = _get_client()
    result = {"recordings": [], "notes": [], "sign_clips": []}

    for container, key in [
        (CONTAINER_MEETINGS, "recordings"),
        (CONTAINER_NOTES, "notes"),
        (CONTAINER_SIGNS, "sign_clips"),
    ]:
        try:
            cc = client.get_container_client(container)
            for blob in cc.list_blobs(name_starts_with=f"{meeting_id}/"):
                result[key].append({
                    "name": blob.name,
                    "size_bytes": blob.size,
                    "created": blob.creation_time.isoformat() if blob.creation_time else None,
                    "download_url": _generate_sas_url(container, blob.name, hours=24),
                })
        except Exception:
            # container might not exist yet if no files were uploaded
            logger.debug("Skipping container %s for meeting %s", container, meeting_id)

    return result


async def delete_meeting_files(meeting_id: str) -> int:
    """Delete all blobs for a meeting. Returns count of deleted files."""
    client = _get_client()
    deleted = 0

    for container in [CONTAINER_MEETINGS, CONTAINER_NOTES, CONTAINER_SIGNS]:
        try:
            cc = client.get_container_client(container)
            for blob in cc.list_blobs(name_starts_with=f"{meeting_id}/"):
                cc.delete_blob(blob.name)
                deleted += 1
        except Exception:
            logger.debug("Nothing to delete in %s for meeting %s", container, meeting_id)

    logger.info("Deleted %d files for meeting %s", deleted, meeting_id)
    return deleted


# ---------------------------------------------------------------------------
# SAS URL generation
# ---------------------------------------------------------------------------

def _generate_sas_url(container: str, blob_name: str, hours: int = 24) -> str:
    """Generate a time-limited download URL with a read-only SAS token.
    Falls back to the plain blob URL if account key isn't available."""
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
    account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "")

    if not account_name or not account_key:
        # without the account key we can't sign a SAS — return the direct URL
        return _get_client().get_blob_client(container, blob_name).url

    sas = generate_blob_sas(
        account_name=account_name,
        container_name=container,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=hours),
    )
    return f"https://{account_name}.blob.core.windows.net/{container}/{blob_name}?{sas}"
