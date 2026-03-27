"""
BridgeCast AI — Azure OpenAI Service
Generates structured meeting notes from a conversation transcript.
"""

import json
import os
import logging
from typing import Any, Dict, List

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _get_client() -> AzureOpenAI:
    """Return an AzureOpenAI client configured from environment variables."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_KEY")

    if not endpoint or not key:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version="2024-06-01",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert meeting-notes assistant for BridgeCast AI, an inclusive communication platform
that bridges Deaf sign language users and hearing participants.

Given a timestamped transcript of a meeting, produce a structured summary
in JSON with the following keys:

{
  "title": "Short meeting title inferred from the content",
  "summary": "2-4 sentence overall summary",
  "topics": [
    {
      "topic": "Topic name",
      "summary": "Brief summary of discussion on this topic"
    }
  ],
  "key_decisions": ["Decision 1", "Decision 2"],
  "action_items": [
    {
      "owner": "Person responsible (or 'Unassigned')",
      "task": "Description of the action item",
      "deadline": "Mentioned deadline or 'TBD'"
    }
  ],
  "participants": ["Speaker A", "Speaker B"],
  "utterance_sentiments": [
    {
      "speaker": "Speaker name",
      "text": "What they said (abbreviated)",
      "sentiment": "positive | neutral | concern | urgent",
      "modality": "voice | sign"
    }
  ],
  "accessibility_report": {
    "total_utterances": 0,
    "sign_utterances": 0,
    "voice_utterances": 0,
    "sentiment_distribution": {
      "positive": 0,
      "neutral": 0,
      "concern": 0,
      "urgent": 0
    },
    "participation_balance": "Description of how balanced participation was across modalities"
  }
}

Rules:
- Output valid JSON only, no markdown fences.
- If a field has no data, use an empty list [] or zeros.
- Keep summaries concise and professional.
- Identify participants from the speaker names in the transcript.
- Speakers with "(sign)" in their name used sign language; tag their modality as "sign".
- Speakers with "(voice)" or no tag used speech; tag their modality as "voice".
- Analyze sentiment of each utterance based on text meaning and context.
- The accessibility_report should quantify participation across modalities to demonstrate measurable accessibility impact.
"""


def generate_meeting_notes(
    transcript: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Generate structured meeting notes from a transcript.

    Parameters
    ----------
    transcript : list of dict
        Each item has keys: ``speaker``, ``text``, and ``timestamp``.
        Example::

            [
                {"speaker": "Alice", "text": "Let's discuss the roadmap.", "timestamp": "00:00:12"},
                {"speaker": "Bob",   "text": "I think we should prioritize the API.", "timestamp": "00:00:25"},
            ]

    Returns
    -------
    dict
        Structured meeting notes (see SYSTEM_PROMPT for schema).
    """
    if not transcript:
        return {
            "title": "Empty meeting",
            "summary": "No transcript was provided.",
            "topics": [],
            "key_decisions": [],
            "action_items": [],
            "participants": [],
        }

    # Format transcript into a readable block
    lines = []
    for entry in transcript:
        ts = entry.get("timestamp", "??:??:??")
        speaker = entry.get("speaker", "Unknown")
        text = entry.get("text", "")
        lines.append(f"[{ts}] {speaker}: {text}")
    transcript_text = "\n".join(lines)

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript_text},
            ],
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        notes = json.loads(raw)
        logger.info("Meeting notes generated successfully")
        return notes

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse OpenAI response as JSON: %s", exc)
        raise RuntimeError("OpenAI returned invalid JSON for meeting notes.") from exc

    except Exception as exc:
        logger.error("Azure OpenAI call failed: %s", exc)
        raise RuntimeError(f"Meeting notes generation failed: {exc}") from exc
