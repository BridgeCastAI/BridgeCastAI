"""
BridgeCast AI — Extended Meeting API Server
Includes sign-recognition endpoints (from api_server.py) plus
STT, TTS, meeting-notes, translation, content safety, Cosmos DB,
Azure Communication Services, and Application Insights integration.
"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, Request, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from speech_service import SpeechToText, TextToSpeech
from openai_service import generate_meeting_notes
from translator_service import translate_text, detect_language
from content_safety_service import check_text_safety
from cosmos_service import get_cosmos_service
from communication_service import create_meeting_room, get_room_token, list_participants
from monitor_service import track_event, track_metric, track_sign_recognition, track_stt_recognition, get_meeting_analytics
from avatar_service import text_to_sign_sequence, get_vocabulary
import ksl_service
from functions_service import trigger_meeting_summary, trigger_emergency_alert, trigger_accessibility_report
from blob_service import upload_meeting_recording, upload_meeting_pdf, upload_sign_clip, list_meeting_files, delete_meeting_files
from rai_assessment import get_rai_report_router
from signalr_service import negotiate as signalr_negotiate, send_to_all as signalr_broadcast, send_to_group as signalr_send_to_group
from language_service import detect_pii, redact_pii, analyze_sentiment

# Load .env file if present
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory counters for content safety statistics
# ---------------------------------------------------------------------------
_safety_stats = {"total_checks": 0, "flagged": 0, "safe": 0}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BridgeCast AI - Meeting API",
    description=(
        "Sign recognition, STT, TTS, meeting-notes, translation, "
        "content safety, Cosmos DB, ACS rooms, and Application Insights."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Responsible AI Toolbox assessment endpoints
app.include_router(get_rai_report_router())


# ---------------------------------------------------------------------------
# Middleware — Application Insights request tracking
# ---------------------------------------------------------------------------

@app.middleware("http")
async def track_requests_middleware(request: Request, call_next):
    """Track every HTTP request as an Application Insights event."""
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000

    try:
        track_event("HTTPRequest", {
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        })
        track_metric("http_request_latency_ms", round(latency_ms, 2))
    except Exception:
        pass  # Never let telemetry break the request

    return response


# ---------------------------------------------------------------------------
# Meeting room WebSocket hub — broadcasts messages to all participants
# ---------------------------------------------------------------------------

class MeetingRoom:
    """In-memory room that tracks connected WebSocket clients.
    Broadcasts STT results, sign recognition, and chat messages
    to every participant in the same room."""

    def __init__(self):
        # room_id → set of WebSocket connections
        self.rooms: Dict[str, set] = {}

    def join(self, room_id: str, ws: WebSocket):
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(ws)
        logger.info("Room %s: participant joined (%d total)", room_id, len(self.rooms[room_id]))

    def leave(self, room_id: str, ws: WebSocket):
        if room_id in self.rooms:
            self.rooms[room_id].discard(ws)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
            else:
                logger.info("Room %s: participant left (%d remain)", room_id, len(self.rooms[room_id]))

    async def broadcast(self, room_id: str, message: dict, exclude: Optional[WebSocket] = None):
        """Send a message to all participants in the room except the sender."""
        if room_id not in self.rooms:
            return
        dead = []
        for ws in self.rooms[room_id]:
            if ws is exclude:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.rooms[room_id].discard(ws)


meeting_hub = MeetingRoom()


# ---------------------------------------------------------------------------
# Sign-recognition model (ported from api_server.py)
# ---------------------------------------------------------------------------

model = None
model_loaded = False


def load_model():
    """Load Uni-Sign model (called once at startup)."""
    global model, model_loaded

    unisign_dir = os.path.expanduser("~/Uni-Sign")
    if not os.path.exists(unisign_dir):
        logger.warning("Uni-Sign not found. Sign recognition disabled.")
        return

    sys.path.insert(0, unisign_dir)

    try:
        logger.info("Loading Uni-Sign model...")
        start = time.time()

        weight_path = os.path.join(unisign_dir, "weights", "wlasl_pose_only_islr.pth")
        if not os.path.exists(weight_path):
            logger.warning("Weight file not found: %s", weight_path)
            return

        model = {
            "unisign_dir": unisign_dir,
            "weight_path": weight_path,
        }

        elapsed = time.time() - start
        logger.info("Model config ready in %.1fs", elapsed)
        model_loaded = True

    except Exception as exc:
        logger.error("Model loading failed: %s", exc)


@app.on_event("startup")
async def startup():
    """Run startup tasks: optionally load secrets from Key Vault, then load model."""
    # If AZURE_KEYVAULT_URL is set, bootstrap secrets from Key Vault
    if os.environ.get("AZURE_KEYVAULT_URL"):
        try:
            from keyvault_service import load_all_secrets
            count = load_all_secrets()
            logger.info("Loaded %d secrets from Key Vault at startup", count)
        except Exception as exc:
            logger.warning("Key Vault bootstrap failed: %s", exc)

    load_model()

    # Load KSL recognition model
    if ksl_service.load_model():
        logger.info("KSL recognition model loaded successfully")
    else:
        logger.warning("KSL recognition model not available")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check — reports model, GPU, and service configuration status."""
    gpu_available = False
    gpu_name = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except ImportError:
        pass

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "ksl_model_loaded": ksl_service.is_loaded(),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "azure_speech_configured": bool(os.environ.get("AZURE_SPEECH_KEY")),
        "azure_openai_configured": bool(os.environ.get("AZURE_OPENAI_KEY")),
        "azure_translator_configured": bool(os.environ.get("AZURE_TRANSLATOR_KEY")),
        "azure_content_safety_configured": bool(os.environ.get("AZURE_CONTENT_SAFETY_KEY")),
        "azure_cosmos_configured": bool(os.environ.get("AZURE_COSMOS_ENDPOINT")),
        "azure_communication_configured": bool(os.environ.get("AZURE_COMMUNICATION_CONNECTION_STRING")),
        "azure_appinsights_configured": bool(os.environ.get("AZURE_APPINSIGHTS_CONNECTION_STRING")),
        "azure_keyvault_configured": bool(os.environ.get("AZURE_KEYVAULT_URL")),
        "azure_functions_configured": bool(os.environ.get("AZURE_FUNCTIONS_URL")),
        "azure_blob_storage_configured": bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING")),
        "azure_signalr_configured": bool(os.environ.get("AZURE_SIGNALR_CONNECTION_STRING")),
        "azure_language_configured": bool(os.environ.get("AZURE_LANGUAGE_KEY")),
    }


# ---------------------------------------------------------------------------
# Sign-recognition endpoints (from api_server.py)
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """Sign language recognition from uploaded video."""
    if not model_loaded:
        raise HTTPException(503, "Model not loaded yet. Check /health endpoint.")

    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.time()
        import subprocess
        cmd = [
            sys.executable, "-m", "demo.online_inference",
            "--online_video", tmp_path,
            "--finetune", model["weight_path"],
        ]
        proc = subprocess.run(
            cmd,
            cwd=model["unisign_dir"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        result = "unknown"
        for line in proc.stdout.splitlines():
            if "Prediction result is:" in line:
                result = line.split("Prediction result is:")[-1].strip()
                break

        if proc.returncode != 0 and result == "unknown":
            raise RuntimeError(f"Inference failed: {proc.stderr[-500:]}")

        elapsed = time.time() - start
        latency_ms = round(elapsed * 1000)

        # Track sign recognition in Application Insights
        track_sign_recognition(result, confidence=1.0, latency_ms=latency_ms)

        return JSONResponse({"text": result, "latency_ms": latency_ms})

    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {str(exc)}")
    finally:
        os.unlink(tmp_path)


@app.post("/predict/frames")
async def predict_frames(video: UploadFile = File(...)):
    """Real-time sign recognition from short video clips."""
    return await predict(video)


# ---------------------------------------------------------------------------
# KSL Recognition (HuggingFace CTC Transformer)
# ---------------------------------------------------------------------------

@app.post("/predict/ksl")
async def predict_ksl(video: UploadFile = File(...)):
    """Korean Sign Language recognition from uploaded video."""
    if not ksl_service.is_loaded():
        raise HTTPException(503, "KSL model not loaded")

    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ksl_service.recognize_from_video(tmp_path)

        if "error" in result:
            raise HTTPException(500, result["error"])

        track_sign_recognition(
            result.get("korean", ""),
            confidence=0.65,
            latency_ms=result.get("latency_ms", 0),
        )

        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"KSL prediction failed: {str(exc)}")
    finally:
        os.unlink(tmp_path)


@app.get("/ksl/status")
async def ksl_status():
    """Return KSL model status and capabilities."""
    loaded = ksl_service.is_loaded()
    return JSONResponse({
        "loaded": loaded,
        "model": "CTC Transformer (gydms/korean-sign-language-recognition)",
        "glosses": len(ksl_service._idx2gloss) if loaded and ksl_service._idx2gloss else 0,
        "device": str(ksl_service._device) if loaded else None,
        "pipeline": "Video → MediaPipe Keypoints → CTC Transformer → Gloss → Korean",
    })


# ---------------------------------------------------------------------------
# STT — start session (returns WebSocket URL)
# ---------------------------------------------------------------------------

class STTStartRequest(BaseModel):
    language: str = "en-US"


class STTStartResponse(BaseModel):
    session_id: str
    websocket_url: str


@app.post("/stt/start", response_model=STTStartResponse)
async def stt_start(req: STTStartRequest = STTStartRequest()):
    """Start an STT session. Returns a WebSocket URL to stream audio."""
    session_id = uuid.uuid4().hex[:12]
    ws_url = f"/ws/stt?session_id={session_id}&language={req.language}"
    return STTStartResponse(session_id=session_id, websocket_url=ws_url)


# ---------------------------------------------------------------------------
# STT — WebSocket endpoint
# ---------------------------------------------------------------------------

_stt_sessions: Dict[str, SpeechToText] = {}


@app.websocket("/ws/stt")
async def ws_stt(ws: WebSocket):
    """Real-time STT over WebSocket.

    Query params:
      - language  (default: en-US)
      - session_id (optional, for tracking)

    Protocol:
      Client sends binary audio frames (16 kHz, 16-bit, mono PCM).
      Server sends JSON messages:
        {"type": "recognizing", "text": "partial..."}
        {"type": "recognized",  "text": "Final sentence."}
        {"type": "error",       "message": "..."}
    """
    await ws.accept()

    language = ws.query_params.get("language", "en-US")
    session_id = ws.query_params.get("session_id", uuid.uuid4().hex[:12])

    queue: asyncio.Queue = asyncio.Queue()

    def on_recognizing(text: str):
        queue.put_nowait({"type": "recognizing", "text": text})

    def on_recognized(text: str):
        queue.put_nowait({"type": "recognized", "text": text})
        # Track STT recognition
        track_stt_recognition(text, latency_ms=0)

    stt = SpeechToText(
        language=language,
        on_recognized=on_recognized,
        on_recognizing=on_recognizing,
    )

    try:
        stt.start_push_stream()
        _stt_sessions[session_id] = stt
        logger.info("STT WebSocket session started: %s (%s)", session_id, language)

        async def _sender():
            try:
                while True:
                    msg = await queue.get()
                    await ws.send_json(msg)
            except asyncio.CancelledError:
                pass

        sender_task = asyncio.create_task(_sender())

        while True:
            data = await ws.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data:
                stt.feed_audio(data["bytes"])
            elif "text" in data:
                try:
                    ctrl = json.loads(data["text"])
                    if ctrl.get("action") == "stop":
                        break
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("STT WebSocket disconnected: %s", session_id)
    except Exception as exc:
        logger.error("STT WebSocket error: %s", exc)
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        stt.stop()
        _stt_sessions.pop(session_id, None)
        sender_task.cancel()
        logger.info("STT session cleaned up: %s", session_id)


# ---------------------------------------------------------------------------
# Meeting Room WebSocket — real-time bidirectional communication
# ---------------------------------------------------------------------------

@app.websocket("/ws/meeting/{room_id}")
async def ws_meeting(ws: WebSocket, room_id: str):
    """Real-time meeting room communication.

    All participants in the same room_id receive each other's messages.
    Supports: chat, STT results, sign recognition results, avatar commands.

    Client sends JSON:
      {"type": "chat", "speaker": "Somi", "text": "Hello!"}
      {"type": "stt", "speaker": "Somi", "text": "recognized text"}
      {"type": "sign", "speaker": "Somi", "text": "HELLO", "confidence": 0.95}
      {"type": "avatar", "command": "sign", "glosses": ["HELLO"]}

    Server broadcasts to all OTHER participants in the room.
    """
    await ws.accept()
    meeting_hub.join(room_id, ws)

    speaker = ws.query_params.get("speaker", "Guest")

    try:
        # Notify others that someone joined
        await meeting_hub.broadcast(room_id, {
            "type": "system",
            "text": f"{speaker} joined the meeting",
            "speaker": speaker,
        }, exclude=ws)

        while True:
            data = await ws.receive_json()

            # Attach speaker name if not present
            data.setdefault("speaker", speaker)

            # Broadcast to all other participants
            await meeting_hub.broadcast(room_id, data, exclude=ws)

    except WebSocketDisconnect:
        logger.info("Meeting WS disconnected: room=%s speaker=%s", room_id, speaker)
    except Exception as exc:
        logger.error("Meeting WS error: %s", exc)
    finally:
        meeting_hub.leave(room_id, ws)
        await meeting_hub.broadcast(room_id, {
            "type": "system",
            "text": f"{speaker} left the meeting",
            "speaker": speaker,
        })


# ---------------------------------------------------------------------------
# TTS — text to speech
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    language: str = "en-US"
    voice: str | None = None


@app.post("/tts")
async def tts(req: TTSRequest):
    """Convert text to speech. Returns a WAV audio file."""
    try:
        engine = TextToSpeech(language=req.language, voice=req.voice)
        audio_bytes = engine.synthesize(req.text)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="tts_output.wav"',
            "Content-Length": str(len(audio_bytes)),
        },
    )


# ---------------------------------------------------------------------------
# Meeting notes
# ---------------------------------------------------------------------------

class TranscriptEntry(BaseModel):
    speaker: str
    text: str
    timestamp: str = ""


class MeetingNotesRequest(BaseModel):
    transcript: list[TranscriptEntry]


@app.post("/meeting/notes")
async def meeting_notes(req: MeetingNotesRequest):
    """Generate structured meeting notes from a transcript."""
    if not req.transcript:
        raise HTTPException(400, "Transcript must not be empty.")

    transcript_dicts = [entry.model_dump() for entry in req.transcript]

    try:
        notes = generate_meeting_notes(transcript_dicts)
        return JSONResponse(notes)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str
    from_lang: Optional[str] = None
    to_lang: str = "en"


@app.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    """Translate text between supported languages (en, ko, zh-Hant, ja)."""
    try:
        result = await translate_text(req.text, req.from_lang, req.to_lang)
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Translation failed: %s", exc)
        raise HTTPException(500, f"Translation failed: {exc}")


# ---------------------------------------------------------------------------
# Content Safety
# ---------------------------------------------------------------------------

class SafetyCheckRequest(BaseModel):
    text: str
    threshold: Optional[int] = None


@app.post("/safety/check")
async def safety_check_endpoint(req: SafetyCheckRequest):
    """Check text for harmful content (hate, self-harm, sexual, violence)."""
    try:
        result = await check_text_safety(req.text, req.threshold)
        # Track safety statistics
        _safety_stats["total_checks"] += 1
        if result.get("flagged"):
            _safety_stats["flagged"] += 1
        else:
            _safety_stats["safe"] += 1
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Content safety check failed: %s", exc)
        raise HTTPException(500, f"Content safety check failed: {exc}")


@app.get("/safety/stats")
async def safety_stats():
    """Return content safety check statistics — demonstrates RAI transparency."""
    return JSONResponse({
        "total_checks": _safety_stats["total_checks"],
        "safe": _safety_stats["safe"],
        "flagged": _safety_stats["flagged"],
        "safe_rate": round(
            _safety_stats["safe"] / max(_safety_stats["total_checks"], 1) * 100, 1
        ),
        "categories_monitored": ["hate", "self_harm", "sexual", "violence"],
        "service": "Azure Content Safety",
    })


# ---------------------------------------------------------------------------
# Cosmos DB — Meeting persistence
# ---------------------------------------------------------------------------

class MeetingSaveRequest(BaseModel):
    user_id: str
    title: str = ""
    transcript: list[TranscriptEntry] = []
    notes: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


@app.post("/meeting/save")
async def meeting_save(req: MeetingSaveRequest):
    """Save meeting data (transcript + notes) to Cosmos DB."""
    try:
        cosmos = get_cosmos_service()
        meeting_data = {
            "user_id": req.user_id,
            "title": req.title,
            "transcript": [e.model_dump() for e in req.transcript],
            "notes": req.notes,
            "metadata": req.metadata,
        }
        result = cosmos.save_meeting(meeting_data)
        track_event("MeetingSaved", {"meeting_id": result["id"], "user_id": req.user_id})
        return JSONResponse({"id": result["id"], "created_at": result.get("created_at")})
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.error("Failed to save meeting: %s", exc)
        raise HTTPException(500, f"Failed to save meeting: {exc}")


@app.get("/meeting/{meeting_id}")
async def meeting_get(meeting_id: str, user_id: str = Query(...)):
    """Retrieve a meeting by ID from Cosmos DB."""
    try:
        cosmos = get_cosmos_service()
        meeting = cosmos.get_meeting(meeting_id, user_id)
        if meeting is None:
            raise HTTPException(404, "Meeting not found.")
        return JSONResponse(meeting)
    except HTTPException:
        raise
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Failed to get meeting: %s", exc)
        raise HTTPException(500, f"Failed to get meeting: {exc}")


@app.get("/meetings")
@app.get("/meeting/list")
async def meetings_list(user_id: str = Query(...)):
    """List all meetings for a user from Cosmos DB."""
    try:
        cosmos = get_cosmos_service()
        meetings = cosmos.list_meetings(user_id)
        return JSONResponse({"meetings": meetings, "count": len(meetings)})
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Failed to list meetings: %s", exc)
        raise HTTPException(500, f"Failed to list meetings: {exc}")


@app.delete("/meeting/{meeting_id}")
async def meeting_delete(meeting_id: str, user_id: str = Query(...)):
    """Delete a meeting from Cosmos DB."""
    try:
        cosmos = get_cosmos_service()
        meeting = cosmos.get_meeting(meeting_id, user_id)
        if meeting is None:
            raise HTTPException(404, "Meeting not found.")
        cosmos.delete_meeting(meeting_id, user_id)
        track_event("MeetingDeleted", {"meeting_id": meeting_id, "user_id": user_id})
        return JSONResponse({"deleted": True, "meeting_id": meeting_id})
    except HTTPException:
        raise
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Failed to delete meeting: %s", exc)
        raise HTTPException(500, f"Failed to delete meeting: {exc}")


# ---------------------------------------------------------------------------
# Azure Communication Services — Rooms
# ---------------------------------------------------------------------------

@app.post("/room/create")
async def room_create():
    """Create a new ACS meeting room."""
    try:
        result = await create_meeting_room()
        track_event("RoomCreated", {"room_id": result["room_id"]})
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Failed to create room: %s", exc)
        raise HTTPException(500, f"Failed to create room: {exc}")


@app.get("/room/{room_id}/token")
async def room_token(room_id: str, user_id: Optional[str] = Query(None)):
    """Generate an access token for a participant in the given room."""
    try:
        result = await get_room_token(room_id, user_id)
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Failed to get room token: %s", exc)
        raise HTTPException(500, f"Failed to get room token: {exc}")


@app.get("/room/{room_id}/participants")
async def room_participants(room_id: str):
    """List all participants in a room."""
    try:
        result = await list_participants(room_id)
        return JSONResponse({"participants": result, "count": len(result)})
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Failed to list participants: %s", exc)
        raise HTTPException(500, f"Failed to list participants: {exc}")


# ---------------------------------------------------------------------------
# Application Insights — Analytics
# ---------------------------------------------------------------------------

@app.get("/analytics/meeting/{meeting_id}")
async def analytics_meeting(meeting_id: str):
    """Return participation and usage analytics for a meeting."""
    result = get_meeting_analytics(meeting_id)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Sign Language Avatar — Text to Sign Animation
# ---------------------------------------------------------------------------

class AvatarSignRequest(BaseModel):
    text: str
    language: str = "asl"  # "asl" | "ksl" | "tsl"


@app.post("/avatar/sign")
async def avatar_sign(req: AvatarSignRequest):
    """Convert text to sign language gloss sequence and animation data.

    Supports multiple sign languages via modular architecture:
    - language="asl": English text → ASL gloss → ASL animations (114 signs)
    - language="ksl": Korean text → KSL gloss → KSL animations (22 signs)
    - language="tsl": Chinese text → TSL gloss → TSL animations (20 signs)

    Example request: {"text": "Hello, nice to meet you", "language": "asl"}
    Example request: {"text": "만나서 반갑습니다", "language": "ksl"}
    Example request: {"text": "很高興認識你", "language": "tsl"}
    """
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Text must not be empty.")

    lang = req.language.lower().strip()
    if lang not in ("asl", "ksl", "tsl"):
        raise HTTPException(400, f"Unsupported language: {req.language}. Supported: asl, ksl, tsl")

    try:
        result = text_to_sign_sequence(req.text.strip(), language=lang)
        track_event("AvatarSign", {
            "text_length": len(req.text),
            "language": lang,
            "sign_count": result["sign_count"],
            "known_signs": result["known_signs"],
            "unknown_signs": result["unknown_signs"],
        })
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Avatar sign conversion failed: %s", exc)
        raise HTTPException(500, f"Avatar sign conversion failed: {exc}")


@app.get("/avatar/vocabulary")
async def avatar_vocabulary(language: str = Query("asl")):
    """Return the list of supported sign vocabulary for a given language.

    Query params:
      - language: "asl" (default), "ksl", or "tsl"
    """
    lang = language.lower().strip()
    if lang not in ("asl", "ksl", "tsl"):
        raise HTTPException(400, f"Unsupported language: {language}. Supported: asl, ksl, tsl")

    try:
        vocab = get_vocabulary(language=lang)
        return JSONResponse({
            "vocabulary": vocab,
            "count": len(vocab),
            "language": lang,
        })
    except Exception as exc:
        logger.error("Failed to get avatar vocabulary: %s", exc)
        raise HTTPException(500, f"Failed to get vocabulary: {exc}")


# ---------------------------------------------------------------------------
# Azure Functions — Serverless Triggers
# ---------------------------------------------------------------------------

class MeetingSummaryRequest(BaseModel):
    meeting_id: str
    transcript: list


@app.post("/functions/summarize")
async def functions_summarize(req: MeetingSummaryRequest):
    """Trigger Azure Function to generate meeting summary."""
    try:
        result = await trigger_meeting_summary(req.meeting_id, req.transcript)
        track_event("FunctionsSummarize", {"meeting_id": req.meeting_id})
        return JSONResponse(result)
    except Exception as exc:
        logger.error("Functions summarize failed: %s", exc)
        raise HTTPException(500, str(exc))


class EmergencyAlertRequest(BaseModel):
    meeting_id: str
    text: str
    speaker: str


@app.post("/functions/emergency-alert")
async def functions_emergency_alert(req: EmergencyAlertRequest):
    """Trigger Azure Function for emergency alert detection."""
    try:
        result = await trigger_emergency_alert(req.meeting_id, req.text, req.speaker)
        if result.get("alert"):
            track_event("EmergencyAlert", {"meeting_id": req.meeting_id, "speaker": req.speaker})
        return JSONResponse(result)
    except Exception as exc:
        logger.error("Emergency alert failed: %s", exc)
        raise HTTPException(500, str(exc))


class AccessibilityReportRequest(BaseModel):
    meeting_id: str
    metrics: dict


@app.post("/functions/accessibility-report")
async def functions_accessibility_report(req: AccessibilityReportRequest):
    """Trigger Azure Function to generate accessibility report."""
    try:
        result = await trigger_accessibility_report(req.meeting_id, req.metrics)
        track_event("AccessibilityReport", {"meeting_id": req.meeting_id})
        return JSONResponse(result)
    except Exception as exc:
        logger.error("Accessibility report failed: %s", exc)
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------------------
# Azure Blob Storage — Meeting Files
# ---------------------------------------------------------------------------

@app.post("/storage/upload-recording")
async def storage_upload_recording(meeting_id: str = Query(...), file: UploadFile = File(...)):
    """Upload meeting recording to Azure Blob Storage."""
    try:
        data = await file.read()
        result = await upload_meeting_recording(meeting_id, data, file.content_type or "video/webm")
        track_event("BlobUploadRecording", {"meeting_id": meeting_id, "size": len(data)})
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Upload recording failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/storage/upload-pdf")
async def storage_upload_pdf(meeting_id: str = Query(...), file: UploadFile = File(...)):
    """Upload meeting notes PDF to Azure Blob Storage."""
    try:
        data = await file.read()
        result = await upload_meeting_pdf(meeting_id, data, file.filename or "meeting_notes.pdf")
        track_event("BlobUploadPDF", {"meeting_id": meeting_id})
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Upload PDF failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/storage/upload-sign-clip")
async def storage_upload_sign_clip(
    meeting_id: str = Query(...),
    sign_label: str = Query(...),
    file: UploadFile = File(...),
):
    """Upload sign language video clip to Azure Blob Storage."""
    try:
        data = await file.read()
        result = await upload_sign_clip(meeting_id, data, sign_label, file.content_type or "video/webm")
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Upload sign clip failed: %s", exc)
        raise HTTPException(500, str(exc))


@app.get("/storage/meeting-files/{meeting_id}")
async def storage_meeting_files(meeting_id: str):
    """List all files associated with a meeting in Blob Storage."""
    try:
        result = await list_meeting_files(meeting_id)
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("List meeting files failed: %s", exc)
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------------------
# Batch Translation
# ---------------------------------------------------------------------------

class BatchTranslateRequest(BaseModel):
    texts: List[str]
    to: str = "en"
    from_lang: Optional[str] = None


@app.post("/translate/batch")
async def translate_batch(req: BatchTranslateRequest):
    """Translate multiple texts in a single request."""
    if not req.texts:
        raise HTTPException(400, "texts list must not be empty.")
    if len(req.texts) > 100:
        raise HTTPException(400, "Maximum 100 texts per batch request.")

    results = []
    errors = []

    for idx, text in enumerate(req.texts):
        try:
            result = await translate_text(text, req.from_lang, req.to)
            results.append({"index": idx, "original": text, **result})
        except Exception as exc:
            logger.error("Batch translate item %d failed: %s", idx, exc)
            errors.append({"index": idx, "original": text, "error": str(exc)})

    track_event("BatchTranslate", {
        "count": len(req.texts),
        "to_lang": req.to,
        "success_count": len(results),
        "error_count": len(errors),
    })

    return JSONResponse({
        "results": results,
        "errors": errors,
        "total": len(req.texts),
        "success_count": len(results),
        "error_count": len(errors),
    })


# ---------------------------------------------------------------------------
# Export Meeting as PDF
# ---------------------------------------------------------------------------

class MeetingExportPDFRequest(BaseModel):
    meeting_id: str
    user_id: str
    title: str = "Meeting Transcript"


@app.post("/meeting/export-pdf")
async def meeting_export_pdf(req: MeetingExportPDFRequest):
    """Export a meeting transcript as a PDF file.

    Retrieves the meeting from Cosmos DB, generates a PDF from the transcript
    and notes, then uploads it to Blob Storage.
    Falls back to JSON export if PDF libraries are not available.
    """
    try:
        cosmos = get_cosmos_service()
        meeting = cosmos.get_meeting(req.meeting_id, req.user_id)
        if meeting is None:
            raise HTTPException(404, "Meeting not found.")

        transcript = meeting.get("transcript", [])
        notes = meeting.get("notes", {})
        title = req.title or meeting.get("title", "Meeting Transcript")
        created_at = meeting.get("created_at", "")

        # Try generating a real PDF with reportlab
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.units import mm

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=20 * mm, rightMargin=20 * mm,
                                    topMargin=20 * mm, bottomMargin=20 * mm)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("MeetingTitle", parent=styles["Title"], fontSize=18)
            body_style = styles["BodyText"]
            speaker_style = ParagraphStyle("Speaker", parent=styles["BodyText"],
                                           fontName="Helvetica-Bold")

            elements = []
            elements.append(Paragraph(title, title_style))
            elements.append(Spacer(1, 5 * mm))
            if created_at:
                elements.append(Paragraph(f"Date: {created_at}", body_style))
                elements.append(Spacer(1, 3 * mm))

            # Transcript section
            elements.append(Paragraph("Transcript", styles["Heading2"]))
            elements.append(Spacer(1, 2 * mm))
            for entry in transcript:
                speaker = entry.get("speaker", "Unknown")
                text = entry.get("text", "")
                ts = entry.get("timestamp", "")
                line = f"<b>{speaker}</b>{' [' + ts + ']' if ts else ''}: {text}"
                elements.append(Paragraph(line, body_style))
                elements.append(Spacer(1, 1 * mm))

            # Notes summary section
            if notes:
                elements.append(Spacer(1, 5 * mm))
                elements.append(Paragraph("Meeting Notes", styles["Heading2"]))
                elements.append(Spacer(1, 2 * mm))
                if isinstance(notes.get("summary"), str):
                    elements.append(Paragraph(notes["summary"], body_style))
                if isinstance(notes.get("action_items"), list):
                    elements.append(Spacer(1, 3 * mm))
                    elements.append(Paragraph("Action Items:", speaker_style))
                    for item in notes["action_items"]:
                        elements.append(Paragraph(f"  - {item}", body_style))

            doc.build(elements)
            pdf_bytes = buf.getvalue()

            # Upload PDF to blob storage
            try:
                filename = f"{req.meeting_id}_transcript.pdf"
                await upload_meeting_pdf(req.meeting_id, pdf_bytes, filename)
            except Exception as upload_exc:
                logger.warning("PDF blob upload failed (returning PDF directly): %s", upload_exc)

            track_event("MeetingExportPDF", {"meeting_id": req.meeting_id, "size": len(pdf_bytes)})

            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{req.meeting_id}_transcript.pdf"',
                    "Content-Length": str(len(pdf_bytes)),
                },
            )

        except ImportError:
            # reportlab not installed — return structured JSON for client-side rendering
            logger.info("reportlab not available; returning JSON export for meeting %s", req.meeting_id)
            track_event("MeetingExportJSON", {"meeting_id": req.meeting_id})
            return JSONResponse({
                "format": "json",
                "message": "PDF library not available. Returning data for client-side rendering.",
                "meeting_id": req.meeting_id,
                "title": title,
                "created_at": created_at,
                "transcript": transcript,
                "notes": notes,
            })

    except HTTPException:
        raise
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Meeting PDF export failed: %s", exc)
        raise HTTPException(500, f"Meeting PDF export failed: {exc}")


# ---------------------------------------------------------------------------
# ACS Room — Join
# ---------------------------------------------------------------------------

class RoomJoinRequest(BaseModel):
    room_id: str
    user_id: Optional[str] = None
    display_name: str = "Participant"


@app.post("/room/join")
async def room_join(req: RoomJoinRequest):
    """Join an ACS room. Creates a new identity if user_id is not provided,
    then adds the user as a participant and returns an access token."""
    try:
        result = await get_room_token(req.room_id, req.user_id)
        track_event("RoomJoined", {
            "room_id": req.room_id,
            "user_id": result["user_id"],
            "display_name": req.display_name,
        })
        return JSONResponse({
            **result,
            "display_name": req.display_name,
            "joined": True,
        })
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Failed to join room: %s", exc)
        raise HTTPException(500, f"Failed to join room: {exc}")


# ---------------------------------------------------------------------------
# ACS Room — Get Details
# ---------------------------------------------------------------------------

@app.get("/room/{room_id}")
async def room_get(room_id: str):
    """Get details of an ACS room including participants."""
    try:
        from communication_service import _get_rooms_client

        rooms_client = _get_rooms_client()
        room = rooms_client.get_room(room_id=room_id)

        participants = await list_participants(room_id)

        return JSONResponse({
            "room_id": room.id,
            "created_at": room.created_at.isoformat() if hasattr(room, "created_at") and room.created_at else None,
            "valid_from": room.valid_from.isoformat() if hasattr(room, "valid_from") and room.valid_from else None,
            "valid_until": room.valid_until.isoformat() if hasattr(room, "valid_until") and room.valid_until else None,
            "participants": participants,
            "participant_count": len(participants),
        })
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Failed to get room: %s", exc)
        raise HTTPException(500, f"Failed to get room: {exc}")


# ---------------------------------------------------------------------------
# Avatar — Generate Animation from Text
# ---------------------------------------------------------------------------

class AvatarGenerateRequest(BaseModel):
    text: str
    language: str = "asl"
    include_timing: bool = True


@app.post("/avatar/generate")
async def avatar_generate(req: AvatarGenerateRequest):
    """Generate avatar animation data from text input.

    Converts text to sign language gloss, then generates full animation
    data including per-sign timing and transitions.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Text must not be empty.")

    lang = req.language.lower().strip()
    if lang not in ("asl", "ksl", "tsl"):
        raise HTTPException(400, f"Unsupported language: {req.language}. Supported: asl, ksl, tsl")

    try:
        sequence = text_to_sign_sequence(req.text.strip(), language=lang)

        # Enrich with timing data for animation playback
        animations = sequence.get("signs", [])
        if req.include_timing:
            total_duration_ms = 0
            for sign in animations:
                sign_duration = sign.get("duration_ms", 800)
                sign["start_ms"] = total_duration_ms
                sign["end_ms"] = total_duration_ms + sign_duration
                total_duration_ms += sign_duration + 200  # 200ms transition gap

            sequence["total_duration_ms"] = total_duration_ms
            sequence["transition_gap_ms"] = 200

        track_event("AvatarGenerate", {
            "text_length": len(req.text),
            "language": lang,
            "sign_count": sequence.get("sign_count", 0),
        })

        return JSONResponse(sequence)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Avatar generation failed: %s", exc)
        raise HTTPException(500, f"Avatar generation failed: {exc}")


# ---------------------------------------------------------------------------
# Avatar — List Available Signs
# ---------------------------------------------------------------------------

@app.get("/avatar/signs")
async def avatar_signs(language: str = Query("asl")):
    """List all available sign vocabulary with sign names only.

    Query params:
      - language: "asl" (default), "ksl", or "tsl"
    """
    lang = language.lower().strip()
    if lang not in ("asl", "ksl", "tsl"):
        raise HTTPException(400, f"Unsupported language: {language}. Supported: asl, ksl, tsl")

    try:
        vocab = get_vocabulary(language=lang)
        sign_names = [entry.get("gloss") or entry.get("name", "") for entry in vocab]
        return JSONResponse({
            "signs": sign_names,
            "count": len(sign_names),
            "language": lang,
        })
    except Exception as exc:
        logger.error("Failed to list avatar signs: %s", exc)
        raise HTTPException(500, f"Failed to list signs: {exc}")


# ---------------------------------------------------------------------------
# SignalR — Real-time Messaging
# ---------------------------------------------------------------------------

@app.post("/signalr/negotiate")
async def signalr_negotiate_endpoint(user_id: Optional[str] = None):
    """Negotiate a SignalR connection for real-time bidirectional messaging.

    Returns a WebSocket URL and access token for the client to connect
    to Azure SignalR Service. Used for broadcasting transcriptions,
    sign recognition results, and emergency alerts in real-time.
    """
    try:
        result = await signalr_negotiate(user_id)
        track_event("SignalRNegotiate", {"user_id": user_id or "anonymous"})
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("SignalR negotiate failed: %s", exc)
        raise HTTPException(500, f"SignalR negotiation failed: {exc}")


class SignalRMessageRequest(BaseModel):
    message: dict
    group: Optional[str] = None


@app.post("/signalr/broadcast")
async def signalr_broadcast_endpoint(req: SignalRMessageRequest):
    """Broadcast a message to all connected clients or a specific group.

    When 'group' is provided (typically a meeting room ID), the message
    is sent only to participants in that group. Otherwise, it is broadcast
    to all connected clients across the hub.
    """
    try:
        if req.group:
            result = await signalr_send_to_group(req.group, req.message)
        else:
            result = await signalr_broadcast(req.message)

        track_event("SignalRBroadcast", {
            "group": req.group or "all",
            "success": result.get("success", False),
        })
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("SignalR broadcast failed: %s", exc)
        raise HTTPException(500, f"Broadcast failed: {exc}")


# ---------------------------------------------------------------------------
# PII Detection & Redaction (Azure AI Language)
# ---------------------------------------------------------------------------

class PIIRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/pii/detect")
async def pii_detect_endpoint(req: PIIRequest):
    """Detect and redact Personally Identifiable Information (PII) from text.

    Uses Azure AI Language to identify entities such as names, emails,
    phone numbers, and addresses. Returns the original text with PII
    entities replaced by asterisks, plus a list of detected entities
    with their categories and confidence scores.

    Responsible AI: Ensures user privacy by preventing accidental
    exposure of personal data in meeting transcripts and summaries.
    """
    try:
        result = await detect_pii(req.text, req.language)
        track_event("PIIDetection", {
            "language": req.language,
            "pii_detected": result.get("pii_detected", False),
            "entity_count": len(result.get("entities", [])),
        })
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("PII detection failed: %s", exc)
        raise HTTPException(500, f"PII detection failed: {exc}")


@app.post("/pii/redact")
async def pii_redact_endpoint(req: PIIRequest):
    """Redact PII from text and return only the sanitized version.

    Convenience endpoint that returns just the redacted text string,
    suitable for pipeline integration where only the clean output
    is needed (e.g., before storing transcripts in Cosmos DB).
    """
    try:
        redacted = await redact_pii(req.text, req.language)
        return JSONResponse({"redactedText": redacted})
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("PII redaction failed: %s", exc)
        raise HTTPException(500, f"PII redaction failed: {exc}")


# ---------------------------------------------------------------------------
# Sentiment Analysis (Azure AI Language)
# ---------------------------------------------------------------------------

@app.post("/sentiment")
async def sentiment_endpoint(req: PIIRequest):
    """Analyze the sentiment of text (positive, neutral, negative, mixed).

    Uses Azure AI Language to provide document-level and sentence-level
    sentiment scores. Integrated into the meeting flow to tag each
    transcription with emotional tone — enabling the avatar to display
    matching facial expressions and the transcript to show emotion badges.
    """
    try:
        result = await analyze_sentiment(req.text, req.language)
        track_event("SentimentAnalysis", {
            "language": req.language,
            "sentiment": result.get("sentiment", "unknown"),
        })
        return JSONResponse(result)
    except EnvironmentError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error("Sentiment analysis failed: %s", exc)
        raise HTTPException(500, f"Sentiment analysis failed: {exc}")


# ---------------------------------------------------------------------------
# Analytics — Dashboard
# ---------------------------------------------------------------------------

@app.get("/analytics/dashboard")
async def analytics_dashboard():
    """Return aggregated dashboard analytics including model status,
    service health, and usage counters."""
    gpu_available = False
    gpu_name = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except ImportError:
        pass

    # Get avatar vocabulary counts per language
    vocab_counts = {}
    for lang in ("asl", "ksl", "tsl"):
        try:
            vocab = get_vocabulary(language=lang)
            vocab_counts[lang] = len(vocab)
        except Exception:
            vocab_counts[lang] = 0

    return JSONResponse({
        "models": {
            "unisign_loaded": model_loaded,
            "ksl_loaded": ksl_service.is_loaded(),
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
        },
        "services": {
            "speech": bool(os.environ.get("AZURE_SPEECH_KEY")),
            "openai": bool(os.environ.get("AZURE_OPENAI_KEY")),
            "translator": bool(os.environ.get("AZURE_TRANSLATOR_KEY")),
            "content_safety": bool(os.environ.get("AZURE_CONTENT_SAFETY_KEY")),
            "cosmos_db": bool(os.environ.get("AZURE_COSMOS_ENDPOINT")),
            "communication": bool(os.environ.get("AZURE_COMMUNICATION_CONNECTION_STRING")),
            "app_insights": bool(os.environ.get("AZURE_APPINSIGHTS_CONNECTION_STRING")),
            "key_vault": bool(os.environ.get("AZURE_KEYVAULT_URL")),
            "functions": bool(os.environ.get("AZURE_FUNCTIONS_URL")),
            "blob_storage": bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING")),
            "signalr": bool(os.environ.get("AZURE_SIGNALR_CONNECTION_STRING")),
            "language_pii": bool(os.environ.get("AZURE_LANGUAGE_KEY")),
        },
        "content_safety": {
            "total_checks": _safety_stats["total_checks"],
            "flagged": _safety_stats["flagged"],
            "safe": _safety_stats["safe"],
        },
        "avatar_vocabulary": vocab_counts,
        "active_stt_sessions": len(_stt_sessions),
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="BridgeCast AI Meeting API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--keyvault",
        action="store_true",
        help="Load secrets from Azure Key Vault at startup (requires AZURE_KEYVAULT_URL)",
    )
    args = parser.parse_args()

    if args.keyvault:
        os.environ.setdefault("AZURE_KEYVAULT_URL", "")
        if not os.environ.get("AZURE_KEYVAULT_URL"):
            logger.error("--keyvault flag used but AZURE_KEYVAULT_URL is not set.")
            sys.exit(1)

    uvicorn.run(app, host=args.host, port=args.port)
