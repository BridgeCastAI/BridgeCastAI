"""
BridgeCast AI — Azure Speech Service (STT & TTS)
Wraps Azure Cognitive Services Speech SDK for real-time
speech-to-text and text-to-speech conversion.
"""

import asyncio
import io
import os
import logging
from typing import Callable, Optional

import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_speech_config(language: str = "en-US") -> speechsdk.SpeechConfig:
    """Build a SpeechConfig from environment variables."""
    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION", "eastus")

    if not key:
        raise EnvironmentError(
            "AZURE_SPEECH_KEY is not set. "
            "Export it or add it to your .env file."
        )

    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.speech_recognition_language = language
    return config


# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------

class SpeechToText:
    """Real-time speech-to-text using Azure Speech SDK.

    Supports two modes:
    1. Microphone — continuous recognition from the default mic.
    2. Push stream — feed raw PCM/WAV audio chunks programmatically
       (used by the WebSocket endpoint).
    """

    SUPPORTED_LANGUAGES = ("en-US", "ko-KR")

    def __init__(
        self,
        language: str = "en-US",
        on_recognized: Optional[Callable[[str], None]] = None,
        on_recognizing: Optional[Callable[[str], None]] = None,
    ):
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Choose from {self.SUPPORTED_LANGUAGES}"
            )

        self.language = language
        self._on_recognized = on_recognized
        self._on_recognizing = on_recognizing
        self._recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self._push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None

    # --- Microphone mode -------------------------------------------------

    def start_microphone(self) -> None:
        """Begin continuous recognition from the default microphone."""
        config = _get_speech_config(self.language)
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=config, audio_config=audio_config
        )
        self._attach_callbacks()
        self._recognizer.start_continuous_recognition()
        logger.info("STT microphone recognition started (%s)", self.language)

    # --- Push-stream mode (for WebSocket) --------------------------------

    def start_push_stream(self) -> None:
        """Begin continuous recognition from a push audio stream.

        Call `feed_audio(chunk)` to push PCM data and
        `stop()` when finished.
        """
        config = _get_speech_config(self.language)

        # 16 kHz, 16-bit, mono PCM — the most common browser format
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1,
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=audio_format
        )
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=config, audio_config=audio_config
        )
        self._attach_callbacks()
        self._recognizer.start_continuous_recognition()
        logger.info("STT push-stream recognition started (%s)", self.language)

    def feed_audio(self, chunk: bytes) -> None:
        """Push a raw PCM audio chunk into the recognizer."""
        if self._push_stream is None:
            raise RuntimeError("Push stream not started. Call start_push_stream() first.")
        self._push_stream.write(chunk)

    # --- Shared ----------------------------------------------------------

    def stop(self) -> None:
        """Stop recognition and release resources."""
        if self._recognizer is not None:
            try:
                self._recognizer.stop_continuous_recognition()
            except Exception as exc:
                logger.warning("Error stopping recognizer: %s", exc)
            self._recognizer = None

        if self._push_stream is not None:
            try:
                self._push_stream.close()
            except Exception as exc:
                logger.warning("Error closing push stream: %s", exc)
            self._push_stream = None

        logger.info("STT stopped")

    # --- Internal --------------------------------------------------------

    def _attach_callbacks(self) -> None:
        """Wire SDK events to the user-supplied callbacks."""
        if self._recognizer is None:
            return

        # Final (committed) result
        def _recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            text = evt.result.text
            if text and self._on_recognized:
                self._on_recognized(text)

        # Interim (partial) result
        def _recognizing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            text = evt.result.text
            if text and self._on_recognizing:
                self._on_recognizing(text)

        # Error / cancellation
        def _canceled_cb(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
            details = evt.cancellation_details
            logger.error(
                "STT canceled: reason=%s  error=%s",
                details.reason,
                details.error_details,
            )

        self._recognizer.recognized.connect(_recognized_cb)
        self._recognizer.recognizing.connect(_recognizing_cb)
        self._recognizer.canceled.connect(_canceled_cb)


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------

class TextToSpeech:
    """Convert text to speech audio bytes using Azure TTS."""

    # Map language codes to default Neural voices
    DEFAULT_VOICES = {
        "en-US": "en-US-JennyNeural",
        "ko-KR": "ko-KR-SunHiNeural",
    }

    def __init__(self, language: str = "en-US", voice: Optional[str] = None):
        if language not in self.DEFAULT_VOICES and voice is None:
            raise ValueError(
                f"No default voice for '{language}'. "
                f"Provide an explicit voice name."
            )
        self.language = language
        self.voice = voice or self.DEFAULT_VOICES[language]

    def _synthesize_sync(self, text: str) -> bytes:
        """Internal synchronous synthesis — runs in a thread pool to
        avoid blocking the FastAPI event loop."""
        config = _get_speech_config(self.language)
        config.speech_synthesis_voice_name = self.voice
        config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config, audio_config=None
        )

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(
                "TTS completed: %d bytes, voice=%s", len(result.audio_data), self.voice
            )
            return result.audio_data

        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(
                f"TTS canceled: {details.reason} — {details.error_details}"
            )

        raise RuntimeError(f"TTS failed with reason: {result.reason}")

    def synthesize(self, text: str) -> bytes:
        """Convert *text* to WAV audio and return the raw bytes.

        Returns an in-memory WAV file (RIFF header + PCM data).
        Raises RuntimeError on synthesis failure.
        """
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")

        return self._synthesize_sync(text)

    async def synthesize_async(self, text: str) -> bytes:
        """Async-safe TTS synthesis — offloads the blocking SDK call
        to a thread pool so the FastAPI event loop stays responsive.

        Prefer this method over synthesize() in async endpoint handlers.
        """
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")

        return await asyncio.to_thread(self._synthesize_sync, text)
