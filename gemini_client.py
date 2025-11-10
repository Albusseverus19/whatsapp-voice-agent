import os
import asyncio
import logging
from typing import Callable, Awaitable, Optional

from google import genai
from google.genai import types

logger = logging.getLogger("gemini-live")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-native-audio-preview-09-2025"
)

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set. Gemini Live will not work until configured.")


class GeminiLiveSession:
    """
    One Gemini Live session per call.
    - send_audio_chunk(pcm16) is called with user audio.
    - When Gemini completes a turn, it calls async on_final_text(text).
    """

    def __init__(self, call_sid: str, on_final_text: Callable[[str], Awaitable[None]]):
        self.call_sid = call_sid
        self.on_final_text = on_final_text

        self._client: Optional[genai.Client] = None
        self._session = None
        self._listen_task: Optional[asyncio.Task] = None
        self._current_text_parts = []

    async def start(self):
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY missing, cannot start GeminiLiveSession.")
            return

        self._client = genai.Client(api_key=GEMINI_API_KEY)

        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": (
                "You are a helpful real-time voice assistant. "
                "User primarily speaks Georgian. Respond briefly, clearly, and in Georgian."
            ),
        }

        logger.info(f"[{self.call_sid}] Starting Gemini Live session with model={GEMINI_MODEL}")

        self._session = await self._client.aio.live.connect(
            model=GEMINI_MODEL,
            config=config,
        )

        # background listener
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        try:
            async for event in self._session.receive():
                server_content = getattr(event, "server_content", None)
                if not server_content:
                    continue

                model_turn = getattr(server_content, "model_turn", None)
                if not model_turn:
                    continue

                for part in model_turn.parts:
                    if hasattr(part, "text") and part.text:
                        self._current_text_parts.append(part.text)

                if getattr(server_content, "turn_complete", False):
                    final_text = "".join(self._current_text_parts).strip()
                    self._current_text_parts = []
                    if final_text:
                        logger.info(f"[#{self.call_sid}] Gemini final: {final_text}")
                        try:
                            # async callback: let app.py handle TTS+Twilio
                            await self.on_final_text(final_text)
                        except Exception as e:
                            logger.error(f"[{self.call_sid}] on_final_text error: {e}")
        except Exception as e:
            logger.error(f"[{self.call_sid}] Gemini listen loop error: {e}")

    async def send_audio_chunk(self, pcm16_bytes: bytes, sample_rate: int = 8000):
        if not self._session:
            return
        try:
            blob = types.Blob(
                data=pcm16_bytes,
                mime_type=f"audio/pcm;rate={sample_rate}",
            )
            await self._session.send_realtime_input(audio=blob)
        except Exception as e:
            logger.error(f"[{self.call_sid}] Error sending audio to Gemini: {e}")

    async def close(self):
        try:
            if self._session:
                await self._session.close()
        except Exception as e:
            logger.error(f"[{self.call_sid}] Error closing Gemini session: {e}")
