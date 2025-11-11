import os
import asyncio
import logging
import base64
import json
from typing import Callable, Awaitable, Optional

from google import genai
from google.genai import types

logger = logging.getLogger("gemini-live")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-live-001")


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
        self._session_cm = None  # async context manager
        self._session = None     # live session object
        self._listen_task: Optional[asyncio.Task] = None

        self._current_text_parts = []
        self._closed = False

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
            # Let Gemini send us transcripts of what it thinks the user said
            "input_audio_transcription": {},
        }


        logger.info(f"[{self.call_sid}] Starting Gemini Live session with model={GEMINI_MODEL}")

        # live.connect() returns an async context manager -> must use __aenter__()
        self._session_cm = self._client.aio.live.connect(
            model=GEMINI_MODEL,
            config=config,
        )
        self._session = await self._session_cm.__aenter__()

        # Start background listener
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        """
        Listen for Gemini responses, collect text, and call on_final_text()
        whenever a turn is completed.
        """
        try:
            async for event in self._session.receive():
                logger.info(f"[{self.call_sid}] Gemini event: {event}")

                # 1) If the SDK surfaces plain text (per docs), treat it as a turn.
                plain_text = getattr(event, "text", None)
                if plain_text:
                    final_text = plain_text.strip()
                    if final_text:
                        logger.info(f"[{self.call_sid}] Gemini final (text field): {final_text}")
                        try:
                            await self.on_final_text(final_text)
                        except Exception as e:
                            logger.error(f"[{self.call_sid}] on_final_text error: {e}")
                    continue

                server_content = getattr(event, "server_content", None)
                if not server_content:
                    continue

                # 2) Log input transcription (what Gemini heard from user) – debug only.
                input_tr = getattr(server_content, "input_transcription", None)
                if input_tr and getattr(input_tr, "text", None):
                    logger.info(f"[{self.call_sid}] Gemini heard (user): {input_tr.text}")

                # 3) Accumulate any model_turn text chunks.
                model_turn = getattr(server_content, "model_turn", None)
                if model_turn:
                    for part in getattr(model_turn, "parts", []):
                        text = getattr(part, "text", None)
                        if text:
                            self._current_text_parts.append(text)

                # 4) Consider both turn_complete and generation_complete as end-of-turn.
                is_turn_complete = bool(getattr(server_content, "turn_complete", False))
                is_gen_complete = bool(getattr(server_content, "generation_complete", False))

                if is_turn_complete or is_gen_complete:
                    final_text = "".join(self._current_text_parts).strip()
                    self._current_text_parts = []

                    if final_text:
                        logger.info(f"[{self.call_sid}] Gemini final: {final_text}")
                        try:
                            await self.on_final_text(final_text)
                        except Exception as e:
                            logger.error(f"[{self.call_sid}] on_final_text error: {e}")

        except Exception as e:
            if not self._closed:
                logger.error(f"[{self.call_sid}] Gemini listen loop error: {e}")



    async def send_audio_chunk(self, pcm16_bytes: bytes, sample_rate: int = 16000):
        """
        Stream PCM16 mono audio chunks to Gemini Live.

        Expects: 16-bit PCM, little-endian, mono, `sample_rate` Hz (we use 16000).
        """
        if not self._session:
            return

        if not pcm16_bytes:
            return

        try:
            blob = types.Blob(
                data=pcm16_bytes,
                mime_type=f"audio/pcm;rate={sample_rate}",
            )
            # google-genai Live API: send audio via send_realtime_input
            await self._session.send_realtime_input(audio=blob)

        except Exception as e:
            logger.error(f"[{self.call_sid}] Error sending audio to Gemini: {e}")
