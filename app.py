import os
import io
import json
import base64
import logging
import asyncio
import audioop
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse
from pydub import AudioSegment

from config import MEDIA_STREAM_WS_URL
from gemini_client import GeminiLiveSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-eleven-gemini")

app = FastAPI()

# ----------- Config from env -----------

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_v3")

# Automatic greeting text (Georgian)
INITIAL_GREETING = os.getenv(
    "INITIAL_GREETING",
    "გამარჯობა, მე ვარ თქვენი ვოის ასისტენტი. გისმენთ."
)

# ----------- μ-law encoder (Twilio-compatible) -----------

def pcm16_to_mulaw(pcm16: bytes) -> bytes:
    """
    Convert 16-bit linear PCM (little-endian) to 8-bit μ-law.
    Uses Python's audioop.lin2ulaw, which matches Twilio's μ-law expectations.
    """
    if not pcm16:
        return b""
    # width=2 because samples are 16-bit
    return audioop.lin2ulaw(pcm16, 2)


# ----------- Call state -----------

class CallState:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.ws: Optional[WebSocket] = None          # Twilio WebSocket connection
        self.stream_sid: Optional[str] = None        # Twilio Media Stream SID
        self.gemini_session: Optional[GeminiLiveSession] = None
        self.speaking = False                        # prevent overlapping TTS
        self.greeted = False                         # ensure greeting only once

    async def start_gemini(self):
        if self.gemini_session is None:
            async def on_final_text(text: str):
                """
                Called when Gemini has a complete reply.
                Trigger ElevenLabs TTS and send audio back to Twilio.
                """
                await self.speak_text(text)

            self.gemini_session = GeminiLiveSession(
                call_sid=self.call_sid,
                on_final_text=on_final_text,
            )
            await self.gemini_session.start()

    async def handle_audio_from_twilio(self, mulaw_bytes: bytes):
        """
        Convert Twilio μ-law (8-bit, 8kHz) -> 16-bit PCM and stream to Gemini.
        """
        if not self.gemini_session:
            await self.start_gemini()

        if not mulaw_bytes:
            return

        # Properly decode μ-law to 16-bit linear PCM.
        # width=2 => output sample width is 16-bit.
        try:
            pcm16 = audioop.ulaw2lin(mulaw_bytes, 2)
        except Exception as e:
            logger.error(f"[{self.call_sid}] Error decoding μ-law from Twilio: {e}")
            return

        await self.gemini_session.send_audio_chunk(
            pcm16_bytes=pcm16,
            sample_rate=8000,  # Twilio stream is 8kHz
        )

    async def send_twilio_media(self, mulaw_bytes: bytes):
        """
        Send μ-law 8kHz audio back to Twilio via the media stream
        in strict 20 ms (160-byte) frames, padding the last frame if needed.
        """
        if not self.ws or not self.stream_sid:
            logger.warning(
                f"[{self.call_sid}] Cannot send media: ws={bool(self.ws)}, stream_sid={self.stream_sid}"
            )
            return

        FRAME_SIZE = 160            # 20ms @ 8000 Hz μ-law
        FRAME_DURATION = 0.02       # 20ms in seconds

        total = len(mulaw_bytes)
        if total == 0:
            return

        logger.info(f"[{self.call_sid}] Sending {total} bytes of μ-law audio to Twilio")

        offset = 0
        while offset < total:
            chunk = mulaw_bytes[offset:offset + FRAME_SIZE]
            offset += FRAME_SIZE

            # Pad last frame if it's shorter than 160 bytes
            if len(chunk) < FRAME_SIZE:
                padding = FRAME_SIZE - len(chunk)
                chunk += b'\xFF' * padding  # μ-law silence

            payload = base64.b64encode(chunk).decode("ascii")

            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": payload,
                },
            }

            await self.ws.send_text(json.dumps(message))
            await asyncio.sleep(FRAME_DURATION)

    async def speak_text(self, text: str):
        """
        Use ElevenLabs TTS to synthesize `text` and send it back
        to the caller as μ-law frames over the Twilio WebSocket.
        """
        if not self.ws:
            logger.warning(f"[{self.call_sid}] No WebSocket set; cannot send audio.")
            return
        if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
            logger.error(f"[{self.call_sid}] ElevenLabs config missing.")
            return
        if not self.stream_sid:
            logger.error(f"[{self.call_sid}] streamSid missing; cannot send audio.")
            return
        if self.speaking:
            logger.info(f"[{self.call_sid}] Already speaking, skipping overlapping TTS.")
            return

        self.speaking = True
        logger.info(f"[{self.call_sid}] Speaking via ElevenLabs: {text}")

        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            }
            body = {
                "text": text,
                "model_id": ELEVENLABS_MODEL_ID,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "use_speaker_boost": True,
                },
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                mp3_bytes = resp.content

            # Decode MP3 -> 16-bit PCM 8kHz mono
            audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            audio = audio.set_channels(1).set_frame_rate(8000).set_sample_width(2)
            pcm16 = audio.raw_data

            # Convert to μ-law bytes for Twilio Media Streams
            mulaw = pcm16_to_mulaw(pcm16)

            # Send to Twilio with correct format + streamSid
            await self.send_twilio_media(mulaw)

        except Exception as e:
            logger.error(f"[{self.call_sid}] Error in speak_text/ElevenLabs: {e}")
        finally:
            self.speaking = False

    async def send_initial_greeting(self):
        """
        Play an automatic greeting once, via ElevenLabs, at call start.
        """
        if self.greeted:
            return
        self.greeted = True
        if INITIAL_GREETING.strip():
            await self.speak_text(INITIAL_GREETING)

    async def close(self):
        if self.gemini_session:
            await self.gemini_session.close()


call_states: Dict[str, CallState] = {}


def get_call_state(call_sid: str) -> CallState:
    if call_sid not in call_states:
        call_states[call_sid] = CallState(call_sid)
    return call_states[call_sid]


# ----------- 1. Twilio Answer URL -----------

@app.post("/twilio/voice-answer", response_class=PlainTextResponse)
async def twilio_voice_answer(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")

    logger.info(f"Incoming call: {call_sid}")
    logger.info(f"Using MEDIA_STREAM_WS_URL={MEDIA_STREAM_WS_URL}")

    vr = VoiceResponse()
    connect = vr.connect()
    connect.stream(
        url=MEDIA_STREAM_WS_URL
    )

    twiml_str = str(vr)
    logger.info(f"TwiML response for {call_sid}: {twiml_str}")

    return twiml_str


# ----------- 2. Twilio Media Stream WebSocket -----------

@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio WebSocket connected")

    call_sid: Optional[str] = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                call_sid = msg["start"]["callSid"]
                stream_sid = msg["start"]["streamSid"]

                state = get_call_state(call_sid)
                state.ws = ws
                state.stream_sid = stream_sid

                logger.info(f"[{call_sid}] Media stream started (streamSid={stream_sid})")

                # Start Gemini and immediately play ElevenLabs greeting
                await state.start_gemini()
                await state.send_initial_greeting()

            elif event == "media":
                if not call_sid:
                    continue

                payload_b64 = msg["media"]["payload"]
                mulaw_bytes = base64.b64decode(payload_b64)

                logger.info(f"[{call_sid}] Received media frame: {len(mulaw_bytes)} bytes")

                state = get_call_state(call_sid)
                await state.handle_audio_from_twilio(mulaw_bytes)

            elif event == "stop":
                if call_sid:
                    logger.info(f"[{call_sid}] Media stream stopped")
                    state = call_states.pop(call_sid, None)
                    if state:
                        await state.close()
                break

            else:
                logger.debug(f"Unknown Twilio event: {msg}")

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
        if call_sid:
            state = call_states.pop(call_sid, None)
            if state:
                await state.close()


# ----------- Local run -----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
