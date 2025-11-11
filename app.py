import os
import io
import json
import base64
import logging
import asyncio

from typing import Dict, Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse
from pydub import AudioSegment

from config import MEDIA_STREAM_WS_URL, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID
from gemini_client import GeminiLiveSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-eleven-gemini")

app = FastAPI()


# ----------- μ-law encoder (no audioop, works on 3.13+) -----------

MU_MAX = 32635
MU_BIAS = 0x84
MU = 255

def linear16_sample_to_mulaw(sample: int) -> int:
    """
    Convert one 16-bit PCM sample to 8-bit μ-law.
    """
    sign = 0
    if sample < 0:
        sample = -sample
        sign = 0x80

    if sample > MU_MAX:
        sample = MU_MAX

    sample += MU_BIAS

    exponent = 7
    mask = 0x4000
    for exp in range(7):
        if sample & mask:
            exponent = exp
            break
        mask >>= 1

    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw


def pcm16_to_mulaw(pcm16: bytes) -> bytes:
    """
    Convert linear16 PCM bytes (little-endian) to μ-law bytes.
    """
    out = bytearray()
    for i in range(0, len(pcm16), 2):
        if i + 1 >= len(pcm16):
            break
        sample = int.from_bytes(pcm16[i:i+2], "little", signed=True)
        out.append(linear16_sample_to_mulaw(sample))
    return bytes(out)


# ----------- Call state -----------

class CallState:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.ws: Optional[WebSocket] = None
        self.gemini_session: Optional[GeminiLiveSession] = None
        self.speaking = False  # simple lock to avoid overlap

    async def start_gemini(self):
        if self.gemini_session is None:
            async def on_final_text(text: str):
                """
                Called when Gemini has a complete reply.
                Here we trigger ElevenLabs TTS and send audio back to Twilio.
                """
                await self.speak_text(text)

            self.gemini_session = GeminiLiveSession(
                call_sid=self.call_sid,
                on_final_text=on_final_text,
            )
            await self.gemini_session.start()

    async def handle_audio_from_twilio(self, mulaw_bytes: bytes):
        """
        Convert Twilio μ-law -> PCM16 and stream to Gemini.
        """
        if not self.gemini_session:
            await self.start_gemini()

        # Twilio: 8-bit μ-law, 8kHz mono.
        audio = AudioSegment(
            data=mulaw_bytes,
            sample_width=1,
            frame_rate=8000,
            channels=1,
        )
        pcm_audio = audio.set_sample_width(2)
        pcm16 = pcm_audio.raw_data

        await self.gemini_session.send_audio_chunk(pcm16_bytes=pcm16, sample_rate=8000)

    async def speak_text(self, text: str):
        """
        Use ElevenLabs v3 TTS to synthesize `text` and send it back
        to the caller as μ-law frames over the Twilio WebSocket.
        """
        if not self.ws:
            logger.warning(f"[{self.call_sid}] No WebSocket set; cannot send audio.")
            return
        if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
            logger.error(f"[{self.call_sid}] ElevenLabs config missing.")
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

            # Send in chunks (~20ms frames) to Twilio
            frame_samples = 160  # 20ms at 8kHz
            frame_size = frame_samples  # since μ-law is 1 byte per sample

            for i in range(0, len(mulaw), frame_size):
                frame = mulaw[i:i+frame_size]
                if not frame:
                    continue
                b64 = base64.b64encode(frame).decode("utf-8")
                await self.ws.send_text(json.dumps({
                    "event": "media",
                    "media": {"payload": b64}
                }))
                # small delay to approximate natural playback pacing
                await asyncio.sleep(0.02)

        except Exception as e:
            logger.error(f"[{self.call_sid}] Error in speak_text/ElevenLabs: {e}")
        finally:
            self.speaking = False

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
                state = get_call_state(call_sid)
                state.ws = ws
                logger.info(f"[{call_sid}] Media stream started")

                await state.start_gemini()

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
    import asyncio
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
