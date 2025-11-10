import os
import json
import base64
import queue
import threading
import time
import logging

from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse

import requests
from google import genai
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# ================== CONFIG ==================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

genai_client = genai.Client(api_key=GEMINI_API_KEY)

eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Models
GEMINI_STT_MODEL = "gemini-1.5-pro"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"

DEFAULT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")  # Jessica
DEFAULT_TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_v3")

# Twilio media: 8kHz, mono, μ-law
TWILIO_SAMPLE_RATE = 8000
TWILIO_AUDIO_MIME = "audio/mulaw;rate=8000"

# Segmentation / worker
SEGMENT_MS = 800             # 0.8s chunks → more responsive
MAX_SEGMENTS_PER_CALL = 40
WORKER_IDLE_TIMEOUT = 15

# ================== LOGGING ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("voice-agent")

# ================== APP / SOCK ==================

app = Flask(__name__)
sock = Sock(app)

# streamSid -> CallContext
calls = {}


# ================== μ-LAW -> PCM16 ==================

def mulaw_to_linear16(mulaw_bytes: bytes) -> bytes:
    """
    Convert 8-bit μ-law bytes (G.711) to 16-bit PCM (little-endian).
    This lets us send audio/pcm to Gemini, which it supports.
    """
    out = bytearray()
    for b in mulaw_bytes:
        u = ~b & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        # Standard G.711 μ-law decode
        sample = ((mantissa << 4) + 0x08) << exponent
        sample += 0x84
        if sign:
            sample = -sample
        # clamp just in case
        if sample > 32767:
            sample = 32767
        elif sample < -32768:
            sample = -32768
        out.extend(sample.to_bytes(2, byteorder="little", signed=True))
    return bytes(out)


# ================== CALL CONTEXT ==================

class CallContext:
    def __init__(self, stream_sid: str, ws):
        self.stream_sid = stream_sid
        self.ws = ws
        self.buffer = bytearray()
        self.segment_queue = queue.Queue()
        self.active = True
        self.last_activity = time.time()
        self.segment_count = 0

        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"worker-{stream_sid}",
            daemon=True,
        )
        self.worker_thread.start()

    # ---------- Worker loop ----------

    def _worker_loop(self):
        log.info(f"[Call {self.stream_sid}] Worker started")
        try:
            while self.active:
                try:
                    segment = self.segment_queue.get(timeout=1)
                except queue.Empty:
                    if (time.time() - self.last_activity) > WORKER_IDLE_TIMEOUT:
                        log.info(f"[Call {self.stream_sid}] Worker idle timeout, stopping")
                        break
                    continue

                if segment is None:
                    log.info(f"[Call {self.stream_sid}] Worker got stop signal")
                    break

                self.last_activity = time.time()

                # STT
                text = safe_stt_georgian(segment, self.stream_sid)
                if not text:
                    continue

                log.info(f"[Call {self.stream_sid}] STT text: {text}")

                # Chat
                reply = generate_reply(text, self.stream_sid)
                log.info(f"[Call {self.stream_sid}] Reply: {reply}")

                # TTS
                audio_bytes = tts_elevenlabs(reply, self.stream_sid)
                if audio_bytes:
                    send_audio_to_twilio(self.ws, audio_bytes, self.stream_sid)

        except Exception as e:
            log.error(f"[Call {self.stream_sid}] Worker fatal error: {e}")
        finally:
            self.active = False
            log.info(f"[Call {self.stream_sid}] Worker stopped")

    # ---------- Media handling ----------

    def add_media(self, chunk_bytes: bytes):
        if not self.active:
            return

        self.buffer.extend(chunk_bytes)

        # 8kHz μ-law -> 8000 bytes/sec -> 8 bytes/ms
        bytes_per_ms = 8
        target_len = SEGMENT_MS * bytes_per_ms

        while len(self.buffer) >= target_len:
            if self.segment_count >= MAX_SEGMENTS_PER_CALL:
                log.warning(f"[Call {self.stream_sid}] MAX_SEGMENTS_PER_CALL reached; dropping further audio")
                self.buffer.clear()
                return

            segment = self.buffer[:target_len]
            del self.buffer[:target_len]

            self.segment_queue.put(segment)
            self.segment_count += 1
            log.info(f"[Call {self.stream_sid}] Queued segment #{self.segment_count} ({SEGMENT_MS} ms)")

    def stop(self):
        self.active = False
        try:
            self.segment_queue.put_nowait(None)
        except Exception:
       	    pass


# ================== GEMINI STT ==================

def safe_stt_georgian(mulaw_bytes: bytes, stream_sid: str) -> str | None:
    """
    Convert Twilio μ-law -> PCM16 and send to Gemini 1.5 Pro as audio/pcm.
    """
    try:
        pcm16 = mulaw_to_linear16(mulaw_bytes)
        b64 = base64.b64encode(pcm16).decode("utf-8")

        resp = genai_client.models.generate_content(
            model=GEMINI_STT_MODEL,
            contents=[{
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "გთხოვ ზუსტად გადააქციე ეს აუდიო ქართულ ტექსტად. "
                            "მომეცი მხოლოდ ის, რაც ითქვა. "
                            "თუ არ გესმის, დაწერე მხოლოდ: 'ვერ გავიგე'."
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "audio/pcm",
                            "data": b64,
                        }
                    },
                ],
            }],
        )

        text = (resp.text or "").strip()
        if not text:
            log.info(f"[Call {stream_sid}] STT: empty")
            return None

        log.info(f"[Call {stream_sid}] STT raw: {text}")
        return text

    except Exception as e:
        log.error(f"[Call {stream_sid}] STT error: {e}")
        return None


# ================== GEMINI CHAT ==================

def generate_reply(user_text: str, stream_sid: str) -> str:
    try:
        resp = genai_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[{
                "role": "user",
                "parts": [{
                    "text": (
                        "შენ ხარ ქართული ენის ვირტუალური ასისტენტი სატელეფონო ზარებისთვის. "
                        "უპასუხე ძალიან მოკლედ, გასაგებად და თავაზიანად.\n\n"
                        f"მომხმარებლის ტექსტი: {user_text}"
                    )
                }]
            }],
        )
        text = (resp.text or "").strip()
        if not text:
            return "ვწუხვარ, ვერ გავიგე ზუსტად. გთხოვთ განმეორებით მითხრათ, რა გსურთ."
        return text
    except Exception as e:
        log.error(f"[Call {stream_sid}] Chat error: {e}")
        return "ვწუხვარ, ტექნიკური პრობლემა მოხდა. გთხოვთ სცადოთ თავიდან."


# ================== ELEVENLABS TTS ==================

def tts_elevenlabs(text: str, stream_sid: str) -> bytes | None:
    """
    Use ElevenLabs to get μ-law/8000 *raw* audio for Twilio.
    """
    if not eleven_client:
        log.error(f"[Call {stream_sid}] ELEVENLABS_API_KEY not configured")
        return None

    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)
    model_id = os.environ.get("ELEVENLABS_TTS_MODEL", DEFAULT_TTS_MODEL)

    try:
        audio = eleven_client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text,
            output_format="ulaw_8000",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.7,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        if isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
        else:
            # handle generator/stream
            buf = bytearray()
            for chunk in audio:
                if isinstance(chunk, (bytes, bytearray)):
                    buf.extend(chunk)
            data = bytes(buf)

        if not data:
            log.error(f"[Call {stream_sid}] TTS empty")
            return None

        log.info(f"[Call {stream_sid}] TTS bytes: {len(data)}")
        return data

    except Exception as e:
        log.error(f"[Call {stream_sid}] TTS error: {e}")
        return None


# ================== SEND AUDIO BACK TO TWILIO ==================

def send_audio_to_twilio(ws, audio_bytes: bytes, stream_sid: str):
    """
    Send μ-law/8000 audio to Twilio as 'media' events.
    Chunk into frames Twilio can play.
    """
    if not audio_bytes:
        return

    frame_size = 160  # 20ms @ 8000 bytes/sec
    total = len(audio_bytes)
    pos = 0
    frames = 0

    try:
        while pos < total:
            chunk = audio_bytes[pos:pos + frame_size]
            pos += frame_size
            if not chunk:
                break

            if len(chunk) < frame_size:
                chunk += b"\xff" * (frame_size - len(chunk))  # μ-law silence padding

            payload = base64.b64encode(chunk).decode("ascii")
            msg = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": payload
                }
            }
            ws.send(json.dumps(msg))
            frames += 1

        log.info(f"[Call {stream_sid}] Sent {frames} TTS frames ({total} bytes)")
    except Exception as e:
        log.error(f"[Call {stream_sid}] Error sending TTS to Twilio: {e}")


# ================== TWILIO WEBHOOKS ==================

@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/voice", methods=["POST"])
def voice():
    """
    Answer call:
      - Twilio <Say> greeting so it's never silent.
      - <Connect><Stream> to /media for realtime assistant.
    """
    public_ws_url = os.environ.get(
        "PUBLIC_WS_URL",
        f"wss://{request.host}/media"
    )
    log.info(f"[VOICE] Using PUBLIC_WS_URL={public_ws_url}")

    vr = VoiceResponse()

    vr.say(
        "გამარჯობა, თქვენ დაუკავშირდით ვირტუალურ ასისტენტს. "
        "ბიპის შემდეგ ილაპარაკეთ, და გიპასუხებთ.",
        language="ka-GE",
        voice="woman"
    )

    connect = vr.connect()
    connect.stream(url=public_ws_url)

    return Response(str(vr), mimetype="text/xml")


# ================== MEDIA WEBSOCKET ==================

@sock.route("/media")
def media(ws):
    stream_sid = None
    ctx: CallContext | None = None

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                log.info("[WS] Received None (close), breaking")
                break
            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"[WS] Non-JSON frame: {repr(raw)[:120]}")
                continue

            event = msg.get("event")
            log.info(f"[WS] Incoming event: {event} | msg: {str(msg)[:240]}")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                log.info(f"[Twilio] Stream started: {stream_sid}")
                ctx = CallContext(stream_sid, ws)
                calls[stream_sid] = ctx

                # Optional: ElevenLabs greeting over stream
                greeting = (
                    "გამარჯობა, მე გისმენ. გთხოვთ მითხრათ, რით შემიძლია დაგეხმაროთ."
                )
                audio_bytes = tts_elevenlabs(greeting, stream_sid)
                if audio_bytes:
                    send_audio_to_twilio(ws, audio_bytes, stream_sid)

            elif event == "media" and ctx:
                media_obj = msg.get("media", {})
                payload = media_obj.get("payload")
                if payload:
                    try:
                        chunk = base64.b64decode(payload)
                        ctx.add_media(chunk)
                    except Exception as e:
                        log.error(f"[Call {ctx.stream_sid}] Error decoding media: {e}")

            elif event == "stop":
                log.info(f"[Twilio] Stream stopped: {stream_sid}")
                break

            else:
                log.info(f"[WS] Unhandled event: {event}")

    except Exception as e:
        log.error(f"[WS] Error: {e}")

    finally:
        if ctx:
            ctx.stop()
            calls.pop(ctx.stream_sid, None)
        try:
            ws.close()
        except Exception:
            pass
        log.info("[WS] Closing sockets")


# ================== LOCAL ENTRYPOINT ==================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=True
    )
