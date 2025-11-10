import os
import json
import base64
import queue
import threading
import time
import logging

import requests
from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse

# ---------- CONFIG ----------

# Set these in Render dashboard -> Environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

# Gemini client (google-genai / google-genai>=1.x)
from google import genai

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Models
GEMINI_STT_MODEL = "gemini-2.5-flash"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"

# Twilio Media Streams: 8kHz, mono, μ-law
TWILIO_AUDIO_MIME = "audio/mulaw;rate=8000"

# Segmentation / worker
SEGMENT_MS = 2000            # 2s per STT chunk
MAX_SEGMENTS_PER_CALL = 40   # safety cap
WORKER_IDLE_TIMEOUT = 15     # seconds of silence -> stop

# ---------- LOGGING ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("voice-agent")

# ---------- FLASK / SOCK ----------

app = Flask(__name__)
sock = Sock(app)

# Active calls: {stream_sid: CallContext}
calls = {}


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
            daemon=True
        )
        self.worker_thread.start()

    # ---------- Worker loop ----------

    def _worker_loop(self):
        log.info(f"[Call {self.stream_sid}] Worker started")
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

            # LLM reply
            reply = generate_reply(text, self.stream_sid)
            log.info(f"[Call {self.stream_sid}] Reply: {reply}")

            # TTS
            audio_bytes = tts_elevenlabs(reply)
            if audio_bytes:
                send_audio_to_twilio(self.ws, audio_bytes, self.stream_sid)

        self.active = False
        log.info(f"[Call {self.stream_sid}] Worker stopped")

    # ---------- Media handling ----------

    def add_media(self, chunk_bytes: bytes):
        """Collect inbound media and slice into SEGMENT_MS chunks for STT."""
        if not self.active:
            return

        self.buffer.extend(chunk_bytes)

        # 8kHz μ-law: 8000 bytes/sec → 8 bytes/ms
        bytes_per_ms = 8
        target_len = SEGMENT_MS * bytes_per_ms

        while len(self.buffer) >= target_len:
            if self.segment_count >= MAX_SEGMENTS_PER_CALL:
                log.warning(
                    f"[Call {self.stream_sid}] Reached MAX_SEGMENTS_PER_CALL, "
                    "dropping further audio"
                )
                self.buffer.clear()
                return

            segment = self.buffer[:target_len]
            del self.buffer[:target_len]

            self.segment_queue.put(segment)
            self.segment_count += 1
            log.info(
                f"[Call {self.stream_sid}] Queued segment "
                f"#{self.segment_count} ({SEGMENT_MS} ms)"
            )

    def stop(self):
        self.active = False
        try:
            self.segment_queue.put_nowait(None)
        except Exception:
            pass


# ---------- GEMINI STT ----------

def safe_stt_georgian(audio_bytes: bytes, stream_sid: str) -> str | None:
    """
    STT via Gemini. Keep it simple; log what we get.
    """
    try:
        b64 = base64.b64encode(audio_bytes).decode("utf-8")

        resp = genai_client.models.generate_content(
            model=GEMINI_STT_MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "გთხოვ ზუსტად გადააქციე ეს აუდიო ქართულ ტექსტად. "
                                "არ დაამატო არაფერი გარდა დიქტორის ნათქვამის ზუსტი გადმოცემისა. "
                                "თუ არ გესმის, დაწერე მხოლოდ: 'ვერ გავიგე'."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": TWILIO_AUDIO_MIME,
                                "data": b64,
                            }
                        },
                    ],
                }
            ],
        )

        text = (resp.text or "").strip()
        if not text:
            log.info(f"[Call {stream_sid}] STT: empty result")
            return None

        log.info(f"[Call {stream_sid}] STT raw: {text}")
        return text

    except Exception as e:
        log.error(f"[Call {stream_sid}] STT error: {e}")
        return None


# ---------- GEMINI CHAT ----------

def generate_reply(user_text: str, stream_sid: str) -> str:
    """
    Short, polite Georgian reply.
    """
    try:
        resp = genai_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "შენ ხარ ქართული ენის ვირტუალური ასისტენტი სატელეფონო ზარებისთვის. "
                                "უპასუხე მარტივად, მოკლედ, გასაგებად და თავაზიანად.\n\n"
                                f"მომხმარებლის ტექსტი: {user_text}"
                            )
                        }
                    ],
                }
            ],
        )
        text = (resp.text or "").strip()
        if not text:
            return (
                "ვწუხვარ, ვერ გავიგე ზუსტად. "
                "გთხოვთ განმეორებით მითხრათ, რა გსურთ."
            )
        return text

    except Exception as e:
        log.error(f"[Call {stream_sid}] Reply generation error: {e}")
        return (
            "ვწუხვარ, ტექნიკური შეცდომა მოხდა. "
            "გთხოვთ სცადოთ კიდევ ერთხელ."
        )


# ---------- ELEVENLABS TTS ----------

def tts_elevenlabs(text: str) -> bytes | None:
    """
    Get μ-law 8kHz bytes directly from ElevenLabs for Twilio.
    """
    if not ELEVENLABS_API_KEY:
        log.error("ELEVENLABS_API_KEY not set")
        return None

    voice_id = os.environ.get(
        "ELEVENLABS_VOICE_ID",
        "cgSgspJ2msm6clMCkdW9"  # Jessica default
    )
    if not voice_id:
        log.error("ELEVENLABS_VOICE_ID missing even after default fallback")
        return None

    tts_model = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_v3")

    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        }
        payload = {
            "text": text,
            "model_id": tts_model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.7,
            },
            # ElevenLabs docs: ulaw_8000 = raw 8kHz μ-law, perfect for Twilio
            "output_format": "ulaw_8000",
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        audio_bytes = resp.content

        if not audio_bytes:
            log.error("ElevenLabs TTS returned empty body")
            return None

        log.info(f"[TTS] Got {len(audio_bytes)} bytes from ElevenLabs")
        return audio_bytes

    except Exception as e:
        log.error(f"ElevenLabs TTS error: {e}")
        return None


# ---------- SEND AUDIO BACK TO TWILIO ----------

def send_audio_to_twilio(ws, audio_bytes: bytes, stream_sid: str):
    """
    Send μ-law 8kHz audio to Twilio as 'media' events.
    Chunked into 20ms frames (160 bytes).
    """
    if not audio_bytes:
        log.warning(f"[Call {stream_sid}] No audio to send")
        return

    frame_size = 160  # 20ms at 8000 bytes/sec
    total = len(audio_bytes)
    pos = 0
    frames = 0

    try:
        while pos < total:
            chunk = audio_bytes[pos:pos + frame_size]
            pos += frame_size
            if not chunk:
                break

            # pad last chunk with μ-law silence (0xFF)
            if len(chunk) < frame_size:
                chunk += b"\xff" * (frame_size - len(chunk))

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

        log.info(
            f"[Call {stream_sid}] Sent {frames} TTS frames ({total} bytes)"
        )

    except Exception as e:
        log.error(f"[Call {stream_sid}] Error sending TTS to Twilio: {e}")


# ---------- TWILIO WEBHOOKS ----------

@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/voice", methods=["POST"])
def voice():
    """
    Twilio Voice webhook: start bidirectional Media Stream.
    Greeting + replies are handled over WebSocket via ElevenLabs.
    """
    public_ws_url = os.environ.get(
        "PUBLIC_WS_URL",
        f"wss://{request.host}/media"
    )

    log.info(f"[VOICE] Using PUBLIC_WS_URL={public_ws_url}")

    vr = VoiceResponse()
    connect = vr.connect()
    connect.stream(url=public_ws_url)

    return Response(str(vr), mimetype="text/xml")


# ---------- MEDIA WEBSOCKET ----------

@sock.route("/media")
def media(ws):
    """
    Handle Twilio Media Streams JSON over WebSocket.
    """
    stream_sid = None
    ctx: CallContext | None = None

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                log.info("[WS] Received None (close), breaking loop")
                break

            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"[WS] Non-JSON frame: {repr(raw)[:120]}")
                continue

            event = msg.get("event")
            log.info(f"[WS] Incoming event: {event} | msg: {str(msg)[:200]}")

            # --- Start ---
            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                log.info(f"[Twilio] Stream started: {stream_sid}")
                ctx = CallContext(stream_sid, ws)
                calls[stream_sid] = ctx

                # Greeting via ElevenLabs
                greeting = (
                    "გამარჯობა, თქვენ დაუკავშირდით ვირტუალურ ასისტენტს. "
                    "გთხოვთ მითხრათ, რა საკითხზე გსურთ დახმარება?"
                )
                audio_bytes = tts_elevenlabs(greeting)
                if audio_bytes:
                    send_audio_to_twilio(ws, audio_bytes, stream_sid)
                    log.info(f"[Call {stream_sid}] Sent greeting via ElevenLabs")
                else:
                    log.error(f"[Call {stream_sid}] Failed to generate greeting TTS")

            # --- Inbound media from caller ---
            elif event == "media" and ctx:
                media_obj = msg.get("media", {})
                payload = media_obj.get("payload")
                if payload:
                    chunk = base64.b64decode(payload)
                    ctx.add_media(chunk)

            # --- Stop ---
            elif event == "stop":
                log.info(f"[Twilio] Stream stopped event for {stream_sid}")
                break

            else:
                # includes "connected" etc.
                log.info(f"[WS] Unhandled event type: {event}")

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


# ---------- GUNICORN ENTRY ----------

if __name__ == "__main__":
    # Local debugging only
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=True,
    )
