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

# ================== CONFIG ==================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

if not ELEVENLABS_API_KEY:
    # We don't crash for this, but TTS won't work until it's set
    print("[WARN] ELEVENLABS_API_KEY is not set. Only Twilio <Say> greeting will work.")

# Google Gemini client (google-genai)
from google import genai

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ElevenLabs official SDK
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Models
GEMINI_STT_MODEL = "gemini-2.5-flash"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"

# ElevenLabs voice config
DEFAULT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")  # Jessica
DEFAULT_TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_v3")

# Twilio media format
TWILIO_AUDIO_MIME = "audio/mulaw;rate=8000"

SEGMENT_MS = 2000           # 2s audio chunks
MAX_SEGMENTS_PER_CALL = 40  # safety limit
WORKER_IDLE_TIMEOUT = 15    # stop worker if idle

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


class CallContext:
    def __init__(self, stream_sid, ws):
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
                        log.info(f"[Call {self.stream_sid}] Worker idle timeout")
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

        # ~8 bytes/ms for 8kHz 8bit μ-law
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

def safe_stt_georgian(audio_bytes: bytes, stream_sid: str) -> str | None:
    """
    Transcribe one short μ-law/8kHz segment with Gemini.
    """
    try:
        b64 = base64.b64encode(audio_bytes).decode("utf-8")

        resp = genai_client.models.generate_content(
            model=GEMINI_STT_MODEL,
            contents=[{
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "გთხოვ ზუსტად გადააქციე ეს აუდიო ქართულ ტექსტად. "
                            "არაფერი არ დაამატო. "
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
            }],
        )

        text = (resp.text or "").strip()
        if not text:
            log.info(f"[Call {stream_sid}] STT empty")
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
    Use ElevenLabs SDK to get μ-law 8kHz *headerless* audio (ulaw_8000),
    exactly what Twilio expects for Media Streams playback.
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

        # SDK returns raw bytes for this call
        if not isinstance(audio, (bytes, bytearray)):
            # If it's a generator/stream, join it
            data = bytearray()
            for chunk in audio:
                if isinstance(chunk, (bytes, bytearray)):
                    data.extend(chunk)
            audio = bytes(data)

        log.info(f"[Call {stream_sid}] TTS bytes: {len(audio)}")
        return audio

    except Exception as e:
        log.error(f"[Call {stream_sid}] TTS error: {e}")
        return None


# ================== SEND AUDIO TO TWILIO ==================

def send_audio_to_twilio(ws, audio_bytes: bytes, stream_sid: str):
    """
    Send μ-law/8000 audio back to Twilio over the same bidirectional Stream.
    """
    try:
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": b64
            }
        }
        ws.send(json.dumps(msg))
        log.info(f"[Call {stream_sid}] Sent {len(audio_bytes)} bytes to Twilio")
    except Exception as e:
        log.error(f"[Call {stream_sid}] Error sending audio to Twilio: {e}")


# ================== TWILIO WEBHOOKS ==================

@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/voice", methods=["POST"])
def voice():
    """
    Twilio Voice webhook:
      1. Twilio <Say> greeting (guaranteed audible).
      2. <Connect><Stream> to /media for real-time assistant.
    """
    public_ws_url = os.environ.get(
        "PUBLIC_WS_URL",
        f"wss://{request.host}/media"
    )

    log.info(f"[VOICE] Using PUBLIC_WS_URL={public_ws_url}")

    vr = VoiceResponse()

    # 1) Twilio-built-in greeting so call is never silent
    vr.say(
        "გამარჯობა, თქვენ დაუკავშირდით ვირტუალურ ასისტენტს. "
        "გთხოვთ დარჩეთ ხაზზე და ილაპარაკეთ ბიპის შემდეგ.",
        language="ka-GE",
        voice="woman"
    )

    # 2) Start bidirectional media stream
    connect = vr.connect()
    connect.stream(url=public_ws_url)

    return Response(str(vr), mimetype="text/xml")


# ================== MEDIA WEBSOCKET ==================

@sock.route("/media")
def media(ws):
    """
    Handle Twilio Media Streams over WebSocket.
    """
    stream_sid = None
    ctx: CallContext | None = None

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                log.info("[WS] None frame (close), breaking")
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

                # ElevenLabs greeting over the stream (optional, on top of <Say>)
                greeting = (
                    "გამარჯობა, თქვენ დაგიკავშირდათ ვირტუალური ასისტენტი. "
                    "კითხვა დაუსვით და გიპასუხებთ ქართულად."
                )
                audio_bytes = tts_elevenlabs(greeting, stream_sid)
                if audio_bytes:
                    send_audio_to_twilio(ws, audio_bytes, stream_sid)
                else:
                    log.error(f"[Call {stream_sid}] Greeting TTS failed")

            elif event == "media" and ctx:
                media_obj = msg.get("media", {})
                payload = media_obj.get("payload")
                if payload:
                    try:
                        chunk = base64.b64decode(payload)
                        ctx.add_media(chunk)
                    except Exception as e:
                        log.error(f"[Call {stream_sid}] Error decoding media: {e}")

            elif event == "stop":
                log.info(f"[Twilio] Stream stopped: {stream_sid}")
                break

            else:
                # connected, dtmf, mark, etc. (we just log)
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
