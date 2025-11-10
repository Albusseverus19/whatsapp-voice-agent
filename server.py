import os
import json
import base64
import queue
import threading
import time
import logging

from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Start, Stream

# ---------- CONFIG ----------

# IMPORTANT: set these in Render dashboard -> Environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

# Use current Gemini API (python-genai / google-genai style client)
# Docs: https://ai.google.dev/gemini-api/docs
from google import genai

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Model for STT + reasoning.
# 2.5 Flash supports audio + is cheap/fast; we instruct it to ONLY TRANSCRIBE for STT.
GEMINI_STT_MODEL = "gemini-2.5-flash"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"

# Audio from Twilio: 8kHz, mono, μ-law
TWILIO_AUDIO_MIME = "audio/mulaw;rate=8000"

SEGMENT_MS = 2000          # ~2s segments for STT
MAX_SEGMENTS_PER_CALL = 40 # safety: prevents unbounded growth
WORKER_IDLE_TIMEOUT = 15   # stop worker if no segments for N seconds

# ---------- LOGGING ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("voice-agent")

# ---------- FLASK / SOCK SETUP ----------

app = Flask(__name__)
sock = Sock(app)

# Per-call context
calls = {}  # { stream_sid: CallContext }


class CallContext:
    def __init__(self, stream_sid, ws):
        self.stream_sid = stream_sid
        self.ws = ws
        self.buffer = bytearray()
        self.segment_queue = queue.Queue()
        self.active = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self.last_activity = time.time()
        self.segment_count = 0
        self.worker_thread.start()

    # ------------ Worker ------------

    def _worker_loop(self):
        """
        Consume audio segments, run STT + reply, send TTS back.
        Dies automatically on timeout or stop.
        """
        log.info(f"[Call {self.stream_sid}] Worker started")
        while self.active:
            try:
                segment = self.segment_queue.get(timeout=1)
            except queue.Empty:
                # idle timeout
                if (time.time() - self.last_activity) > WORKER_IDLE_TIMEOUT:
                    log.info(f"[Call {self.stream_sid}] Worker idle timeout, stopping")
                    break
                continue

            if segment is None:
                log.info(f"[Call {self.stream_sid}] Worker got stop signal")
                break

            self.last_activity = time.time()
            text = safe_stt_georgian(segment, self.stream_sid)

            if not text:
                continue

            log.info(f"[Call {self.stream_sid}] STT text: {text}")

            reply = generate_reply(text, self.stream_sid)
            log.info(f"[Call {self.stream_sid}] Reply: {reply}")

            audio_bytes = tts_elevenlabs(reply)
            if audio_bytes:
                send_audio_to_twilio(self.ws, audio_bytes, self.stream_sid)

        self.active = False
        log.info(f"[Call {self.stream_sid}] Worker stopped")

    # ------------ Media handling ------------

    def add_media(self, chunk_bytes: bytes):
        """Accumulate media, cut into fixed-size segments for STT."""
        if not self.active:
            return

        self.buffer.extend(chunk_bytes)

        # 8000 bytes ~= 1 second for 8kHz 8bit μ-law
        bytes_per_ms = 8  # approx, safe for our purpose
        target_len = SEGMENT_MS * bytes_per_ms

        while len(self.buffer) >= target_len:
            if self.segment_count >= MAX_SEGMENTS_PER_CALL:
                log.warning(f"[Call {self.stream_sid}] Reached MAX_SEGMENTS_PER_CALL, dropping further audio")
                self.buffer.clear()
                return

            segment = self.buffer[:target_len]
            del self.buffer[:target_len]

            self.segment_queue.put(segment)
            self.segment_count += 1
            log.info(f"[Call {self.stream_sid}] Queued segment #{self.segment_count} ({SEGMENT_MS} ms)")

    def stop(self):
        self.active = False
        # Signal worker to exit
        try:
            self.segment_queue.put_nowait(None)
        except Exception:
            pass


# ---------- GEMINI STT ----------

def safe_stt_georgian(audio_bytes: bytes, stream_sid: str) -> str | None:
    """
    Send one short audio segment to Gemini for transcription.
    Return plain text (we'll log and use whatever we get for now).
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
                                "არ დაამატო არააფერი გარდა დიქტორის ნათქვამის ზუსტი გადმოცემისა. "
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
    Use Gemini to generate a short Georgian reply.
    You can enrich this with tools / context later.
    """
    try:
        resp = genai_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[{
                "role": "user",
                "parts": [{
                    "text": (
                        "შენ ხარ ქართული ენის ვირტუალური ასისტენტი სატელეფონო ზარებისთვის. "
                        "უპასუხე მოკლედ, გასაგებად და თავაზიანად.\n\n"
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
        log.error(f"[Call {stream_sid}] Reply generation error: {e}")
        return "ვწუხვარ, შევეჯახე ტექნიკურ შეცდომას. გთხოვთ სცადოთ თავიდან."


# ---------- ELEVENLABS TTS ----------

import requests

def tts_elevenlabs(text: str) -> bytes | None:
    """
    Convert reply text to 8kHz μ-law mono audio for Twilio.
    Make sure your ElevenLabs voice is configured; adjust URL/params if needed.
    """
    if not ELEVENLABS_API_KEY:
        log.error("ELEVENLABS_API_KEY not set")
        return None

    try:
        # Example ElevenLabs v1 text-to-speech call.
        # Adjust voice_id / format to your setup.
        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
        if not voice_id:
            log.error("ELEVENLABS_VOICE_ID not set")
            return None

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        # Request μ-law 8kHz mono so we can forward directly to Twilio
        payload = {
            "text": text,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.7,
            },
            "model_id": "eleven_monolingual_v1",
            "output_format": "ulaw_8000"
        }

        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        return r.content

    except Exception as e:
        log.error(f"TTS error: {e}")
        return None


# ---------- SEND AUDIO BACK TO TWILIO ----------

def send_audio_to_twilio(ws, audio_bytes: bytes, stream_sid: str):
    """
    Send audio back over the same WebSocket as a Twilio 'media' message.
    Twilio will play it to the caller.
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
        log.info(f"[Call {stream_sid}] Sent TTS audio to Twilio ({len(audio_bytes)} bytes)")
    except Exception as e:
        log.error(f"[Call {stream_sid}] Error sending audio to Twilio: {e}")


# ---------- TWILIO WEBHOOKS ----------

@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/voice", methods=["POST"])
def voice():
    """
    Twilio Voice webhook.

    Use <Connect><Stream> for bidirectional Media Streams:
    - Keeps the call open
    - Twilio sends continuous media events
    - We can send audio back on the same WebSocket
    """
    public_ws_url = os.environ.get(
        "PUBLIC_WS_URL",
        f"wss://{request.host}/media"
    )

    vr = VoiceResponse()

    # Bidirectional stream: call stays connected as long as WS is open
    connect = vr.connect()
    connect.stream(url=public_ws_url)

    # No <Say> here: our bot will speak via ElevenLabs over the stream.
    return Response(str(vr), mimetype="text/xml")


# ---------- MEDIA WEBSOCKET ----------

@sock.route("/media")
def media(ws):
    """
    Handles Twilio Media Streams over WebSocket.
    Twilio sends JSON messages as text frames.
    """
    stream_sid = None
    ctx: CallContext | None = None

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                # Client (Twilio) closed connection
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

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                log.info(f"[Twilio] Stream started: {stream_sid}")
                ctx = CallContext(stream_sid, ws)
                calls[stream_sid] = ctx

                # Send greeting over the stream via ElevenLabs TTS
                greeting = (
                    "გამარჯობა, თქვენ დაგიკავშირდათ ვირტუალური ასისტენტი. "
                    "გთხოვთ მოკლედ მითხრათ, რა საკითხზე გსურთ დახმარება?"
                )
                audio_bytes = tts_elevenlabs(greeting)
                if audio_bytes:
                    send_audio_to_twilio(ws, audio_bytes, stream_sid)
                else:
                    log.error("[Call %s] Failed to generate greeting TTS", stream_sid)


            elif event == "media" and ctx:
                media = msg.get("media", {})
                payload = media.get("payload")
                if payload:
                    chunk = base64.b64decode(payload)
                    ctx.add_media(chunk)

            elif event == "stop":
                log.info(f"[Twilio] Stream stopped event for {stream_sid}")
                break

            # (Optional) just log anything unexpected
            else:
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

# Gunicorn will look for 'app'
if __name__ == "__main__":
    # For local testing only
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
