import os
import json
import time
import base64
import threading
import io
import wave
import requests

from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Connect

from elevenlabs.client import ElevenLabs
import google.generativeai as genai

app = Flask(__name__)
sock = Sock(app)

# === Config from environment ===
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not ELEVENLABS_API_KEY:
    print("[Config] ELEVENLABS_API_KEY is not set!")
if not GEMINI_API_KEY:
    print("[Config] GEMINI_API_KEY is not set!")

# === Initialize external clients ===
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None

# Voice & model settings
VOICE_ID = "cgSgspJ2msm6clMCkdW9"  # Jessica
TTS_MODEL = "eleven_v3"
STT_MODEL = "scribe_v1"

# Segmentation tuning
MIN_UTTERANCE_MS = 700
SILENCE_GAP_MS = 900
CHECK_INTERVAL = 0.1

# === Mu-law decoder (no audioop) ===

MU_LAW_BIAS = 0x84
MU_LAW_CLIP = 32635


def _ulaw_byte_to_linear(sample: int) -> int:
    sample = ~sample & 0xFF
    sign = sample & 0x80
    exponent = (sample >> 4) & 0x07
    mantissa = sample & 0x0F
    magnitude = ((mantissa << 4) + 0x08) << exponent
    magnitude = magnitude - MU_LAW_BIAS
    if magnitude > MU_LAW_CLIP:
        magnitude = MU_LAW_CLIP
    if sign != 0:
        magnitude = -magnitude
    return magnitude


def ulaw_to_linear16(data: bytes) -> bytes:
    out = bytearray()
    for b in data:
        s = _ulaw_byte_to_linear(b)
        out.append(s & 0xFF)
        out.append((s >> 8) & 0xFF)
    return bytes(out)


def elevenlabs_stt(wav_bytes: bytes, language_code: str = "ka") -> str:
    """
    Call ElevenLabs Speech-to-Text over HTTP directly.
    This avoids relying on `client.speech_to_text` attribute,
    which is missing in your deployed environment.
    """
    if not ELEVENLABS_API_KEY:
        print("[STT] ELEVENLABS_API_KEY not set")
        return ""

    try:
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
        }
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }
        data = {
            "model_id": STT_MODEL,
            "language_code": language_code,
        }

        resp = requests.post(url, headers=headers, files=files, data=data, timeout=20)

        if resp.status_code != 200:
            print(f"[STT] HTTP {resp.status_code}: {resp.text[:300]}")
            return ""

        try:
            j = resp.json()
        except Exception as e:
            print(f"[STT] JSON decode error: {e}, body: {resp.text[:300]}")
            return ""

        text = (j.get("text") or "").strip()
        if text:
            print(f"[STT] -> {text}")
        else:
            print("[STT] Empty text in response")

        return text

    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""


@app.route("/voice", methods=["POST"])
def voice():
    """
    Twilio webhook for incoming WhatsApp voice calls.
    Starts a bidirectional Media Stream to /media-stream.
    """
    vr = VoiceResponse()

    base = request.host_url.rstrip("/")
    if base.startswith("http://"):
        ws_base = "wss://" + base[len("http://"):]
    elif base.startswith("https://"):
        ws_base = "wss://" + base[len("https://"):]
    else:
        ws_base = "wss://" + base

    connect = Connect()
    connect.stream(url=f"{ws_base}/media-stream")
    vr.append(connect)

    return Response(str(vr), mimetype="text/xml")


@sock.route("/media-stream")
def media_stream(twilio_ws):
    """
    Twilio Media Streams handler.

    Flow:
      - On start: async Georgian greeting
      - Then:
        - Receive caller audio (mu-law 8k) -> buffer
        - Worker segments by duration
        - For each utterance:
            STT (ElevenLabs Scribe, ka)
            -> Gemini 2.5 Flash (Georgian)
            -> TTS (Eleven v3, Jessica, ulaw_8000)
            -> back to Twilio as media events
    """
    print("[Twilio] WebSocket connected")

    stream_sid = None
    call_sid = None

    audio_buffer = bytearray()
    processing = False
    closed = False
    conversation_history = []
    greeted = False

    def log(msg: str):
        print(f"[Call {call_sid or '?'}] {msg}")

    def mulaw_buffer_duration_ms(buf: bytes) -> int:
        if not buf:
            return 0
        return int(len(buf) / 8000 * 1000)  # 8kHz, 1 byte/sample

    def stream_tts_text(text: str):
        """
        TTS (eleven_v3, ulaw_8000) -> send to Twilio as media messages.
        """
        nonlocal stream_sid

        if not eleven_client:
            log("No ElevenLabs client; cannot TTS.")
            return
        if not stream_sid:
            log("No streamSid; cannot send media.")
            return

        try:
            audio_iter = eleven_client.text_to_speech.convert(
                voice_id=VOICE_ID,
                model_id=TTS_MODEL,
                text=text,
                output_format="ulaw_8000",
            )
            tts_bytes = b"".join(chunk for chunk in audio_iter)
            if not tts_bytes:
                log("Empty TTS output.")
                return

            frame_size = 320  # 20ms @ 8kHz
            idx = 0
            while idx < len(tts_bytes):
                chunk = tts_bytes[idx: idx + frame_size]
                idx += frame_size

                payload = base64.b64encode(chunk).decode("utf-8")
                media_msg = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }
                try:
                    twilio_ws.send(json.dumps(media_msg))
                except Exception as e:
                    log(f"TTS send error: {e}")
                    break

            log(f"TTS sent: {text}")

        except Exception as e:
            log(f"TTS error: {e}")

    def run_pipeline_on_buffer(buf: bytes):
        """
        One utterance:
          mu-law -> PCM16 WAV -> ElevenLabs STT -> Gemini -> TTS -> Twilio
        """
        nonlocal conversation_history

        if not buf:
            return
        if not gemini_model:
            log("Gemini model missing; cannot respond.")
            return

        # 1) mu-law -> PCM16 WAV
        try:
            pcm16 = ulaw_to_linear16(buf)
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(pcm16)
            wav_bytes = wav_io.getvalue()
        except Exception as e:
            log(f"WAV encode error: {e}")
            return

        # 2) STT via ElevenLabs HTTP
        log("STT (ka) starting...")
        user_text = elevenlabs_stt(wav_bytes, language_code="ka")
        if not user_text:
            log("STT empty; skip this chunk.")
            return

        log(f"User 🇬🇪: {user_text}")

        # 3) Gemini reasoning in Georgian
        try:
            prompt_parts = [
                "შენ ხარ ქართული საკონტაქტო ცენტრის ვირტუალური ოპერატორი.",
                "საუბრობ მხოლოდ ქართულად, ბუნებრივი, პროფესიული და მეგობრული ტონით.",
                "პასუხობ მოკლედ და გასაგებად. თითო პასუხში ერთ მთავარ იდეას ხსნი.",
                "ქვემოთ არის დიალოგის ისტორია:",
            ]
            for turn in conversation_history[-10:]:
                role = "მომხმარებელი" if turn["role"] == "user" else "ოპერატორი"
                prompt_parts.append(f"{role}: {turn['content']}")
            prompt_parts.append(f"მომხმარებელი: {user_text}")
            prompt_parts.append("ოპერატორი (მოკლე, გასაგები პასუხი ქართულად):")

            full_prompt = "\n".join(prompt_parts)

            log("Gemini thinking...")
            gemini_resp = gemini_model.generate_content(full_prompt)
            ai_text = (getattr(gemini_resp, "text", "") or "").strip()
            if not ai_text:
                ai_text = "უკაცრავად, ხარვეზი წარმოიქმნა. შეგიძლიათ გაიმეოროთ კითხვა?"

            log(f"AI 🤖: {ai_text}")

            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": ai_text})

            # 4) TTS back to caller
            stream_tts_text(ai_text)

        except Exception as e:
            log(f"Pipeline error: {e}")

    def buffer_worker():
        """
        Simple segmentation worker:
        - Every ~2s of audio currently in buffer => treat as one utterance.
        - Run STT -> Gemini -> TTS on that segment.
        - On close, flush remaining.
        """
        nonlocal audio_buffer, processing, closed

        SEGMENT_MS = 2000

        while not closed:
            try:
                if not processing:
                    dur_ms = mulaw_buffer_duration_ms(audio_buffer)
                    if dur_ms >= SEGMENT_MS:
                        processing = True
                        segment = bytes(audio_buffer)
                        audio_buffer = bytearray()

                        print(f"[Worker] Processing segment of {dur_ms} ms")
                        run_pipeline_on_buffer(segment)

                        processing = False

                time.sleep(0.2)

            except Exception as e:
                print(f"[Worker] Error: {e}")
                time.sleep(0.5)

        # Flush remaining on close
        try:
            if audio_buffer:
                dur_ms = mulaw_buffer_duration_ms(audio_buffer)
                print(f"[Worker] Flushing final segment of {dur_ms} ms")
                run_pipeline_on_buffer(bytes(audio_buffer))
        except Exception as e:
            print(f"[Worker] Flush error: {e}")

    worker_thread = threading.Thread(target=buffer_worker, daemon=True)
    worker_thread.start()

    # === Main Twilio WS loop ===
    try:
        while True:
            raw = twilio_ws.receive()
            if raw is None:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print("[Twilio] Invalid JSON")
                continue

            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"].get("callSid")
                print(f"[Twilio] Stream started: {stream_sid} (Call: {call_sid})")

                # Greeting (async)
                if eleven_client and not greeted:
                    greeted = True
                    greeting = (
                        "გამარჯობა, თქვენ დაგიკავშირდათ ვირტუალური ასისტენტი. "
                        "გთხოვთ მოკლედ მითხრათ, რა საკითხზე გსურთ დახმარება?"
                    )
                    threading.Thread(
                        target=stream_tts_text,
                        args=(greeting,),
                        daemon=True,
                    ).start()

            elif event == "media":
                # Incoming caller audio
                payload_b64 = msg["media"]["payload"]
                try:
                    chunk = base64.b64decode(payload_b64)
                    audio_buffer.extend(chunk)
                    log(f"Media chunk received: {len(chunk)} bytes")
                except Exception as e:
                    print(f"[Twilio] Media decode error: {e}")

            elif event == "stop":
                print("[Twilio] Stream stopped")
                break

            elif event == "clear":
                audio_buffer = bytearray()

    except Exception as e:
        print(f"[WS] Error: {e}")

    finally:
        closed = True
        print("[WS] Closing sockets")
        try:
            twilio_ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Local dev only; Render will run via gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
