import os
import json
import time
import base64
import threading
import io
import wave

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

# Voice & model settings (your tested setup)
VOICE_ID = "cgSgspJ2msm6clMCkdW9"  # Jessica
TTS_MODEL = "eleven_v3"
STT_MODEL = "scribe_v1"

# Silence / segmentation tuning
MIN_UTTERANCE_MS = 700       # minimum speech length to process
SILENCE_GAP_MS = 900         # ms of silence to close an utterance
CHECK_INTERVAL = 0.1         # worker loop sleep


# === Mu-law (G.711) decoder (no audioop, works on Python 3.13) ===

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


@app.route("/voice", methods=["POST"])
def voice():
    """
    Twilio webhook for incoming WhatsApp voice calls.
    Returns TwiML that starts a bidirectional Media Stream to /media-stream.
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
      - Receive caller audio (mulaw 8k) from Twilio
      - Segment based on silence
      - For each utterance:
          STT: ElevenLabs Scribe v1 (ka)
          LLM: Gemini 2.5 Flash (Georgian)
          TTS: ElevenLabs eleven_v3 (Jessica, ulaw_8000)
          Stream audio back to Twilio
    """
    print("[Twilio] WebSocket connected")

    stream_sid = None
    call_sid = None

    audio_buffer = bytearray()
    last_audio_time = 0.0
    processing = False
    closed = False
    conversation_history = []

    def log(msg: str):
        print(f"[Call {call_sid or '?'}] {msg}")

    def mulaw_buffer_duration_ms(buf: bytes) -> int:
        if not buf:
            return 0
        # 8000 samples/sec, 1 byte/sample
        return int(len(buf) / 8000 * 1000)

    def run_pipeline_on_buffer(buf: bytes):
        nonlocal conversation_history

        if not buf:
            return
        if not eleven_client or not gemini_model:
            log("Missing ElevenLabs or Gemini client; cannot process.")
            return

        try:
            # 1) mu-law -> PCM16 WAV
            pcm16 = ulaw_to_linear16(buf)
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(pcm16)
            wav_io.seek(0)

            # 2) STT (Georgian)
            log("STT (ka) starting...")
            stt_result = eleven_client.speech_to_text.convert(
                model_id=STT_MODEL,
                file=wav_io,
                language_code="ka"
            )
            user_text = (getattr(stt_result, "text", "") or "").strip()
            if not user_text:
                log("STT empty, skip.")
                return

            log(f"User 🇬🇪: {user_text}")

            # 3) Gemini reply (Georgian)
            prompt_parts = [
                "შენ ხარ ქართული საკონტაქტო ცენტრის ვირტუალური ოპერატორი.",
                "საუბრობ მხოლოდ ქართულად, ბუნებრივი, პროფესიული და მეგობრული ტონით.",
                "პასუხობ მოკლედ და გასაგებად. ნაბიჯ-ნაბიჯ უსვამ შეკითხვებს.",
                "ქვემოთ არის დიალოგის ისტორია:",
            ]

            for turn in conversation_history[-10:]:
                role = "მომხმარებელი" if turn["role"] == "user" else "ოპერატორი"
                prompt_parts.append(f"{role}: {turn['content']}")

            prompt_parts.append(f"მომხმარებელი: {user_text}")
            prompt_parts.append("ოპერატორი (მოკლე, გასაგები პასუხი სრულად ქართულად):")

            full_prompt = "\n".join(prompt_parts)

            log("Gemini thinking...")
            gemini_resp = gemini_model.generate_content(full_prompt)
            ai_text = (getattr(gemini_resp, "text", "") or "").strip()
            if not ai_text:
                ai_text = "უკაცრავად, ტექნიკური ხარვეზი მოხდა. შეგიძლიათ გაიმეოროთ კითხვა?"

            log(f"AI 🤖: {ai_text}")

            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": ai_text})

            # 4) TTS: ulaw_8000 for direct Twilio playback
            log("TTS (eleven_v3, Jessica, ulaw_8000)...")
            audio_iter = eleven_client.text_to_speech.convert(
                voice_id=VOICE_ID,
                model_id=TTS_MODEL,
                text=ai_text,
                output_format="ulaw_8000",
            )
            tts_bytes = b"".join(chunk for chunk in audio_iter)
            if not tts_bytes:
                log("Empty TTS, skip.")
                return

            # 5) Stream TTS back to Twilio
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
                    log(f"Error sending TTS chunk: {e}")
                    break

            log("TTS streamed to caller.")

        except Exception as e:
            log(f"Pipeline error: {e}")

    def buffer_worker():
        nonlocal audio_buffer, last_audio_time, processing, closed

        while not closed:
            try:
                if not processing and audio_buffer:
                    now = time.time()
                    gap_ms = (now - last_audio_time) * 1000.0
                    dur_ms = mulaw_buffer_duration_ms(audio_buffer)

                    if dur_ms >= MIN_UTTERANCE_MS and gap_ms >= SILENCE_GAP_MS:
                        processing = True
                        segment = bytes(audio_buffer)
                        audio_buffer = bytearray()
                        run_pipeline_on_buffer(segment)
                        processing = False

                time.sleep(CHECK_INTERVAL)
            except Exception as e:
                print(f"[Worker] Error: {e}")
                time.sleep(CHECK_INTERVAL)

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

            elif event == "media":
                payload_b64 = msg["media"]["payload"]
                try:
                    chunk = base64.b64decode(payload_b64)
                    audio_buffer.extend(chunk)
                    last_audio_time = time.time()
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
    app.run(host="0.0.0.0", port=5000, debug=True)
