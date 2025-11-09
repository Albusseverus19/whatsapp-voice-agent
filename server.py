from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Connect
import os
import json
import websocket
import threading
import time

app = Flask(__name__)
sock = Sock(app)

# Get credentials from environment variables
ELEVENLABS_AGENT_ID = os.environ.get('ELEVENLABS_AGENT_ID')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

@app.route('/voice', methods=['POST'])
def voice():
    """Handle incoming voice calls from Twilio WhatsApp"""
    response = VoiceResponse()
    
    # Get the server URL
    server_url = request.host_url.replace('http://', 'wss://').replace('https://', 'wss://').rstrip('/')
    
    # Connect to our WebSocket endpoint for media streaming
    connect = Connect()
    connect.stream(url=f'{server_url}/media-stream')
    
    response.append(connect)
    
    return Response(str(response), mimetype='text/xml')

@sock.route('/media-stream')
def media_stream(twilio_ws):
    """Handle WebSocket connection from Twilio for media streaming"""
    print("[Twilio] WebSocket connection established")
    
    stream_sid = None
    elevenlabs_ws = None
    
    def on_elevenlabs_message(ws, message):
        """Handle messages from ElevenLabs"""
        try:
            data = json.loads(message)
            
            # Handle audio response from ElevenLabs
            if data.get('type') == 'audio':
                audio_data = data.get('audio', '')
                
                # Send audio back to Twilio
                media_message = {
                    'event': 'media',
                    'streamSid': stream_sid,
                    'media': {
                        'payload': audio_data
                    }
                }
                twilio_ws.send(json.dumps(media_message))
                print("[Audio] Sent to Twilio")
            
            # Handle interruption
            elif data.get('type') == 'interruption':
                clear_message = {
                    'event': 'clear',
                    'streamSid': stream_sid
                }
                twilio_ws.send(json.dumps(clear_message))
                print("[Interruption] Cleared Twilio buffer")
                
        except Exception as e:
            print(f"[ElevenLabs] Error: {e}")
    
    def on_elevenlabs_error(ws, error):
        print(f"[ElevenLabs] Error: {error}")
    
    def on_elevenlabs_close(ws, close_status_code, close_msg):
        print("[ElevenLabs] Connection closed")
    
    def on_elevenlabs_open(ws):
        print("[ElevenLabs] Connected successfully")
    
    try:
        while True:
            message = twilio_ws.receive()
            if message is None:
                break
                
            try:
                data = json.loads(message)
                event = data.get('event')
                
                if event == 'start':
                    stream_sid = data['start']['streamSid']
                    print(f"[Twilio] Stream started: {stream_sid}")
                    
                    # Connect to ElevenLabs
                    elevenlabs_url = f'wss://api.elevenlabs.io/v1/convai/conversation?agent_id={ELEVENLABS_AGENT_ID}'
                    
                    elevenlabs_ws = websocket.WebSocketApp(
                        elevenlabs_url,
                        header={'xi-api-key': ELEVENLABS_API_KEY},
                        on_open=on_elevenlabs_open,
                        on_message=on_elevenlabs_message,
                        on_error=on_elevenlabs_error,
                        on_close=on_elevenlabs_close
                    )
                    
                    # Run ElevenLabs WebSocket in separate thread
                    wst = threading.Thread(target=elevenlabs_ws.run_forever)
                    wst.daemon = True
                    wst.start()
                    
                    # Wait a bit for connection
                    time.sleep(1)
                    
                elif event == 'media' and elevenlabs_ws:
                    # Forward audio from Twilio to ElevenLabs
                    audio_payload = data['media']['payload']
                    
                    audio_message = {
                        'user_audio_chunk': audio_payload
                    }
                    
                    elevenlabs_ws.send(json.dumps(audio_message))
                    
                elif event == 'stop':
                    print("[Twilio] Stream stopped")
                    if elevenlabs_ws:
                        elevenlabs_ws.close()
                    break
                    
            except json.JSONDecodeError:
                print("[Twilio] Received invalid JSON")
            except Exception as e:
                print(f"[Twilio] Error: {e}")
                
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        print("[WebSocket] Connection closed")
        if elevenlabs_ws:
            elevenlabs_ws.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)