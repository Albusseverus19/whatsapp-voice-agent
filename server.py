from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Connect
import os
import json
import base64
import asyncio
import websockets
import threading

app = Flask(__name__)
sock = Sock(app)

# Get credentials from environment variables
ELEVENLABS_AGENT_ID = os.environ.get('ELEVENLABS_AGENT_ID')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

@app.route('/voice', methods=['POST'])
def voice():
    """Handle incoming voice calls from Twilio WhatsApp"""
    response = VoiceResponse()
    
    # Get the server URL from environment or use the request host
    server_url = os.environ.get('SERVER_URL', request.host_url.replace('http://', 'wss://').replace('https://', 'wss://').rstrip('/'))
    
    # Connect to our WebSocket endpoint for media streaming
    connect = Connect()
    connect.stream(url=f'{server_url}/media-stream')
    
    response.append(connect)
    
    return Response(str(response), mimetype='text/xml')

@sock.route('/media-stream')
def media_stream(ws):
    """Handle WebSocket connection from Twilio for media streaming"""
    print("[Twilio] WebSocket connection established")
    
    stream_sid = None
    elevenlabs_ws = None
    
    async def connect_to_elevenlabs():
        """Connect to ElevenLabs WebSocket"""
        nonlocal elevenlabs_ws
        
        # Get signed URL for ElevenLabs
        elevenlabs_url = f'wss://api.elevenlabs.io/v1/convai/conversation?agent_id={ELEVENLABS_AGENT_ID}'
        
        print(f"[ElevenLabs] Connecting to: {elevenlabs_url}")
        
        try:
            elevenlabs_ws = await websockets.connect(
                elevenlabs_url,
                extra_headers={
                    'xi-api-key': ELEVENLABS_API_KEY
                }
            )
            print("[ElevenLabs] Connected successfully")
            
            # Start listening to ElevenLabs responses
            async for message in elevenlabs_ws:
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
                        ws.send(json.dumps(media_message))
                        print("[Audio] Sent to Twilio")
                    
                    # Handle interruption
                    elif data.get('type') == 'interruption':
                        clear_message = {
                            'event': 'clear',
                            'streamSid': stream_sid
                        }
                        ws.send(json.dumps(clear_message))
                        print("[Interruption] Cleared Twilio buffer")
                        
                except json.JSONDecodeError:
                    print("[ElevenLabs] Received non-JSON message")
                except Exception as e:
                    print(f"[ElevenLabs] Error processing message: {e}")
                    
        except Exception as e:
            print(f"[ElevenLabs] Connection error: {e}")
    
    def run_elevenlabs_connection():
        """Run ElevenLabs connection in async context"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(connect_to_elevenlabs())
    
    # Start ElevenLabs connection in separate thread
    elevenlabs_thread = None
    
    try:
        while True:
            message = ws.receive()
            if message is None:
                break
                
            try:
                data = json.loads(message)
                event = data.get('event')
                
                if event == 'start':
                    stream_sid = data['start']['streamSid']
                    print(f"[Twilio] Stream started: {stream_sid}")
                    
                    # Start ElevenLabs connection
                    elevenlabs_thread = threading.Thread(target=run_elevenlabs_connection)
                    elevenlabs_thread.daemon = True
                    elevenlabs_thread.start()
                    
                elif event == 'media':
                    # Forward audio from Twilio to ElevenLabs
                    if elevenlabs_ws and elevenlabs_ws.open:
                        audio_payload = data['media']['payload']
                        
                        # Send to ElevenLabs
                        audio_message = {
                            'user_audio_chunk': audio_payload
                        }
                        
                        asyncio.run_coroutine_threadsafe(
                            elevenlabs_ws.send(json.dumps(audio_message)),
                            elevenlabs_thread._target.__self__
                        )
                        
                elif event == 'stop':
                    print("[Twilio] Stream stopped")
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
            asyncio.run(elevenlabs_ws.close())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)