from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Connect
import os

app = Flask(__name__)

# Get credentials from environment variables
ELEVENLABS_AGENT_ID = os.environ.get('ELEVENLABS_AGENT_ID')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

@app.route('/voice', methods=['POST'])
def voice():
    """Handle incoming voice calls from Twilio WhatsApp"""
    response = VoiceResponse()
    
    # Connect to ElevenLabs via WebSocket
    connect = Connect()
    connect.stream(url=f'wss://api.elevenlabs.io/v1/convai/conversation?agent_id={ELEVENLABS_AGENT_ID}')
    
    response.append(connect)
    
    return Response(str(response), mimetype='text/xml')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)