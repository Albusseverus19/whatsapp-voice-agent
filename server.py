from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Connect
import config

app = Flask(__name__)

@app.route('/voice', methods=['POST'])
def voice():
    """Handle incoming voice calls from Twilio WhatsApp"""
    response = VoiceResponse()
    
    # Connect to ElevenLabs via WebSocket
    connect = Connect()
    connect.stream(url=f'wss://api.elevenlabs.io/v1/convai/conversation?agent_id={config.ELEVENLABS_AGENT_ID}')
    
    response.append(connect)
    
    return Response(str(response), mimetype='text/xml')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)