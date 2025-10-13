from flask import Flask, request, jsonify
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading
import time

app = Flask(__name__)

# Initialize TTS model
print("Loading TTS model...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
print("TTS model loaded successfully!")

# Global lock to prevent overlapping speech
speech_lock = threading.Lock()

def speak_text(text):
    """Synthesize and play text using TTS"""
    try:
        # Generate audio from text
        audio = tts.tts(text)
        
        # Play the audio
        sd.play(np.array(audio), samplerate=tts.synthesizer.output_sample_rate)
        sd.wait()  # Wait for playback to complete
        
        return True
    except Exception as e:
        print(f"Error in speech synthesis: {e}")
        return False

@app.route('/speak', methods=['POST'])
def speak():
    """API endpoint to convert text to speech"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract text from request
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Validate text length
        if len(text.strip()) == 0:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) > 1000:  # Limit text length
            return jsonify({'error': 'Text too long (max 1000 characters)'}), 400
        
        # Use lock to prevent overlapping speech
        with speech_lock:
            success = speak_text(text)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Text spoken successfully',
                'text': text
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to synthesize speech'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'TTS API',
        'port': 1098
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'service': 'Text-to-Speech API',
        'version': '1.0',
        'endpoints': {
            'POST /speak': 'Convert text to speech',
            'GET /health': 'Health check',
            'GET /': 'This documentation'
        },
        'usage': {
            'method': 'POST',
            'url': '/speak',
            'headers': {'Content-Type': 'application/json'},
            'body': {'text': 'Your text to speak'}
        },
        'example': {
            'curl': 'curl -X POST http://localhost:1098/speak -H "Content-Type: application/json" -d \'{"text": "Hello world"}\''
        }
    }), 200

if __name__ == '__main__':
    print("Starting TTS API server on port 1098...")
    print("API Documentation available at: http://localhost:1098")
    print("Health check available at: http://localhost:1098/health")
    print("Speak endpoint: POST http://localhost:1098/speak")
    
    app.run(host='0.0.0.0', port=1098, debug=True)
