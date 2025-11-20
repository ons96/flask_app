from flask import Flask, jsonify
import requests, os

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Chat</h1><p>Working!</p>'

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        msg = data.get('messages', [])
        key = os.getenv('GROQ_API_KEY', '')
        
        r = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
            json={'model': 'llama-3.1-70b-versatile', 'messages': msg},
            timeout=30
        )
        
        return jsonify({'response': r.json()['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
