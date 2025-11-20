from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Chat</h1><p>Working minimal Flask app</p>'

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        api_key = os.getenv('GROQ_API_KEY', '')

        if not api_key:
            return jsonify({'error': 'GROQ_API_KEY not set'}), 500
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400

        r = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': 'Bearer ' + api_key,
                'Content-Type': 'application/json'
            },
            json={
                'model': 'llama-3.1-70b-versatile',
                'messages': messages,
                'temperature': 0.7
            },
            timeout=30
        )
        r.raise_for_status()
        j = r.json()
        return jsonify({'response': j['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
