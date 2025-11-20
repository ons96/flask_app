from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os

app = Flask(__name__, static_folder='../public', static_url_path='')
CORS(app)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')

# Serve the main chat interface
@app.route('/')
def home():
    return send_from_directory('../public', 'index.html')

# Serve the roleplay interface
@app.route('/roleplay')
def roleplay():
    return send_from_directory('../public', 'roleplay.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

# API endpoint for chat
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        provider = data.get('provider', 'groq')
        model = data.get('model', 'llama-3.1-70b-versatile')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        if provider == 'groq':
            return chat_groq(message, model)
        elif provider == 'openrouter':
            return chat_openrouter(message, model)
        else:
            return jsonify({"error": "Invalid provider"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def chat_groq(message, model):
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        return jsonify({
            "response": result['choices'][0]['message']['content'],
            "model": model,
            "provider": "groq"
        })
    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500

def chat_openrouter(message, model):
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY not configured"}), 500
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get('VERCEL_URL', 'https://blackberry-chat.vercel.app'),
        "X-Title": "BlackBerry Chatbot"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        return jsonify({
            "response": result['choices'][0]['message']['content'],
            "model": model,
            "provider": "openrouter"
        })
    except Exception as e:
        return jsonify({"error": f"OpenRouter API error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()
