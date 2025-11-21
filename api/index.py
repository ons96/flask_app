from flask import Flask, request, render_template, jsonify
import requests
import os
import json
import base64
from PIL import Image
import io

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def home():
    try:
        return render_template('chat.html')
    except Exception as e:
        return f'<h1>Chat Interface</h1><p>Error: {str(e)}</p>'

@app.route('/roleplay')
def roleplay():
    try:
        return render_template('roleplay.html')
    except Exception as e:
        return f'<h1>Roleplay Interface</h1><p>Error: {str(e)}</p>'

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'All systems operational'})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        provider = data.get('provider', 'groq')
        model = data.get('model', 'llama-3.1-70b-versatile')
        custom_endpoint = data.get('custom_endpoint')
        custom_key = data.get('custom_key')
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
            
        if provider == 'groq':
            response_text = call_groq(messages, model)
        elif provider == 'openrouter':
            response_text = call_openrouter(messages, model)
        elif provider == 'custom' and custom_endpoint and custom_key:
            response_text = call_custom_api(messages, model, custom_endpoint, custom_key)
        else:
            return jsonify({'error': 'Invalid provider or missing custom credentials'}), 400
            
        return jsonify({'response': response_text, 'model': model, 'provider': provider})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/roleplay/chat', methods=['POST'])
def roleplay_chat():
    try:
        data = request.json
        message = data.get('message', '')
        character = data.get('character', {})
        history = data.get('history', [])
        provider = data.get('provider', 'openrouter')
        model = data.get('model', 'meta-llama/llama-3.1-70b-instruct:free')
        
        if not message or not character:
            return jsonify({'error': 'Missing message or character'}), 400
            
        system_prompt = build_character_prompt(character)
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(history)
        messages.append({'role': 'user', 'content': message})
        
        if provider == 'groq':
            response_text = call_groq(messages, model)
        elif provider == 'openrouter':
            response_text = call_openrouter(messages, model)
        else:
            return jsonify({'error': 'Invalid provider'}), 400
            
        return jsonify({'response': response_text, 'model': model, 'provider': provider})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/character/parse', methods=['POST'])
def parse_character():
    try:
        # Handle URL-based loading
        if request.is_json:
            data = request.json
            if 'url' in data:
                return parse_character_from_url(data['url'])
            elif 'character' in data:
                return jsonify({'character': data['character'], 'success': True})
        
        # Handle File Upload (Multipart)
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename.lower()
            
            if filename.endswith('.json'):
                content = json.load(file)
                return jsonify({'character': content, 'success': True})
            elif filename.endswith('.png'):
                return parse_character_from_png(file)
                
        return jsonify({'error': 'No valid character data provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def parse_character_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Try parsing as JSON directly
        try:
            character = response.json()
            return jsonify({'character': character, 'success': True})
        except:
            pass
            
        # If it's a PNG (raw bytes), we might need to handle it, but usually URLs are JSONs or raw images
        # For now, assume JSON or fail
        return jsonify({'error': 'Could not parse JSON from URL'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 400

def parse_character_from_png(file_storage):
    try:
        img = Image.open(file_storage)
        img.load() # Load metadata
        
        # Check for 'chara' in text chunks (common in V2/V3 cards)
        if 'chara' in img.info:
            # It's base64 encoded
            decoded = base64.b64decode(img.info['chara']).decode('utf-8')
            character = json.loads(decoded)
            return jsonify({'character': character, 'success': True})
            
        # Check for 'ccv3' (V3 spec)
        if 'ccv3' in img.info:
             # Not fully implemented, but placeholder
             pass
             
        return jsonify({'error': 'No character metadata found in PNG'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to parse PNG: {str(e)}'}), 400

def call_groq(messages, model):
    api_key = os.getenv('GROQ_API_KEY', '')
    if not api_key:
        raise Exception('GROQ_API_KEY not set')
        
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        json={
            'model': model,
            'messages': messages,
            'temperature': 0.7,
            'max_tokens': 2048
        },
        timeout=30
    )
    
    if response.status_code != 200:
        raise Exception(f'Groq API Error: {response.text}')
        
    return response.json()['choices'][0]['message']['content']

def call_openrouter(messages, model):
    api_key = os.getenv('OPENROUTER_API_KEY', '')
    if not api_key:
        raise Exception('OPENROUTER_API_KEY not set')
        
    site_url = os.getenv('SITE_URL', 'https://blackberry-chat.vercel.app')
    
    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': site_url,
            'X-Title': 'BlackBerry Chatbot'
        },
        json={
            'model': model,
            'messages': messages,
            'temperature': 0.9
        },
        timeout=60
    )
    
    if response.status_code != 200:
        raise Exception(f'OpenRouter API Error: {response.text}')
        
    return response.json()['choices'][0]['message']['content']

def call_custom_api(messages, model, endpoint, api_key):
    response = requests.post(
        endpoint,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        json={
            'model': model,
            'messages': messages,
            'temperature': 0.7
        },
        timeout=60
    )
    
    if response.status_code != 200:
        raise Exception(f'Custom API Error: {response.text}')
        
    return response.json()['choices'][0]['message']['content']

def build_character_prompt(character):
    base = 'You are roleplaying as this character. Stay in character. '
    
    # Handle different character card formats (V1, V2, Tavern)
    char_data = character.get('data', character)
    
    name = char_data.get('name', 'Unknown')
    description = char_data.get('description', '')
    personality = char_data.get('personality', '')
    scenario = char_data.get('scenario', '')
    first_mes = char_data.get('first_mes', '')
    
    base += f'Name: {name}\n'
    if description: base += f'Description: {description}\n'
    if personality: base += f'Personality: {personality}\n'
    if scenario: base += f'Scenario: {scenario}\n'
    
    # Note: first_mes is usually sent as the first assistant message, but we can include it in system prompt context if needed
    
    return base

if __name__ == '__main__':
    app.run(debug=True)
