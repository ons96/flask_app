from flask import Flask, request, render_template, jsonify
import requests
import os
import json

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def home():
    try:
        return render_template('chat.html')
    except Exception as e:
        return '<h1>Chat Interface</h1><p>Error: Template not found. Creating templates next...</p>'

@app.route('/roleplay')
def roleplay():
    try:
        return render_template('roleplay.html')
    except Exception as e:
        return '<h1>Roleplay Interface</h1><p>Error: Template not found</p>'

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
            return jsonify({'error': 'Invalid provider'}), 400

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
        if request.json and 'character' in request.json:
            character = request.json['character']
            return jsonify({'character': character, 'success': True})
        else:
            return jsonify({'error': 'No character data provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def call_groq(messages, model):
    api_key = os.getenv('GROQ_API_KEY', '')
    if not api_key:
        raise Exception('GROQ_API_KEY not set in environment variables')

    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
        json={'model': model, 'messages': messages, 'temperature': 0.7, 'max_tokens': 2048},
        timeout=30
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def call_openrouter(messages, model):
    api_key = os.getenv('OPENROUTER_API_KEY', '')
    if not api_key:
        raise Exception('OPENROUTER_API_KEY not set in environment variables')

    site_url = os.getenv('SITE_URL', 'https://blackberry-chat.vercel.app')
    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json', 'HTTP-Referer': site_url, 'X-Title': 'BlackBerry Chatbot'},
        json={'model': model, 'messages': messages, 'temperature': 0.9},
        timeout=60
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def call_custom_api(messages, model, endpoint, api_key):
    response = requests.post(
        endpoint,
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
        json={'model': model, 'messages': messages, 'temperature': 0.7},
        timeout=60
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def build_character_prompt(character):
    prompt = 'You are roleplaying as this character. Stay in character.

'
    char_data = character.get('data', character)

    if isinstance(char_data, dict):
        prompt += 'Name: ' + char_data.get('name', 'Unknown') + '
'
        prompt += 'Description: ' + char_data.get('description', '') + '
'
        prompt += 'Personality: ' + char_data.get('personality', '') + '
'
        if char_data.get('scenario'):
            prompt += 'Scenario: ' + char_data.get('scenario') + '
'

    return prompt

if __name__ == '__main__':
    app.run(debug=True)
