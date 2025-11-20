"""
Flask routes for multi-provider AI chatbot and roleplay
"""
import os
import requests
from flask import Blueprint, request, jsonify

chatbot_bp = Blueprint('chatbot', __name__, url_prefix='/api')

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')


def chat_with_puter(message: str, model: str = 'gpt-4o-mini') -> str:
    """Call Puter's free API"""
    try:
        response = requests.post(
            'https://api.puter.com/v1/chat/completions',
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': message}],
                'stream': False,
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
    except Exception as e:
        raise Exception(f'Puter error: {str(e)}')


def chat_with_groq(message: str, model: str = 'mixtral-8x7b-32768') -> str:
    """Call Groq's API"""
    if not GROQ_API_KEY:
        raise Exception('Groq API key not configured')
    
    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': message}],
                'temperature': 0.7,
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
    except Exception as e:
        raise Exception(f'Groq error: {str(e)}')


def chat_with_openrouter(message: str, model: str = 'gpt-3.5-turbo') -> str:
    """Call OpenRouter's API"""
    if not OPENROUTER_API_KEY:
        raise Exception('OpenRouter API key not configured')
    
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://puter-free-chatbot.vercel.app',
            },
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': message}],
                'temperature': 0.7,
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
    except Exception as e:
        raise Exception(f'OpenRouter error: {str(e)}')


@chatbot_bp.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Multi-provider chat endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        model = data.get('model', 'gpt-4o-mini')
        provider = data.get('provider', 'puter')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if provider == 'groq':
            response = chat_with_groq(message, model)
        elif provider == 'openrouter':
            response = chat_with_openrouter(message, model)
        else:  # puter
            response = chat_with_puter(message, model)
        
        return jsonify({
            'response': response,
            'model': model,
            'provider': provider,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@chatbot_bp.route('/roleplay', methods=['POST', 'OPTIONS'])
def roleplay():
    """Roleplay endpoint with character cards"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        character = data.get('character', {})
        model = data.get('model', 'gpt-4o-mini')
        provider = data.get('provider', 'puter')
        history = data.get('history', [])
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Build system prompt from character card
        system_parts = []
        if character.get('system_prompt'):
            system_parts.append(character['system_prompt'])
        else:
            if character.get('name'):
                system_parts.append(f"You are {character['name']}.")
            if character.get('description'):
                system_parts.append(f"Description: {character['description']}")
            if character.get('personality'):
                system_parts.append(f"Personality: {character['personality']}")
            if character.get('scenario'):
                system_parts.append(f"Scenario: {character['scenario']}")
            if character.get('example_dialogue'):
                system_parts.append(f"Example of how you speak: {character['example_dialogue']}")
        
        system_prompt = '\n\n'.join(system_parts) if system_parts else 'You are a helpful assistant.'
        
        # Prepare messages with system prompt and history
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(history)
        messages.append({'role': 'user', 'content': message})
        
        # Call appropriate provider
        if provider == 'groq':
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {GROQ_API_KEY}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model,
                    'messages': messages,
                    'temperature': 0.8,
                },
                timeout=30
            ).json()
        elif provider == 'openrouter':
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://puter-free-chatbot.vercel.app',
                },
                json={
                    'model': model,
                    'messages': messages,
                    'temperature': 0.8,
                },
                timeout=30
            ).json()
        else:  # puter
            response = requests.post(
                'https://api.puter.com/v1/chat/completions',
                json={
                    'model': model,
                    'messages': messages,
                    'stream': False,
                    'temperature': 0.8,
                },
                timeout=30
            ).json()
        
        ai_response = response.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
        
        return jsonify({
            'response': ai_response,
            'character_name': character.get('name', 'Unknown'),
            'model': model,
            'provider': provider,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@chatbot_bp.route('/models', methods=['GET', 'OPTIONS'])
def models():
    """Get available models from Puter"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        response = requests.get('https://puter.com/puterai/chat/models', timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
