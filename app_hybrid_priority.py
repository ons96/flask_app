from g4f import ChatCompletion
from g4f.models import ModelUtils
import g4f
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import json
import time
import csv
import requests
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import specific providers for priority handling
try:
    from g4f.Provider import Cerebras
    CEREBRAS_PROVIDER = Cerebras
except ImportError:
    CEREBRAS_PROVIDER = None

try:
    from g4f.Provider import Groq
    GROQ_PROVIDER = Groq
except ImportError:
    GROQ_PROVIDER = None

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Global cache
CACHED_MODELS = None
CACHED_TIMESTAMP = None
CACHE_DURATION = 300  # 5 minutes

# Performance data from CSV
PERFORMANCE_DATA = {}
CSV_PATH = os.path.join(os.path.dirname(__file__), "provider_performance.csv")

def load_performance_data():
    """Load performance data from CSV file."""
    global PERFORMANCE_DATA
    
    if not os.path.exists(CSV_PATH):
        print(f"--- Warning: Performance CSV not found at {CSV_PATH} ---")
        return
    
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                model_name = row.get('model_name_scraped', '').strip()
                provider_name = row.get('provider_name_scraped', '').strip()
                
                if model_name and provider_name:
                    # Parse performance metrics
                    try:
                        response_time = float(row.get('response_time_s', 999))
                        intelligence = float(row.get('intelligence_index', 0))
                    except (ValueError, TypeError):
                        response_time = 999
                        intelligence = 0
                    
                    # Store performance data
                    key = f"{model_name}|{provider_name}"
                    PERFORMANCE_DATA[key] = {
                        'model': model_name,
                        'provider': provider_name,
                        'response_time': response_time,
                        'intelligence': intelligence
                    }
        
        print(f"--- Loaded {len(PERFORMANCE_DATA)} performance entries from CSV ---")
        
    except Exception as e:
        print(f"--- Error loading performance data: {e} ---")

def get_performance_score(model_name, provider_name):
    """Get performance score for a model-provider combination."""
    key = f"{model_name}|{provider_name}"
    data = PERFORMANCE_DATA.get(key)
    
    if data:
        return data['response_time'], data['intelligence']
    
    # Default scores if not found in CSV
    return 999, 0

def get_cerebras_models():
    """Get models directly from Cerebras API."""
    if not CEREBRAS_API_KEY:
        return []
    
    try:
        response = requests.get(
            'https://api.cerebras.ai/v1/models',
            headers={'Authorization': f'Bearer {CEREBRAS_API_KEY}'},
            timeout=10
        )
        
        if response.status_code == 200:
            models = response.json()
            return [model['id'] for model in models.get('data', [])]
        else:
            print(f"--- Cerebras API error: {response.status_code} ---")
            return []
            
    except Exception as e:
        print(f"--- Error fetching Cerebras models: {e} ---")
        return []

def get_groq_models():
    """Get models directly from Groq API."""
    if not GROQ_API_KEY:
        return []
    
    try:
        response = requests.get(
            'https://api.groq.com/openai/v1/models',
            headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
            timeout=10
        )
        
        if response.status_code == 200:
            models = response.json()
            return [model['id'] for model in models.get('data', [])]
        else:
            print(f"--- Groq API error: {response.status_code} ---")
            return []
            
    except Exception as e:
        print(f"--- Error fetching Groq models: {e} ---")
        return []

def get_google_ai_models():
    """Get models from Google AI Studio."""
    if not GOOGLE_API_KEY:
        return []
    
    try:
        # List of known Google AI Studio models (including the missing Flash-Lite)
        google_models = [
            'gemini-2.5-flash',
            'gemini-2.5-flash-lite',  # This is missing from g4f!
            'gemini-2.0-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro'
        ]
        
        # For now, assume all are available since Google AI API verification is complex
        # We'll handle errors during actual API calls
        print(f"--- Assuming {len(google_models)} Google AI models are available ---")
        return google_models
        
    except Exception as e:
        print(f"--- Error fetching Google AI models: {e} ---")
        return []

def normalize_model_name(model_name):
    """Normalize model names for comparison."""
    # Remove common prefixes and normalize
    normalized = model_name.lower().strip()
    normalized = normalized.replace('meta-llama/', '').replace('cerebras/', '')
    normalized = normalized.replace('-instruct', '').replace('-chat', '')
    return normalized

def find_best_provider_for_model(model_name):
    """Find the best provider for a model based on performance data."""
    normalized_model = normalize_model_name(model_name)
    
    # Check performance data for this model across all providers
    candidates = []
    
    for key, data in PERFORMANCE_DATA.items():
        stored_model = normalize_model_name(data['model'])
        
        # Check if this performance entry matches our model
        if (normalized_model in stored_model or stored_model in normalized_model or
            any(part in stored_model for part in normalized_model.split('-') if len(part) > 2)):
            
            candidates.append({
                'provider': data['provider'],
                'response_time': data['response_time'],
                'intelligence': data['intelligence'],
                'original_model': data['model']
            })
    
    if candidates:
        # Sort by response time (fastest first)
        candidates.sort(key=lambda x: x['response_time'])
        return candidates
    
    return []

def get_available_models():
    """Get comprehensive model list from all sources."""
    global CACHED_MODELS, CACHED_TIMESTAMP
    
    # Check cache
    if CACHED_MODELS and CACHED_TIMESTAMP:
        if time.time() - CACHED_TIMESTAMP < CACHE_DURATION:
            return CACHED_MODELS
    
    print("--- Fetching models from all sources ---")
    
    # Get models from direct APIs
    cerebras_models = get_cerebras_models()
    groq_models = get_groq_models()
    google_models = get_google_ai_models()
    
    print(f"--- Direct API results: Cerebras={len(cerebras_models)}, Groq={len(groq_models)}, Google={len(google_models)} ---")
    
    # Get models from g4f
    g4f_models = []
    try:
        models = ModelUtils.convert
        g4f_models = list(models.keys())
        print(f"--- g4f models: {len(g4f_models)} ---")
    except Exception as e:
        print(f"--- Error getting g4f models: {e} ---")
    
    # Combine and organize all models
    all_models = {}
    
    # Process direct API models first (highest priority)
    for model in cerebras_models:
        if model not in all_models:
            all_models[model] = {
                'name': model,
                'display_name': format_model_name(model),
                'providers': [],
                'tier': get_model_tier(model)
            }
        if 'cerebras' not in all_models[model]['providers']:
            all_models[model]['providers'].append('cerebras')
    
    for model in groq_models:
        if model not in all_models:
            all_models[model] = {
                'name': model,
                'display_name': format_model_name(model),
                'providers': [],
                'tier': get_model_tier(model)
            }
        if 'groq' not in all_models[model]['providers']:
            all_models[model]['providers'].append('groq')
    
    for model in google_models:
        if model not in all_models:
            all_models[model] = {
                'name': model,
                'display_name': format_model_name(model),
                'providers': [],
                'tier': get_model_tier(model)
            }
        if 'google_ai' not in all_models[model]['providers']:
            all_models[model]['providers'].append('google_ai')
    
    # Add g4f models as fallback
    for model in g4f_models:
        if model not in all_models:
            all_models[model] = {
                'name': model,
                'display_name': format_model_name(model),
                'providers': [],
                'tier': get_model_tier(model)
            }
        if 'g4f' not in all_models[model]['providers']:
            all_models[model]['providers'].append('g4f')
    
    # Sort providers by performance for each model
    for model_data in all_models.values():
        model_name = model_data['name']
        provider_performance = find_best_provider_for_model(model_name)
        
        if provider_performance:
            # Reorder providers based on performance data
            provider_order = []
            for perf in provider_performance:
                provider_name = perf['provider'].lower()
                if 'cerebras' in provider_name and 'cerebras' in model_data['providers'] and 'cerebras' not in provider_order:
                    provider_order.append('cerebras')
                elif 'groq' in provider_name and 'groq' in model_data['providers'] and 'groq' not in provider_order:
                    provider_order.append('groq')
                elif 'google' in provider_name and 'google_ai' in model_data['providers'] and 'google_ai' not in provider_order:
                    provider_order.append('google_ai')
            
            # Add any remaining providers
            for provider in model_data['providers']:
                if provider not in provider_order:
                    provider_order.append(provider)
            
            model_data['providers'] = provider_order
            model_data['best_response_time'] = provider_performance[0]['response_time']
        else:
            model_data['best_response_time'] = 999
    
    # Convert to list and sort
    model_list = list(all_models.values())
    
    # Sort by tier, then by best response time
    model_list.sort(key=lambda x: (x['tier'], x['best_response_time']))
    
    # Cache results
    CACHED_MODELS = model_list
    CACHED_TIMESTAMP = time.time()
    
    print(f"--- Successfully loaded {len(model_list)} models total ---")
    return model_list

def get_model_tier(model_name):
    """Determine model tier based on name and performance."""
    model_lower = model_name.lower()
    
    # Tier 1: High-performance models (including Gemini 2.5 Flash variants)
    tier1_keywords = ['llama-4', 'llama-3.3', 'deepseek-r1', 'qwen-3', 'gemini-2.5-flash']
    if any(keyword in model_lower for keyword in tier1_keywords):
        return 1
    
    # Tier 2: Other Gemini models (excluding 2.5 Flash which is tier 1)
    if 'gemini' in model_lower and not any(t1 in model_lower for t1 in ['gemini-2.5-flash']):
        return 2
    
    # Tier 3: Other quality models
    tier3_keywords = ['gpt-4', 'claude-3', 'mistral', 'llama-3.1']
    if any(keyword in model_lower for keyword in tier3_keywords):
        return 3
    
    return 4

def format_model_name(model_name):
    """Format model name for display."""
    # Handle special cases
    if model_name == 'gemini-2.5-flash-lite':
        return 'Gemini 2.5 Flash-Lite'
    
    # General formatting
    formatted = model_name.replace('-', ' ').replace('_', ' ')
    formatted = formatted.replace('meta llama/', '').replace('cerebras/', '')
    
    words = []
    for word in formatted.split():
        if word.lower() in ['ai', 'gpt', 'llm', 'api']:
            words.append(word.upper())
        elif word.replace('.', '').isdigit():
            words.append(word)
        else:
            words.append(word.capitalize())
    
    return ' '.join(words)

def make_api_request(model_name, messages, temperature=0.7, max_tokens=None):
    """Make API request using the best available provider."""
    models = get_available_models()
    model_info = next((m for m in models if m['name'] == model_name), None)
    
    if not model_info:
        raise ValueError(f"Model {model_name} not found")
    
    providers = model_info['providers']
    print(f"--- Trying model {model_name} with providers: {providers} ---")
    
    # Try providers in order of preference (based on performance)
    for provider in providers:
        try:
            if provider == 'cerebras' and CEREBRAS_API_KEY:
                return make_cerebras_request(model_name, messages, temperature, max_tokens)
            elif provider == 'groq' and GROQ_API_KEY:
                return make_groq_request(model_name, messages, temperature, max_tokens)
            elif provider == 'google_ai' and GOOGLE_API_KEY:
                return make_google_ai_request(model_name, messages, temperature, max_tokens)
            elif provider == 'g4f':
                return make_g4f_request(model_name, messages, temperature, max_tokens)
        except Exception as e:
            print(f"--- Provider {provider} failed for {model_name}: {e} ---")
            continue
    
    raise Exception(f"All providers failed for model {model_name}")

def make_cerebras_request(model_name, messages, temperature, max_tokens):
    """Make request to Cerebras API."""
    response = requests.post(
        'https://api.cerebras.ai/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {CEREBRAS_API_KEY}',
            'Content-Type': 'application/json'
        },
        json={
            'model': model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens or 4000
        },
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        raise Exception(f"Cerebras API error: {response.status_code} - {response.text}")

def make_groq_request(model_name, messages, temperature, max_tokens):
    """Make request to Groq API."""
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        },
        json={
            'model': model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens or 4000
        },
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        raise Exception(f"Groq API error: {response.status_code} - {response.text}")

def make_google_ai_request(model_name, messages, temperature, max_tokens):
    """Make request to Google AI Studio."""
    try:
        model = genai.GenerativeModel(model_name)
        
        # Convert messages to Google AI format
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens or 4000
            )
        )
        
        return response.text
        
    except Exception as e:
        raise Exception(f"Google AI error: {e}")

def make_g4f_request(model_name, messages, temperature, max_tokens):
    """Make request using g4f."""
    response = ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )
    return response

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_models')
def get_models():
    """Get available models organized by priority."""
    try:
        models = get_available_models()
        
        # Organize models by tier
        organized_models = {
            'priority': [],
            'gemini': [],
            'other': []
        }
        
        for model in models:
            model_data = {
                'name': model['name'],
                'display_name': model['display_name'],
                'providers': model['providers'],
                'best_response_time': model.get('best_response_time', 999)
            }
            
            if model['tier'] == 1:
                organized_models['priority'].append(model_data)
            elif model['tier'] == 2:
                organized_models['gemini'].append(model_data)
            else:
                organized_models['other'].append(model_data)
        
        return jsonify({
            'success': True,
            'models': organized_models,
            'total_count': len(models)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        data = request.get_json()
        model_name = data.get('model')
        prompt = data.get('prompt')
        
        if not model_name or not prompt:
            return jsonify({
                'success': False,
                'error': 'Model and prompt are required'
            })
        
        messages = [{"role": "user", "content": prompt}]
        response = make_api_request(model_name, messages)
        
        return jsonify({
            'success': True,
            'response': response,
            'model': model_name,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("--- Starting Hybrid Priority System ---")
    print("--- Loading performance data from CSV ---")
    load_performance_data()
    print("--- Loading initial model cache ---")
    get_available_models()
    print("--- Flask app ready ---")
    app.run(debug=True, host='0.0.0.0', port=5000)