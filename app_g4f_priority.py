from g4f import ChatCompletion
from g4f.models import ModelUtils
import g4f
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import json
import time
from datetime import datetime

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

# Global cache for models
CACHED_MODELS = None
CACHED_TIMESTAMP = None
CACHE_DURATION = 300  # 5 minutes

# Provider priority configuration
PROVIDER_PRIORITY = {
    'cerebras': 1,
    'groq': 2,
    'google': 3,
    'googleai': 3,
    'googleaistudio': 3,
    'openai': 4,
    'anthropic': 4,
    'mistral': 5,
    'together': 6,
    'deepinfra': 7,
    'huggingface': 8,
    'g4f': 9  # g4f providers as fallback
}

# Model performance tiers (based on known characteristics)
MODEL_TIERS = {
    # Tier 1: Fastest, high-performance models
    'tier1': {
        'models': [
            'llama-3.3-70b', 'llama-4-scout', 'llama-4-maverick',
            'deepseek-r1', 'qwen-3-32b', 'gemini-2.5-flash'
        ],
        'priority': 1
    },
    # Tier 2: Gemini models (Google AI Studio)
    'tier2_gemini': {
        'models': [
            'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0',
            'gemini-1.5-flash', 'gemini-1.5-pro'
        ],
        'priority': 2
    },
    # Tier 3: Other high-quality models
    'tier3': {
        'models': [
            'gpt-4', 'gpt-4-turbo', 'claude-3-sonnet', 'claude-3-haiku',
            'mistral-large', 'mixtral-8x7b'
        ],
        'priority': 3
    }
}

def get_model_tier(model_name):
    """Determine which tier a model belongs to."""
    model_lower = model_name.lower()
    
    # Check tier 1 (priority models)
    for tier1_model in MODEL_TIERS['tier1']['models']:
        if tier1_model.lower() in model_lower or model_lower in tier1_model.lower():
            return 1
    
    # Check tier 2 (Gemini models)
    if 'gemini' in model_lower:
        return 2
    
    # Check tier 3 (other quality models)
    for tier3_model in MODEL_TIERS['tier3']['models']:
        if tier3_model.lower() in model_lower or model_lower in tier3_model.lower():
            return 3
    
    # Default tier
    return 4

def get_provider_priority(provider_name):
    """Get priority score for a provider (lower = higher priority)."""
    if not provider_name:
        return 999
    
    provider_lower = provider_name.lower()
    
    for priority_name, score in PROVIDER_PRIORITY.items():
        if priority_name in provider_lower:
            return score
    
    return 999

def get_available_models():
    """Get available models from g4f with priority ordering."""
    global CACHED_MODELS, CACHED_TIMESTAMP
    
    # Check cache
    if CACHED_MODELS and CACHED_TIMESTAMP:
        if time.time() - CACHED_TIMESTAMP < CACHE_DURATION:
            return CACHED_MODELS
    
    print("--- Fetching models from g4f ---")
    
    try:
        # Get all available models from g4f
        models = ModelUtils.convert
        available_models = []
        
        for model_name, model_obj in models.items():
            # Get providers for this model (simplified approach)
            providers = []
            
            # Check if specific providers support this model
            if CEREBRAS_PROVIDER and hasattr(CEREBRAS_PROVIDER, 'models'):
                if model_name in CEREBRAS_PROVIDER.models or any(m in model_name.lower() for m in [m.lower() for m in CEREBRAS_PROVIDER.models]):
                    providers.append('Cerebras')
            
            if GROQ_PROVIDER and hasattr(GROQ_PROVIDER, 'models'):
                if model_name in GROQ_PROVIDER.models or any(m in model_name.lower() for m in [m.lower() for m in GROQ_PROVIDER.models]):
                    providers.append('Groq')
            
            # For Gemini models, assume Google AI Studio is available
            if 'gemini' in model_name.lower():
                providers.append('Google AI Studio')
            
            # Add g4f as fallback provider
            providers.append('g4f')
            
            # Determine model tier and priority
            tier = get_model_tier(model_name)
            
            model_info = {
                'name': model_name,
                'display_name': format_model_name(model_name),
                'providers': providers,
                'tier': tier,
                'provider_priority': min([get_provider_priority(p) for p in providers])
            }
            
            available_models.append(model_info)
        
        # Sort models by tier, then by provider priority
        available_models.sort(key=lambda x: (x['tier'], x['provider_priority']))
        
        # Cache the results
        CACHED_MODELS = available_models
        CACHED_TIMESTAMP = time.time()
        
        print(f"--- Successfully loaded {len(available_models)} models ---")
        return available_models
        
    except Exception as e:
        print(f"--- Error fetching models from g4f: {e} ---")
        return []

def format_model_name(model_name):
    """Format model name for display."""
    # Convert hyphens to spaces and capitalize
    formatted = model_name.replace('-', ' ').replace('_', ' ')
    
    # Capitalize each word
    words = []
    for word in formatted.split():
        if word.lower() in ['ai', 'gpt', 'llm', 'api']:
            words.append(word.upper())
        elif word.isdigit() or '.' in word:
            words.append(word)
        else:
            words.append(word.capitalize())
    
    return ' '.join(words)

def get_best_provider_for_model(model_name, providers):
    """Get the best provider for a given model based on priority."""
    if not providers:
        return None
    
    # Sort providers by priority
    sorted_providers = sorted(providers, key=get_provider_priority)
    return sorted_providers[0]

def make_api_request(model_name, prompt, max_tokens=None, temperature=0.7):
    """Make API request using g4f with provider priority."""
    try:
        # Get available models to find providers
        models = get_available_models()
        model_info = next((m for m in models if m['name'] == model_name), None)
        
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        # Get best provider
        best_provider = get_best_provider_for_model(model_name, model_info['providers'])
        print(f"--- Using model {model_name} with provider {best_provider} ---")
        
        # Try with specific provider first if it's a priority provider
        if best_provider.lower() in ['cerebras', 'groq']:
            try:
                if best_provider.lower() == 'cerebras' and CEREBRAS_PROVIDER:
                    response = ChatCompletion.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        provider=CEREBRAS_PROVIDER,
                        temperature=temperature
                    )
                    return response
                elif best_provider.lower() == 'groq' and GROQ_PROVIDER:
                    response = ChatCompletion.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        provider=GROQ_PROVIDER,
                        temperature=temperature
                    )
                    return response
            except Exception as e:
                print(f"--- Priority provider {best_provider} failed: {e} ---")
        
        # Fallback to g4f automatic provider selection
        response = ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        return response
        
    except Exception as e:
        print(f"--- Error making API request: {e} ---")
        raise

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
            if model['tier'] == 1:
                organized_models['priority'].append({
                    'name': model['name'],
                    'display_name': model['display_name'],
                    'providers': model['providers']
                })
            elif model['tier'] == 2:
                organized_models['gemini'].append({
                    'name': model['name'],
                    'display_name': model['display_name'],
                    'providers': model['providers']
                })
            else:
                organized_models['other'].append({
                    'name': model['name'],
                    'display_name': model['display_name'],
                    'providers': model['providers']
                })
        
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
        
        # Make API request
        response = make_api_request(model_name, prompt)
        
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
    print("--- Starting Flask app with g4f priority system ---")
    print("--- Loading initial model cache ---")
    get_available_models()
    print("--- Flask app ready ---")
    app.run(debug=True, host='0.0.0.0', port=5000)