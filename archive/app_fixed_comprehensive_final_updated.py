import g4f
from flask_session import Session
from g4f import ChatCompletion
from g4f.models import ModelUtils, IterListProvider
import os
import json
import uuid
from datetime import datetime
import asyncio
import platform
import html
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import csv
from duckduckgo_search import DDGS
import aiohttp
import re
import sys
import shutil
import time
from functools import wraps
from flask import Flask, request, session, redirect, url_for, abort, render_template_string

# Load environment variables
load_dotenv()

# Initialize global variables
CACHED_AVAILABLE_MODELS_SORTED_LIST = []
CACHED_MODEL_PROVIDER_INFO = {}
CACHED_PROVIDER_CLASS_MAP = {}
PROVIDER_PERFORMANCE_CACHE = []
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CONTEXT_WINDOW_CACHE = {}

# Set Windows event loop policy if on Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Print API key status
print(f"--- CEREBRAS_API_KEY: {'Available' if CEREBRAS_API_KEY else 'Not Set'} ---")
print(f"--- GROQ_API_KEY: {'Available' if GROQ_API_KEY else 'Not Set'} ---")
print(f"--- GOOGLE_API_KEY: {'Available' if GOOGLE_API_KEY else 'Not Set'} ---")
print(f"--- OPENROUTER_API_KEY: {'Available' if OPENROUTER_API_KEY else 'Not Set'} ---")

# Configuration
CHAT_STORAGE = os.path.join(APP_DIR, "chats.json")
PERFORMANCE_CSV_PATH = os.path.join(APP_DIR, "provider_performance.csv")
DEFAULT_MAX_OUTPUT_TOKENS = 4000

# Free models by provider (updated and comprehensive)
FREE_MODELS_BY_PROVIDER = {
    "OpenRouter": [
        "DeepSeek R1", "DeepSeek R1 Distill Llama 70B", "DeepSeek R1 Distill Qwen 14B",
        "DeepSeek R1 Distill Qwen 32B", "DeepSeek R1 Zero", "DeepSeek V3",
        "Gemma 2 9B Instruct", "Gemma 3 12B Instruct", "Gemma 3 1B Instruct", 
        "Gemma 3 27B Instruct", "Gemma 3 4B Instruct", "Llama 3.1 8B Instruct",
        "Llama 3.2 11B Vision Instruct", "Llama 3.2 1B Instruct", "Llama 3.2 3B Instruct",
        "Llama 3.3 70B Instruct", "Llama 4 Maverick", "Llama 4 Scout",
        "Mistral 7B Instruct", "Mistral Nemo", "Qwen 2.5 72B Instruct", "Qwen 2.5 7B Instruct",
        "Qwen QwQ 32B"
    ],
    "Google": [
        "Gemini 2.5 Flash", "Gemini 2.0 Flash", "Gemini 2.0 Flash-Lite",
        "Gemini 1.5 Flash", "Gemini 1.5 Flash-8B"
    ],
    "Cerebras": ["Qwen 3 32B", "Llama 4 Scout", "Llama 3.1 8B", "Llama 3.3 70B"],
    "Groq": [
        "Allam 2 7B", "DeepSeek R1 Distill Llama 70B", "Gemma 2 9B Instruct",
        "Llama 3 70B", "Llama 3 8B", "Llama 3.1 8B", "Llama 3.3 70B",
        "Llama 4 Maverick", "Llama 4 Scout", "Mistral Saba 24B", "Qwen QwQ 32B"
    ],
    "Chutes": [
        "DeepSeek R1", "DeepSeek R1-Zero", "DeepSeek V3", "DeepSeek V3 Base",
        "Llama 3.1 Nemotron Ultra 253B v1", "Llama 4 Maverick", "Llama 4 Scout",
        "Mistral Small 3.1 24B Instruct", "Qwen 2.5 VL 32B Instruct"
    ],
    "Together": ["Llama 3.2 11B Vision Instruct", "Llama 3.3 70B Instruct", "DeepSeek R1 Distil Llama 70B"],
    "Cloudflare": [
        "DeepSeek R1 Distill Qwen 32B", "Gemma 3 12B Instruct", "Llama 3 8B Instruct",
        "Llama 3.1 8B Instruct", "Llama 3.2 11B Vision Instruct", "Llama 3.2 1B Instruct",
        "Llama 3.2 3B Instruct", "Llama 3.3 70B Instruct", "Llama 4 Scout Instruct",
        "Mistral 7B Instruct", "Mistral Small 3.1 24B Instruct", "Qwen QwQ 32B"
    ]
}

# Model display name mapping for deduplication
MODEL_DISPLAY_NAME_MAP = {
    # Llama 4 Maverick variants
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "meta-llama/llama-4-maverick-17b-16e-instruct": "Llama 4 Maverick",
    "llama-4-maverick": "Llama 4 Maverick",
    
    # Llama 4 Scout variants
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "llama-4-scout": "Llama 4 Scout",
    
    # Llama 3.3 70B variants
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.3-70b": "Llama 3.3 70B",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    
    # DeepSeek variants
    "deepseek-r1": "DeepSeek R1",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
    "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
    
    # QwQ variants
    "qwen-qwq-32b": "Qwen QwQ 32B",
    "qwen/qwq-32b": "Qwen QwQ 32B",
    
    # Gemini variants
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-1.5-flash": "Gemini 1.5 Flash",
}

# Provider-specific model ID mappings
PROVIDER_MODEL_MAPPINGS = {
    "groq": {
        "Llama 4 Maverick": "meta-llama/llama-4-maverick-17b-16e-instruct",
        "Llama 4 Scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        "Llama 3.3 70B": "llama-3.3-70b-versatile",
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "DeepSeek R1 Distill Llama 70B": "deepseek-r1-distill-llama-70b",
        "Qwen QwQ 32B": "qwen-qwq-32b"
    },
    "cerebras": {
        "Llama 4 Scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
        "Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct",
        "Qwen 3 32B": "qwen/qwen-3-32b-instruct"
    },
    "google": {
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 2.0 Flash": "gemini-2.0-flash",
        "Gemini 1.5 Flash": "gemini-1.5-flash"
    }
}

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = os.path.join(APP_DIR, 'flask_session')
Session(app)

# Rate limiting decorator
def rate_limit(max_calls=60, period=60):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple rate limiting logic
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Utility functions
def load_chats():
    """Load chat histories from JSON file."""
    try:
        if os.path.exists(CHAT_STORAGE):
            with open(CHAT_STORAGE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chats: {e}")
    return {}

def save_chats(chats):
    """Save chat histories to JSON file."""
    try:
        os.makedirs(os.path.dirname(CHAT_STORAGE), exist_ok=True)
        with open(CHAT_STORAGE, 'w', encoding='utf-8') as f:
            json.dump(chats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chats: {e}")

def get_normalized_model_name(model_name):
    """Get normalized model name for deduplication."""
    return MODEL_DISPLAY_NAME_MAP.get(model_name, model_name)

def get_provider_model_id(model_name, provider):
    """Get provider-specific model ID."""
    provider_key = provider.lower()
    if provider_key in PROVIDER_MODEL_MAPPINGS:
        return PROVIDER_MODEL_MAPPINGS[provider_key].get(model_name, model_name)
    return model_name

def is_free_model(model_name, provider_name):
    """Check if a model is free for a given provider."""
    provider_models = FREE_MODELS_BY_PROVIDER.get(provider_name, [])
    normalized_name = get_normalized_model_name(model_name)
    return normalized_name in provider_models or model_name in provider_models

def get_available_models_with_provider_counts():
    """Get available models with proper deduplication and provider counting."""
    print("--- [MODEL_DISCOVERY] Starting model discovery ---")
    
    model_dict = {}  # Use dict for deduplication
    model_provider_info = {}
    provider_class_map = {}
    
    try:
        # Get providers from g4f
        providers = getattr(g4f, 'Provider', None)
        if providers:
            provider_list = [getattr(providers, name) for name in dir(providers) 
                           if not name.startswith('_') and hasattr(getattr(providers, name), '__dict__')]
            
            for provider in provider_list:
                try:
                    provider_name = provider.__name__ if hasattr(provider, '__name__') else str(provider)
                    provider_class_map[provider_name] = provider
                    
                    # Get models for this provider
                    if hasattr(provider, 'models') and provider.models:
                        for model in provider.models:
                            model_name = str(model)
                            normalized_name = get_normalized_model_name(model_name)
                            
                            # Initialize model entry if not exists
                            if normalized_name not in model_dict:
                                model_dict[normalized_name] = {
                                    'provider_count': 0,
                                    'intelligence_index': 0,
                                    'response_time': float('inf'),
                                    'is_free': False,
                                    'providers': []
                                }
                            
                            # Update provider count and info
                            model_dict[normalized_name]['provider_count'] += 1
                            model_dict[normalized_name]['providers'].append(provider_name)
                            
                            # Check if free with any provider
                            if is_free_model(model_name, provider_name):
                                model_dict[normalized_name]['is_free'] = True
                            
                            # Store provider info
                            if normalized_name not in model_provider_info:
                                model_provider_info[normalized_name] = []
                            model_provider_info[normalized_name].append(provider)
                            
                except Exception as e:
                    print(f"--- [MODEL_DISCOVERY] Error processing provider {provider}: {e}")
    
    except Exception as e:
        print(f"--- [MODEL_DISCOVERY] Error accessing g4f providers: {e}")
    
    # Add default models if none found
    if not model_dict:
        default_models = [
            "Llama 4 Maverick", "Llama 4 Scout", "Llama 3.3 70B", "DeepSeek R1",
            "Qwen QwQ 32B", "Gemini 2.5 Flash", "Gemini 2.0 Flash"
        ]
        for model in default_models:
            model_dict[model] = {
                'provider_count': 1,
                'intelligence_index': 90,
                'response_time': 1.0,
                'is_free': True,
                'providers': ['Default']
            }
    
    # Convert to sorted list with prioritization
    sorted_models = []
    for model_name, info in model_dict.items():
        # Prioritization logic:
        # 1. Free models first
        # 2. Higher intelligence index
        # 3. Lower response time
        # 4. Higher provider count
        
        priority_score = 0
        if info['is_free']:
            priority_score += 1000
        
        priority_score += info['intelligence_index']
        
        if info['response_time'] != float('inf'):
            priority_score -= info['response_time'] * 10
        
        priority_score += info['provider_count']
        
        sorted_models.append((
            model_name,
            info['provider_count'],
            info['intelligence_index'],
            info['response_time'],
            priority_score
        ))
    
    # Sort by priority score (descending)
    sorted_models.sort(key=lambda x: x[4], reverse=True)
    
    # Remove priority score from final result
    final_models = [(name, count, intel, time) for name, count, intel, time, _ in sorted_models]
    
    print(f"--- [MODEL_DISCOVERY] Found {len(final_models)} unique models ---")
    return final_models, model_provider_info, provider_class_map

def perform_web_search(query, max_results=5):
    """Perform web search using multiple methods with fallback."""
    search_results = ""
    
    # Method 1: DuckDuckGo (free and unlimited)
    try:
        print("--- Attempting DuckDuckGo search ---")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                search_results = "Web Search Results (via DuckDuckGo):\n"
                for i, result in enumerate(results, 1):
                    search_results += f"{i}. {result.get('title', '')}\n"
                    search_results += f"   {result.get('body', '')}\n"
                    search_results += f"   Source: {result.get('href', '')}\n\n"
                return search_results
    except Exception as e:
        print(f"--- DuckDuckGo search failed: {e} ---")
    
    # Method 2: Google AI with grounding (if API key available)
    if GOOGLE_API_KEY:
        try:
            print("--- Attempting Google AI search ---")
            # Implement Google AI search here if needed
            pass
        except Exception as e:
            print(f"--- Google AI search failed: {e} ---")
    
    # Method 3: SerpAPI (limited free tier)
    if SERPAPI_API_KEY:
        try:
            print("--- Attempting SerpAPI search ---")
            url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_API_KEY}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get('organic_results', [])
                if organic_results:
                    search_results = "Web Search Results (via SerpAPI):\n"
                    for i, result in enumerate(organic_results[:max_results], 1):
                        search_results += f"{i}. {result.get('title', '')}\n"
                        search_results += f"   {result.get('snippet', '')}\n"
                        search_results += f"   Source: {result.get('link', '')}\n\n"
                    return search_results
        except Exception as e:
            print(f"--- SerpAPI search failed: {e} ---")
    
    return "No web search results available."

async def call_groq_api(messages, model_name, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Call Groq API directly."""
    if not GROQ_API_KEY:
        raise Exception("Groq API key not available")
    
    model_id = get_provider_model_id(model_name, "groq")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": messages,
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.groq.com/openai/v1/chat/completions", 
                               headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                raise Exception(f"Groq API error: {response.status} - {error_text}")

async def call_cerebras_api(messages, model_name, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Call Cerebras API directly."""
    if not CEREBRAS_API_KEY:
        raise Exception("Cerebras API key not available")
    
    model_id = get_provider_model_id(model_name, "cerebras")
    
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": messages,
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.cerebras.ai/v1/chat/completions", 
                               headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                raise Exception(f"Cerebras API error: {response.status} - {error_text}")

async def call_google_api(messages, model_name, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Call Google Gemini API directly."""
    if not GOOGLE_API_KEY:
        raise Exception("Google API key not available")
    
    model_id = get_provider_model_id(model_name, "google")
    
    # Convert messages to Google format
    contents = []
    for msg in messages:
        contents.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["content"]}]
        })
    
    data = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7
        }
    }
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GOOGLE_API_KEY}"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                error_text = await response.text()
                raise Exception(f"Google API error: {response.status} - {error_text}")

async def call_g4f_api(messages, model_name):
    """Call g4f as fallback."""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: ChatCompletion.create(
                model=model_name,
                messages=messages,
                stream=False
            )
        )
        return response
    except Exception as e:
        raise Exception(f"G4F API error: {str(e)}")

async def get_llm_response(messages, model_name):
    """Get LLM response with provider prioritization."""
    errors = []
    
    # Try direct API providers first (prioritized)
    normalized_model = get_normalized_model_name(model_name)
    
    # 1. Try Groq if model is supported
    if normalized_model in ["Llama 4 Maverick", "Llama 4 Scout", "Llama 3.3 70B", "DeepSeek R1 Distill Llama 70B", "Qwen QwQ 32B"] and GROQ_API_KEY:
        try:
            print(f"--- Attempting Groq API for {model_name} ---")
            response = await call_groq_api(messages, normalized_model)
            return response, "Groq"
        except Exception as e:
            errors.append(f"Groq: {str(e)}")
            print(f"--- Groq API failed: {e} ---")
    
    # 2. Try Cerebras if model is supported
    if normalized_model in ["Llama 4 Scout", "Llama 3.3 70B", "Qwen 3 32B", "Llama 3.1 8B"] and CEREBRAS_API_KEY:
        try:
            print(f"--- Attempting Cerebras API for {model_name} ---")
            response = await call_cerebras_api(messages, normalized_model)
            return response, "Cerebras"
        except Exception as e:
            errors.append(f"Cerebras: {str(e)}")
            print(f"--- Cerebras API failed: {e} ---")
    
    # 3. Try Google if model is supported
    if normalized_model.startswith("Gemini") and GOOGLE_API_KEY:
        try:
            print(f"--- Attempting Google API for {model_name} ---")
            response = await call_google_api(messages, normalized_model)
            return response, "Google AI"
        except Exception as e:
            errors.append(f"Google: {str(e)}")
            print(f"--- Google API failed: {e} ---")
    
    # 4. Fallback to G4F
    try:
        print(f"--- Attempting G4F for {model_name} ---")
        response = await call_g4f_api(messages, model_name)
        return response, "G4F"
    except Exception as e:
        errors.append(f"G4F: {str(e)}")
        print(f"--- G4F failed: {e} ---")
    
    # If all methods fail
    error_msg = f"All providers failed for {model_name}: " + "; ".join(errors)
    raise Exception(error_msg)

# Initialize model cache
def initialize_model_cache():
    """Initialize the global model cache."""
    global CACHED_AVAILABLE_MODELS_SORTED_LIST, CACHED_MODEL_PROVIDER_INFO, CACHED_PROVIDER_CLASS_MAP
    
    print("--- [CACHE] Initializing model cache ---")
    try:
        list_data, info_data, map_data = get_available_models_with_provider_counts()
        
        CACHED_AVAILABLE_MODELS_SORTED_LIST = list_data
        CACHED_MODEL_PROVIDER_INFO = info_data
        CACHED_PROVIDER_CLASS_MAP = map_data
        
        print(f"--- [CACHE] Cached {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)} models ---")
    except Exception as e:
        print(f"--- [CACHE] Error initializing cache: {e} ---")
        # Fallback to default models
        CACHED_AVAILABLE_MODELS_SORTED_LIST = [
            ("Llama 4 Maverick", 1, 95, 0.8),
            ("Llama 4 Scout", 1, 93, 0.9),
            ("DeepSeek R1", 1, 98, 1.2),
            ("Qwen QwQ 32B", 1, 94, 1.0),
            ("Gemini 2.5 Flash", 1, 96, 0.7)
        ]

# Routes
@app.route('/', methods=['GET', 'POST'])
@rate_limit(max_calls=60, period=60)
def index():
    chats = load_chats()
    available_models_sorted_list = CACHED_AVAILABLE_MODELS_SORTED_LIST
    
    # Get unique model names
    available_model_names = {name for name, count, intel, rt in available_models_sorted_list}
    
    # Set default model
    default_model = available_models_sorted_list[0][0] if available_models_sorted_list else "Llama 4 Maverick"
    
    # Initialize or load current chat
    if 'current_chat' not in session or session['current_chat'] not in chats:
        session['current_chat'] = str(uuid.uuid4())
        chats[session['current_chat']] = {
            "history": [], 
            "model": default_model, 
            "name": "New Chat", 
            "created_at": datetime.now().isoformat()
        }
        save_chats(chats)
    
    current_chat = chats[session['current_chat']]
    current_model = current_chat.get("model", default_model)
    
    # Ensure current model is valid
    if current_model not in available_model_names and available_model_names:
        current_model = default_model
        current_chat["model"] = current_model

    if request.method == 'POST':
        # Handle New Chat action
        if 'new_chat' in request.form:
            session['current_chat'] = str(uuid.uuid4())
            chats[session['current_chat']] = {
                "history": [], 
                "model": default_model, 
                "name": "New Chat", 
                "created_at": datetime.now().isoformat()
            }
            save_chats(chats)
            return redirect(url_for('index'))
        
        # Handle Delete Chat action
        if 'delete_chat' in request.form:
            chat_to_delete = session.get('current_chat')
            if chat_to_delete and chat_to_delete in chats:
                del chats[chat_to_delete]
                save_chats(chats)
                session.pop('current_chat', None)
                return redirect(url_for('index'))
        
        # Process message submission or regeneration
        prompt_from_input = request.form.get('prompt', '').strip()
        selected_model_for_request = request.form.get('model', default_model)
        web_search_mode = request.form.get('web_search_mode', 'smart')  # Default to 'smart'
        
        # Validate selected model
        if selected_model_for_request not in available_model_names and available_model_names:
            selected_model_for_request = default_model

        current_chat["model"] = selected_model_for_request
        
        # Handle regeneration
        is_regeneration = 'regenerate' in request.form and current_chat["history"]
        
        if is_regeneration:
            # Remove the last assistant message for regeneration
            if current_chat["history"] and current_chat["history"][-1]["role"] == "assistant":
                current_chat["history"].pop()
            
            # Get the last user prompt for regeneration
            last_user_msg = None
            for msg in reversed(current_chat["history"]):
                if msg["role"] == "user":
                    last_user_msg = msg
                    break
            
            if last_user_msg:
                prompt_to_use = last_user_msg["content"]
            else:
                prompt_to_use = ""
        else:
            prompt_to_use = prompt_from_input

        # Only proceed if there's a valid prompt
        if prompt_to_use:
            # Add user message if not regeneration
            if not is_regeneration:
                current_chat["history"].append({
                    "role": "user", 
                    "content": prompt_to_use, 
                    "timestamp": datetime.now().isoformat()
                })
            
            # Perform web search if needed
            search_results_str = ""
            if web_search_mode == 'on':
                print(f"--- Web search explicitly enabled for: {prompt_to_use[:50]}... ---")
                search_results_str = perform_web_search(prompt_to_use)
            elif web_search_mode == 'smart':
                # Smart search keywords
                smart_keywords = [
                    "latest", "recent", "today", "current", "now", "present", "currently",
                    "current news", "what happened", "breaking news", "update",
                    "who is", "what is", "where is", "when did", "how many",
                    "current president", "current leader", "current status", "current state",
                    "recent event", "recent development", "recent change", "recent update",
                    "latest version", "current version", "newest", "most recent"
                ]
                
                prompt_lower = prompt_to_use.lower()
                should_search = any(keyword in prompt_lower for keyword in smart_keywords)
                
                if should_search:
                    print(f"--- Smart search triggered for: {prompt_to_use[:50]}... ---")
                    search_results_str = perform_web_search(prompt_to_use)
            
            # Prepare messages for API call
            api_messages = []
            for msg in current_chat["history"]:
                if msg["role"] in ["user", "assistant"]:
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Prepend search results if available
            if search_results_str:
                if api_messages and api_messages[-1]["role"] == "user":
                    original_content = api_messages[-1]["content"]
                    api_messages[-1]["content"] = f"Web Search Results:\n{search_results_str}\n\nOriginal User Prompt:\n{original_content}"
            
            # Get LLM response
            try:
                print(f"--- Getting LLM response for model: {selected_model_for_request} ---")
                
                # Use asyncio to call the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response_content, provider_used = loop.run_until_complete(
                        get_llm_response(api_messages, selected_model_for_request)
                    )
                finally:
                    loop.close()
                
                # Add AI response
                current_chat["history"].append({
                    "role": "assistant", 
                    "content": response_content, 
                    "model": selected_model_for_request, 
                    "provider": provider_used, 
                    "timestamp": datetime.now().isoformat()
                })
                
                print(f"--- Successfully got response from {provider_used} ---")
                
            except Exception as e:
                error_msg = f"Error getting response: {str(e)}"
                print(f"--- {error_msg} ---")
                
                # Add error response
                current_chat["history"].append({
                    "role": "assistant", 
                    "content": f"I apologize, but I encountered an error: {error_msg}", 
                    "model": selected_model_for_request, 
                    "provider": "Error", 
                    "timestamp": datetime.now().isoformat()
                })
            
            # Auto-name chat if it's new
            if current_chat.get("name") == "New Chat" and any(msg["role"] == "user" for msg in current_chat["history"]):
                first_user_prompt = next((msg["content"] for msg in current_chat["history"] if msg["role"] == "user"), None)
                if first_user_prompt:
                    clean_prompt = ''.join(c for c in ' '.join(first_user_prompt.split()[:6]) if c.isalnum() or c.isspace()).strip()
                    timestamp_str = datetime.now().strftime("%b %d, %I:%M%p")
                    chat_name = f"{clean_prompt[:30]}... ({timestamp_str})" if clean_prompt else f"Chat ({timestamp_str})"
                    current_chat["name"] = chat_name

        # Save chat history
        save_chats(chats)
        return redirect(url_for('index'))

    # Prepare data for rendering (GET request or after POST redirect)
    history_html = ""
    for msg in current_chat.get("history", []):
        role_display = html.escape(msg["role"].title())
        timestamp_str = msg.get("timestamp", "")
        try:
            timestamp_display = datetime.fromisoformat(timestamp_str).strftime("%I:%M:%S %p") if timestamp_str else "No Time"
        except ValueError:
            timestamp_display = "Invalid Time"
        
        content_display = html.escape(msg["content"])
        model_display = f"<small>Model: {html.escape(msg.get('model', 'N/A'))}</small>" if msg["role"] == "assistant" else ""
        provider_display = f"<small>Provider: {html.escape(msg.get('provider', 'N/A'))}</small>" if msg["role"] == "assistant" else ""
        bg_color = '#f9f9f9' if msg['role'] == 'user' else '#e9f5ff'
        
        history_html += f'''<div style="margin:4px 0; padding:6px; border-bottom:1px solid #eee; background-color:{bg_color}; border-radius: 4px;">
                             <b>{role_display}</b> <small>({timestamp_display})</small> {provider_display}<br>{model_display}<br>
                             <div style="white-space: pre-wrap; word-wrap: break-word;">{content_display}</div>
                           </div>'''

    # Create model options for dropdown
    model_options_html = ''
    for model_name, provider_count, intel_index, resp_time in available_models_sorted_list:
        if not model_name or not model_name.strip():
            continue
        selected_attr = "selected" if model_name == current_model else ""
        
        # Format performance string
        if resp_time != float('inf') and intel_index > 0:
            perf_str = f"(Intel: {intel_index}, Time: {resp_time:.2f}s)"
        elif resp_time != float('inf'):
            perf_str = f"(Intel: N/A, Time: {resp_time:.2f}s)"
        elif intel_index > 0:
            perf_str = f"(Intel: {intel_index}, Time: N/A)"
        else:
            perf_str = "(Performance: N/A)"
        
        model_options_html += f'<option value="{model_name}" {selected_attr}>{model_name} {perf_str}</option>'

    # Get current web search mode (default to 'smart')
    current_web_search_mode = 'smart'  # Always default to smart

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>LLM Chat Interface - Enhanced</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; min-height: 100vh; }}
        .header {{ background-color: #2c3e50; color: white; padding: 15px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        #message-container {{ height: calc(100vh - 200px); overflow-y: auto; padding: 15px; border-bottom: 2px solid #ddd; }}
        #input-area {{ padding: 15px; background-color: #f8f9fa; border-top: 1px solid #ddd; }}
        select {{ width: 100%; padding: 10px; margin-bottom: 10px; font-size: 14px; border: 2px solid #ddd; border-radius: 6px; box-sizing: border-box; }}
        textarea {{ width: 100%; box-sizing: border-box; height: 80px; font-size: 14px; margin-bottom: 10px; padding: 12px; border: 2px solid #ddd; border-radius: 6px; resize: vertical; font-family: inherit; }}
        .form-row {{ display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }}
        .form-row label {{ font-weight: bold; margin-right: 10px; }}
        .form-row input[type="radio"] {{ margin-right: 5px; }}
        .button-row {{ display: flex; gap: 10px; margin-top: 10px; }}
        .button-row input {{ flex-grow: 1; padding: 12px; font-size: 14px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }}
        .button-row input[name="send"] {{ background-color: #27ae60; color: white; }}
        .button-row input[name="regenerate"] {{ background-color: #e67e22; color: white; }}
        .nav-buttons {{ display: flex; gap: 10px; margin-bottom: 15px; }}
        .nav-buttons button {{ padding: 10px 15px; border: 2px solid #3498db; border-radius: 6px; background-color: #ecf0f1; color: #2c3e50; cursor: pointer; font-weight: bold; }}
        .nav-buttons button:hover {{ background-color: #3498db; color: white; }}
        .message {{ margin: 8px 0; padding: 12px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .user-message {{ background-color: #e8f4fd; border-left-color: #3498db; }}
        .assistant-message {{ background-color: #f0f8f0; border-left-color: #27ae60; }}
        .timestamp {{ color: #7f8c8d; font-size: 12px; }}
        .model-info {{ color: #34495e; font-size: 12px; margin-top: 5px; }}
        .provider-info {{ color: #8e44ad; font-size: 12px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Advanced LLM Chat Interface</h1>
            <p style="margin: 5px 0; font-size: 14px;">Choose your model and start chatting!</p>
        </div>
        
        <div id="message-container">{history_html}</div>
        
        <div id="input-area">
            <form method="post" style="margin: 0;">
                <select name="model">{model_options_html}</select>
                
                <div class="nav-buttons">
                    <button type="submit" name="new_chat" value="1">🆕 New Chat</button>
                    <button type="submit" name="delete_chat" value="1" onclick="return confirm('Delete current chat?');">🗑️ Delete Chat</button>
                    <a href="/saved_chats" style="text-decoration: none;"><button type="button">💾 Saved Chats</button></a>
                </div>
                
                <div class="form-row">
                    <label>🔍 Web Search:</label>
                    <input type="radio" name="web_search_mode" value="off" {"checked" if current_web_search_mode == "off" else ""}> Off
                    <input type="radio" name="web_search_mode" value="smart" {"checked" if current_web_search_mode == "smart" else ""}> Smart
                    <input type="radio" name="web_search_mode" value="on" {"checked" if current_web_search_mode == "on" else ""}> Always On
                </div>
                
                <textarea name="prompt" placeholder="Type your message here..." autofocus></textarea>
                
                <div class="button-row">
                    <input type="submit" name="send" value="📤 Send Message">
                    {('<input type="submit" name="regenerate" value="🔄 Regenerate Response">') if current_chat.get("history") and any(msg["role"] == "assistant" for msg in current_chat["history"]) else ""}
                </div>
            </form>
        </div>
    </div>
    
    <script>
        // Auto-scroll to bottom
        var messageContainer = document.getElementById('message-container');
        messageContainer.scrollTop = messageContainer.scrollHeight;
        
        // Focus on textarea
        document.querySelector('textarea[name="prompt"]').focus();
        
        // Handle form submission
        document.querySelector('form').addEventListener('submit', function(e) {{
            var prompt = document.querySelector('textarea[name="prompt"]').value.trim();
            var isRegenerate = e.submitter && e.submitter.name === 'regenerate';
            
            if (!isRegenerate && !prompt) {{
                e.preventDefault();
                alert('Please enter a message');
                return;
            }}
            
            // Show loading state
            if (e.submitter && (e.submitter.name === 'send' || e.submitter.name === 'regenerate')) {{
                e.submitter.value = e.submitter.name === 'send' ? '⏳ Sending...' : '⏳ Regenerating...';
                e.submitter.disabled = true;
            }}
        }});
    </script>
</body>
</html>'''

@app.route('/saved_chats')
def saved_chats():
    """Display saved chats."""
    chats = load_chats()
    sorted_chat_items = sorted(chats.items(), key=lambda item: item[1].get('created_at', '1970-01-01T00:00:00'), reverse=True)
    
    chats_html = ""
    for chat_id, chat_data in sorted_chat_items:
        chat_name = html.escape(chat_data.get('name', 'Unnamed Chat'))
        created_at = chat_data.get('created_at', '1970-01-01T00:00:00')
        try:
            created_display = datetime.fromisoformat(created_at).strftime("%b %d, %Y - %I:%M %p")
        except ValueError:
            created_display = "Unknown Date"
        
        chats_html += f'''
        <div style="margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <a href="/load_chat/{chat_id}" style="text-decoration: none; color: #2c3e50; font-weight: bold; font-size: 16px;">{chat_name}</a>
                    <br><small style="color: #7f8c8d;">Created: {created_display}</small>
                </div>
                <form method="post" action="/delete_saved_chat/{chat_id}" style="display: inline;">
                    <button type="submit" onclick="return confirm('Delete this chat?');" 
                            style="background-color: #e74c3c; color: white; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer;">
                        🗑️ Delete
                    </button>
                </form>
            </div>
        </div>'''
    
    if not chats_html:
        chats_html = '<p style="text-align: center; color: #7f8c8d; font-style: italic; margin: 50px;">No saved chats found.</p>'
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Saved Chats - LLM Interface</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .back-button {{ display: inline-block; padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; border-radius: 6px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💾 Saved Chats</h1>
            <a href="/" class="back-button">⬅️ Back to Chat</a>
        </div>
        {chats_html}
    </div>
</body>
</html>'''

@app.route('/delete_saved_chat/<chat_id>', methods=['POST'])
def delete_saved_chat(chat_id):
    """Delete a saved chat."""
    chats = load_chats()
    if chat_id in chats:
        del chats[chat_id]
        save_chats(chats)
    return redirect(url_for('saved_chats'))

@app.route('/load_chat/<chat_id>')
def load_chat(chat_id):
    """Load a specific chat."""
    chats = load_chats()
    if chat_id in chats:
        session['current_chat'] = chat_id
        return redirect(url_for('index'))
    else:
        return redirect(url_for('saved_chats'))

def initialize_application():
    """Initialize the application."""
    print("\n" + "="*60)
    print("  INITIALIZING ENHANCED LLM CHAT APPLICATION")
    print("="*60)
    
    # Create required directories
    required_dirs = [
        os.path.join(APP_DIR, 'chats'),
        os.path.join(APP_DIR, 'logs'),
        os.path.join(APP_DIR, 'flask_session')
    ]
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"--- [INIT] Directory ready: {dir_path}")
        except Exception as e:
            print(f"--- [ERROR] Failed to create directory {dir_path}: {e}")
    
    # Initialize model cache
    print("\n--- [INIT] Initializing model cache...")
    initialize_model_cache()
    
    print(f"\n--- [INIT] API Keys Status:")
    print(f"  - Groq: {'✓' if GROQ_API_KEY else '✗'}")
    print(f"  - Cerebras: {'✓' if CEREBRAS_API_KEY else '✗'}")
    print(f"  - Google: {'✓' if GOOGLE_API_KEY else '✗'}")
    print(f"  - OpenRouter: {'✓' if OPENROUTER_API_KEY else '✗'}")
    
    print("\n" + "="*60)
    print("  APPLICATION INITIALIZATION COMPLETE")
    print("="*60 + "\n")

if __name__ == '__main__':
    try:
        # Initialize application
        initialize_application()
        
        # Start Flask server
        host = '0.0.0.0'
        port = 5000
        print(f"\n{'='*60}")
        print(f"  STARTING ENHANCED LLM CHAT APPLICATION")
        print(f"  - Host: {host}")
        print(f"  - Port: {port}")
        print(f"  - Models available: {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)}")
        print(f"  - Access at: http://localhost:{port}")
        print(f"{'='*60}\n")
        
        app.run(host=host, port=port, debug=True, use_reloader=False)
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print("  FATAL ERROR DURING APPLICATION STARTUP")
        print(f"{'!'*60}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\n{'!'*60}\n")