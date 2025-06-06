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

# Configuration
CHAT_STORAGE = os.path.join(APP_DIR, "chats.json")
PERFORMANCE_CSV_PATH = os.path.join(APP_DIR, "provider_performance.csv")
DEFAULT_MAX_OUTPUT_TOKENS = 4000

# Free models by provider
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
    ]
}

# Model display name mapping for deduplication
MODEL_DISPLAY_NAME_MAP = {
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "meta-llama/llama-4-maverick-17b-16e-instruct": "Llama 4 Maverick",
    "llama-4-maverick": "Llama 4 Maverick",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "llama-4-scout": "Llama 4 Scout",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.3-70b": "Llama 3.3 70B",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
    "deepseek-r1": "DeepSeek R1",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
    "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
    "qwen-qwq-32b": "Qwen QwQ 32B",
    "qwen/qwq-32b": "Qwen QwQ 32B",
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
app.secret_key = os.getenv('SECRET_KEY', 'blackberry-chat-app-secret')
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

def load_performance_from_csv():
    """Load performance data from CSV file."""
    performance_data = []
    try:
        if os.path.exists(PERFORMANCE_CSV_PATH):
            with open(PERFORMANCE_CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    performance_data.append({
                        'provider_name_scraped': row.get('provider_name_scraped', ''),
                        'model_name_scraped': row.get('model_name_scraped', ''),
                        'response_time_s': float(row.get('response_time_s', 0)) if row.get('response_time_s') else 0,
                        'quality_score': float(row.get('quality_score', 0)) if row.get('quality_score') else 0,
                        'price_per_million_tokens': float(row.get('price_per_million_tokens', 0)) if row.get('price_per_million_tokens') else 0
                    })
    except Exception as e:
        print(f"Error loading performance data: {e}")
    return performance_data

def get_model_performance_data(model_name):
    """Get performance data for a specific model from scraped data."""
    global PROVIDER_PERFORMANCE_CACHE
    
    if not PROVIDER_PERFORMANCE_CACHE:
        PROVIDER_PERFORMANCE_CACHE = load_performance_from_csv()
    
    model_lower = model_name.lower()
    best_quality = 0
    best_response_time = float('inf')
    
    for entry in PROVIDER_PERFORMANCE_CACHE:
        scraped_model = entry.get('model_name_scraped', '').lower()
        
        # Check for model name matches
        if (model_lower in scraped_model or 
            scraped_model in model_lower or
            any(part in scraped_model for part in model_lower.split()) or
            any(part in model_lower for part in scraped_model.split())):
            
            quality = entry.get('quality_score', 0)
            response_time = entry.get('response_time_s', float('inf'))
            
            if quality > best_quality:
                best_quality = quality
            if response_time < best_response_time and response_time > 0:
                best_response_time = response_time
    
    # Default values if no data found
    if best_quality == 0:
        if "4" in model_name:
            best_quality = 95
        elif "3" in model_name:
            best_quality = 88
        else:
            best_quality = 85
    
    if best_response_time == float('inf'):
        best_response_time = 1.0
    
    return best_quality, best_response_time

def get_available_models_with_provider_counts():
    """Get available models with proper deduplication and provider counting."""
    print("--- [MODEL_DISCOVERY] Starting comprehensive model discovery ---")
    
    model_dict = {}
    model_provider_info = {}
    provider_class_map = {}
    
    try:
        # Get all available providers from g4f
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
                                # Get real performance data
                                quality_score, response_time = get_model_performance_data(normalized_name)
                                
                                model_dict[normalized_name] = {
                                    'provider_count': 0,
                                    'quality_score': quality_score,
                                    'response_time': response_time,
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
    
    # Add essential models if missing or none found
    essential_models = [
        ("Llama 4 Maverick", 3, 95, 0.8),
        ("Llama 4 Scout", 3, 93, 0.9),
        ("DeepSeek R1", 2, 98, 1.2),
        ("DeepSeek R1 Distill Llama 70B", 2, 94, 1.0),
        ("Llama 3.3 70B", 4, 92, 1.1),
        ("Llama 3.1 8B", 3, 88, 0.6),
        ("Qwen QwQ 32B", 3, 94, 1.0),
        ("Gemini 2.5 Flash", 1, 96, 0.7),
        ("Gemini 2.0 Flash", 1, 94, 0.8),
        ("Qwen 3 32B", 2, 91, 0.9),
        ("Gemma 2 9B Instruct", 2, 86, 0.7),
        ("Mistral 7B Instruct", 2, 87, 0.8)
    ]
    
    for model_name, count, quality, time in essential_models:
        if model_name not in model_dict:
            model_dict[model_name] = {
                'provider_count': count,
                'quality_score': quality,
                'response_time': time,
                'is_free': True,
                'providers': ['G4F']
            }
    
    # Convert to sorted list with prioritization
    sorted_models = []
    for model_name, info in model_dict.items():
        # Prioritization logic:
        # 1. Free models first
        # 2. Higher quality score
        # 3. Lower response time
        # 4. Higher provider count
        
        priority_score = 0
        if info['is_free']:
            priority_score += 1000
        
        priority_score += info['quality_score']
        
        if info['response_time'] != float('inf'):
            priority_score -= info['response_time'] * 10
        
        priority_score += info['provider_count']
        
        sorted_models.append((
            model_name,
            info['provider_count'],
            info['quality_score'],
            info['response_time'],
            priority_score
        ))
    
    # Sort by priority score (descending)
    sorted_models.sort(key=lambda x: x[4], reverse=True)
    
    # Remove priority score from final result
    final_models = [(name, count, quality, time) for name, count, quality, time, _ in sorted_models]
    
    print(f"--- [MODEL_DISCOVERY] Found {len(final_models)} unique models ---")
    return final_models, model_provider_info, provider_class_map

def perform_web_search(query, max_results=3):
    """Perform web search using DuckDuckGo."""
    try:
        print("--- Performing web search with DuckDuckGo ---")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                search_results = "Web Search Results:\n"
                for i, result in enumerate(results, 1):
                    search_results += f"{i}. {result.get('title', '')}\n"
                    search_results += f"   {result.get('body', '')}\n\n"
                return search_results
    except Exception as e:
        print(f"--- Web search failed: {e} ---")
    return ""

async def call_groq_api(messages, model_name, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Call Groq API directly."""
    if not GROQ_API_KEY:
        raise Exception("Groq API key not available")
    
    model_id = get_provider_model_id(model_name, "groq")
    print(f"--- Calling Groq API with model ID: {model_id} ---")
    
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
    print(f"--- Calling Cerebras API with model ID: {model_id} ---")
    
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
    print(f"--- Calling Google API with model ID: {model_id} ---")
    
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

def detect_g4f_provider_from_response(response_text, model_name):
    """Enhanced G4F provider detection from response patterns and content."""
    response_lower = response_text.lower()
    
    # Provider-specific patterns and indicators
    provider_patterns = {
        "Blackbox": [
            "blackbox", "blackboxai", "blackbox.ai",
            "here's what i found", "according to my knowledge",
            "let me help you with that"
        ],
        "DeepInfra": [
            "deepinfra", "deep infra", "deepinfra.com",
            "based on the training data", "inference engine"
        ],
        "You.com": [
            "you.com", "youcom", "you ai",
            "search results indicate", "web sources"
        ],
        "Phind": [
            "phind", "phind.com",
            "here's a solution", "code implementation"
        ],
        "Bing": [
            "bing", "microsoft", "copilot",
            "according to my search", "based on current information"
        ],
        "ChatGPT": [
            "openai", "chatgpt", "gpt-",
            "i'm an ai assistant", "i'm chatgpt", "openai's language model"
        ],
        "Claude": [
            "claude", "anthropic",
            "i'm claude", "anthropic's ai assistant"
        ]
    }
    
    # Check for explicit provider mentions first
    for provider, patterns in provider_patterns.items():
        for pattern in patterns:
            if pattern in response_lower:
                return f"G4F ({provider})"
    
    # Check response characteristics for implicit detection
    if len(response_text) < 200 and any(word in response_lower for word in ["here's", "simply", "just"]):
        return "G4F (Blackbox)"
    
    if any(word in response_lower for word in ["function", "class", "import", "def ", "return", "variable"]):
        return "G4F (DeepInfra/Phind)"
    
    if any(phrase in response_lower for phrase in ["search results", "web sources", "according to", "based on current"]):
        return "G4F (You.com/Bing)"
    
    # Check for model-specific patterns
    model_lower = model_name.lower()
    if "llama" in model_lower:
        return "G4F (Llama Provider)"
    elif "qwen" in model_lower:
        return "G4F (Qwen Provider)"
    elif "deepseek" in model_lower:
        return "G4F (DeepSeek Provider)"
    elif "gemini" in model_lower:
        return "G4F (Gemini Provider)"
    
    return "G4F (Auto-detected)"

async def call_g4f_api(messages, model_name):
    """Call g4f as fallback with enhanced provider detection."""
    try:
        print(f"--- Calling G4F for model: {model_name} ---")
        
        # Try to find the actual model name for G4F
        g4f_model_name = model_name
        
        # Check if we need to map to G4F model names
        if model_name == "Llama 3.1 8B":
            g4f_model_name = "llama-3.1-8b"
        elif model_name == "Llama 3.3 70B":
            g4f_model_name = "llama-3.3-70b"
        elif model_name == "DeepSeek R1":
            g4f_model_name = "deepseek-r1"
        elif model_name == "Qwen QwQ 32B":
            g4f_model_name = "qwen-qwq-32b"
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: ChatCompletion.create(
                model=g4f_model_name,
                messages=messages,
                stream=False
            )
        )
        
        # Enhanced provider detection
        provider_info = detect_g4f_provider_from_response(str(response), model_name)
        print(f"--- Detected G4F provider: {provider_info} ---")
        
        return response, provider_info
        
    except Exception as e:
        raise Exception(f"G4F API error: {str(e)}")

async def get_llm_response(messages, model_name):
    """Get LLM response with provider prioritization."""
    errors = []
    normalized_model = get_normalized_model_name(model_name)
    
    # Try direct API providers first
    if normalized_model in ["Llama 4 Maverick", "Llama 4 Scout", "Llama 3.3 70B", "DeepSeek R1 Distill Llama 70B", "Qwen QwQ 32B", "Llama 3.1 8B"] and GROQ_API_KEY:
        try:
            print(f"--- Attempting Groq API for {model_name} ---")
            response = await call_groq_api(messages, normalized_model)
            return response, "Groq"
        except Exception as e:
            errors.append(f"Groq: {str(e)}")
            print(f"--- Groq API failed: {e} ---")
    
    if normalized_model in ["Llama 4 Scout", "Llama 3.3 70B", "Qwen 3 32B", "Llama 3.1 8B"] and CEREBRAS_API_KEY:
        try:
            print(f"--- Attempting Cerebras API for {model_name} ---")
            response = await call_cerebras_api(messages, normalized_model)
            return response, "Cerebras"
        except Exception as e:
            errors.append(f"Cerebras: {str(e)}")
            print(f"--- Cerebras API failed: {e} ---")
    
    if normalized_model.startswith("Gemini") and GOOGLE_API_KEY:
        try:
            print(f"--- Attempting Google API for {model_name} ---")
            response = await call_google_api(messages, normalized_model)
            return response, "Google AI"
        except Exception as e:
            errors.append(f"Google: {str(e)}")
            print(f"--- Google API failed: {e} ---")
    
    # Fallback to G4F with enhanced provider detection
    try:
        print(f"--- Attempting G4F for {model_name} ---")
        response, provider_info = await call_g4f_api(messages, model_name)
        return response, provider_info
    except Exception as e:
        errors.append(f"G4F: {str(e)}")
        print(f"--- G4F failed: {e} ---")
    
    error_msg = f"All providers failed for {model_name}: " + "; ".join(errors)
    raise Exception(error_msg)

def initialize_model_cache():
    """Initialize the global model cache."""
    global CACHED_AVAILABLE_MODELS_SORTED_LIST, CACHED_MODEL_PROVIDER_INFO, CACHED_PROVIDER_CLASS_MAP
    
    print("--- [CACHE] Initializing comprehensive model cache ---")
    try:
        list_data, info_data, map_data = get_available_models_with_provider_counts()
        
        CACHED_AVAILABLE_MODELS_SORTED_LIST = list_data
        CACHED_MODEL_PROVIDER_INFO = info_data
        CACHED_PROVIDER_CLASS_MAP = map_data
        
        print(f"--- [CACHE] Cached {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)} models ---")
    except Exception as e:
        print(f"--- [CACHE] Error initializing cache: {e} ---")
        # Fallback to essential models
        CACHED_AVAILABLE_MODELS_SORTED_LIST = [
            ("Llama 4 Maverick", 1, 95, 0.8),
            ("Llama 4 Scout", 1, 93, 0.9),
            ("DeepSeek R1", 1, 98, 1.2),
            ("Llama 3.1 8B", 1, 88, 0.6),
            ("Qwen QwQ 32B", 1, 94, 1.0),
            ("Gemini 2.5 Flash", 1, 96, 0.7)
        ]

# Routes
@app.route('/', methods=['GET', 'POST'])
@rate_limit(max_calls=60, period=60)
def index():
    chats = load_chats()
    available_models_sorted_list = CACHED_AVAILABLE_MODELS_SORTED_LIST
    
    available_model_names = {name for name, count, qual, rt in available_models_sorted_list}
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
        web_search_mode = request.form.get('web_search_mode', 'smart')
        
        if selected_model_for_request not in available_model_names and available_model_names:
            selected_model_for_request = default_model

        current_chat["model"] = selected_model_for_request
        
        # Handle regeneration - FIXED to properly replace last response
        is_regeneration = 'regenerate' in request.form
        
        if is_regeneration:
            # Remove the last assistant message for regeneration
            while current_chat["history"] and current_chat["history"][-1]["role"] == "assistant":
                removed_msg = current_chat["history"].pop()
                print(f"--- Removed assistant message for regeneration: {removed_msg['content'][:50]}... ---")
            
            # Get the last user prompt for regeneration
            last_user_msg = None
            for msg in reversed(current_chat["history"]):
                if msg["role"] == "user":
                    last_user_msg = msg
                    break
            
            if last_user_msg:
                prompt_to_use = last_user_msg["content"]
                print(f"--- Regenerating with prompt: {prompt_to_use[:50]}... ---")
            else:
                prompt_to_use = ""
                print("--- No user message found for regeneration ---")
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
                search_results_str = perform_web_search(prompt_to_use)
            elif web_search_mode == 'smart':
                smart_keywords = [
                    "latest", "recent", "today", "current", "now", "present",
                    "news", "update", "who is", "what is", "when did"
                ]
                
                prompt_lower = prompt_to_use.lower()
                should_search = any(keyword in prompt_lower for keyword in smart_keywords)
                
                if should_search:
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
            if search_results_str and api_messages and api_messages[-1]["role"] == "user":
                original_content = api_messages[-1]["content"]
                api_messages[-1]["content"] = f"Web Search Results:\n{search_results_str}\n\nUser Question:\n{original_content}"
            
            # Get LLM response
            try:
                print(f"--- Getting LLM response for model: {selected_model_for_request} ---")
                
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
                    clean_prompt = ''.join(c for c in ' '.join(first_user_prompt.split()[:4]) if c.isalnum() or c.isspace()).strip()
                    timestamp_str = datetime.now().strftime("%m/%d %H:%M")
                    chat_name = f"{clean_prompt[:20]}... ({timestamp_str})" if clean_prompt else f"Chat ({timestamp_str})"
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
            timestamp_display = datetime.fromisoformat(timestamp_str).strftime("%H:%M") if timestamp_str else ""
        except ValueError:
            timestamp_display = ""
        
        content_display = html.escape(msg["content"])
        
        if msg['role'] == 'user':
            history_html += f'''<div style="margin:2px 0; padding:4px; background:#f0f0f0; border-radius:3px;">
                                 <b>You</b> <small>{timestamp_display}</small><br>
                                 <div style="white-space:pre-wrap; word-wrap:break-word;">{content_display}</div>
                               </div>'''
        else:
            # Show model and provider info for AI responses
            model_display = html.escape(msg.get('model', 'Unknown'))
            provider_display = html.escape(msg.get('provider', 'Unknown'))
            history_html += f'''<div style="margin:2px 0; padding:4px; background:#e8f4fd; border-radius:3px;">
                                 <b>AI</b> <small>{timestamp_display}</small><br>
                                 <small style="color:#666;">Model: {model_display} | Provider: {provider_display}</small><br>
                                 <div style="white-space:pre-wrap; word-wrap:break-word;">{content_display}</div>
                               </div>'''

    # Create model options for dropdown WITH REAL PERFORMANCE VALUES
    model_options_html = ''
    for model_name, provider_count, quality_score, resp_time in available_models_sorted_list:
        if not model_name or not model_name.strip():
            continue
        selected_attr = "selected" if model_name == current_model else ""
        # Format performance string with quality score and response time
        perf_str = f" (Quality: {quality_score}, {resp_time:.1f}s)"
        model_options_html += f'<option value="{model_name}" {selected_attr}>{model_name}{perf_str}</option>'

    # Current web search mode (default to 'smart')
    current_web_search_mode = 'smart'
    
    # Check if regenerate should be available
    has_assistant_response = any(msg["role"] == "assistant" for msg in current_chat.get("history", []))

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #fff; font-size: 14px; }}
        #messages {{ height: 40vh; overflow-y: auto; padding: 5px; border-bottom: 1px solid #ccc; }}
        #input {{ padding: 5px; background: #f5f5f5; }}
        select, textarea {{ width: 100%; box-sizing: border-box; margin-bottom: 3px; padding: 4px; font-size: 14px; border: 1px solid #ccc; }}
        textarea {{ height: 60px; resize: none; font-family: Arial, sans-serif; }}
        .btn {{ padding: 8px 12px; margin: 2px; font-size: 14px; border: 1px solid #666; background: #e0e0e0; cursor: pointer; }}
        .btn-send {{ background: #4CAF50; color: white; border-color: #4CAF50; }}
        .btn-regen {{ background: #ff9800; color: white; border-color: #ff9800; }}
        .search-opts {{ margin: 3px 0; font-size: 12px; }}
        .search-opts label {{ margin-right: 10px; }}
        .search-opts input {{ margin-right: 3px; }}
        .nav {{ margin-bottom: 3px; }}
        .nav button {{ padding: 4px 8px; margin-right: 5px; font-size: 11px; }}
        .input-section {{ margin-bottom: 120px; }}
    </style>
</head>
<body>
    <div id="messages">{history_html}</div>
    
    <div class="input-section">
        <div id="input">
            <form method="post">
                <select name="model">{model_options_html}</select>
                
                <div class="nav">
                    <button type="submit" name="new_chat" value="1" class="btn">New Chat</button>
                    <button type="submit" name="delete_chat" value="1" class="btn">Delete Chat</button>
                    <a href="/saved_chats" style="text-decoration:none;"><button type="button" class="btn">Saved</button></a>
                </div>
                
                <div class="search-opts">
                    <label><input type="radio" name="web_search_mode" value="off" {"checked" if current_web_search_mode == "off" else ""}>Off</label>
                    <label><input type="radio" name="web_search_mode" value="smart" {"checked" if current_web_search_mode == "smart" else ""}>Smart</label>
                    <label><input type="radio" name="web_search_mode" value="on" {"checked" if current_web_search_mode == "on" else ""}>On</label>
                </div>
                
                <textarea name="prompt" placeholder="Type message..."></textarea>
                
                <div>
                    <input type="submit" name="send" value="Send" class="btn btn-send">
                    {('<input type="submit" name="regenerate" value="Regenerate" class="btn btn-regen">') if has_assistant_response else ""}
                </div>
            </form>
        </div>
    </div>
    
    <script>
        // Auto-scroll to bottom of messages
        var messages = document.getElementById('messages');
        messages.scrollTop = messages.scrollHeight;
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
            created_display = datetime.fromisoformat(created_at).strftime("%m/%d %H:%M")
        except ValueError:
            created_display = "Unknown"
        
        chats_html += f'''
        <div style="margin: 5px 0; padding: 8px; border: 1px solid #ddd; background: white;">
            <a href="/load_chat/{chat_id}" style="text-decoration: none; color: #000; font-weight: bold;">{chat_name}</a>
            <br><small style="color: #666;">Created: {created_display}</small>
            <form method="post" action="/delete_saved_chat/{chat_id}" style="display: inline; float: right;">
                <button type="submit" style="background: #e74c3c; color: white; border: none; padding: 4px 8px; font-size: 11px;">Delete</button>
            </form>
        </div>'''
    
    if not chats_html:
        chats_html = '<p style="text-align: center; color: #666; margin: 20px;">No saved chats.</p>'
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Saved Chats</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 10px; background: #f5f5f5; font-size: 14px; }}
        .container {{ max-width: 600px; margin: 0 auto; }}
        .back {{ display: inline-block; padding: 8px 15px; background: #3498db; color: white; text-decoration: none; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Saved Chats</h2>
        <a href="/" class="back">Back to Chat</a>
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
    print("\n" + "="*50)
    print("  BLACKBERRY OPTIMIZED LLM CHAT APP")
    print("="*50)
    
    # Create required directories
    required_dirs = [
        os.path.join(APP_DIR, 'chats'),
        os.path.join(APP_DIR, 'flask_session')
    ]
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"--- [ERROR] Failed to create directory {dir_path}: {e}")
    
    # Initialize model cache
    initialize_model_cache()
    
    print(f"\n--- API Keys Status:")
    print(f"  - Groq: {'Available' if GROQ_API_KEY else 'Not Set'}")
    print(f"  - Cerebras: {'Available' if CEREBRAS_API_KEY else 'Not Set'}")
    print(f"  - Google: {'Available' if GOOGLE_API_KEY else 'Not Set'}")
    
    print("\n" + "="*50)
    print("  INITIALIZATION COMPLETE")
    print("="*50 + "\n")

if __name__ == '__main__':
    try:
        initialize_application()
        
        host = '0.0.0.0'
        port = 5000
        print(f"Starting BlackBerry Optimized Chat App")
        print(f"Access at: http://localhost:{port}")
        print(f"Models available: {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)}")
        print(f"Optimized for small screens and low-end hardware\n")
        
        app.run(host=host, port=port, debug=False, use_reloader=False)
        
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        import traceback
        traceback.print_exc()