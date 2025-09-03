from g4f import ChatCompletion
from g4f.models import ModelUtils
import g4f
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import os
import json
import time
import csv
import requests
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
import html
import uuid
import re
import threading
import queue
from duckduckgo_search import DDGS

# Try to import serpapi
try:
    import serpapi
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    print("Warning: serpapi not installed, SerpAPI functionality will be disabled")

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
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

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

# Chat storage
CHAT_STORAGE = os.path.join(os.path.dirname(__file__), "chats.json")

# Provider-specific model ID mapping
# This maps from display name to provider-specific model IDs
PROVIDER_MODEL_MAP = {
    "Llama 4 Maverick": {
        "cerebras": "cerebras/llama-4-maverick-17b-16e-instruct",
        "groq": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "google_ai": "SKIP_PROVIDER",
        "g4f": "meta-llama/llama-4-maverick-17b-128e-instruct"
    },
    "Llama 4 Scout": {
        "cerebras": "cerebras/llama-4-scout-17b-16e-instruct",
        "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
        "google_ai": "SKIP_PROVIDER",
        "g4f": "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    "Llama 3.3 70B": {
        "cerebras": "cerebras/llama-3.3-70b-instruct",
        "groq": "llama3-70b-8192",
        "google_ai": "SKIP_PROVIDER",
        "g4f": "meta-llama/llama-3.3-70b-instruct"
    },
    "Llama 3.1 8B": {
        "cerebras": "cerebras/llama-3.1-8b-instruct",
        "groq": "llama3-8b-8192",
        "google_ai": "SKIP_PROVIDER",
        "g4f": "meta-llama/llama-3.1-8b-instruct"
    },
    "Qwen 3 32B (Reasoning)": {
        "cerebras": "qwen-3-32b",
        "groq": "qwen/qwen3-32b",
        "google_ai": "SKIP_PROVIDER",
        "g4f": "qwen-3-32b"
    },
    "Gemini 2.5 Flash": {
        "cerebras": "SKIP_PROVIDER",
        "groq": "SKIP_PROVIDER",
        "google_ai": "gemini-2.5-flash",
        "g4f": "gemini-2.5-flash"
    },
    "Gemini 2.5 Flash Lite": {
        "cerebras": "SKIP_PROVIDER",
        "groq": "SKIP_PROVIDER",
        "google_ai": "gemini-2.5-flash-lite",
        "g4f": "gemini-2.5-flash-lite"
    },
    "Gemini 2.0 Flash": {
        "cerebras": "SKIP_PROVIDER",
        "groq": "SKIP_PROVIDER",
        "google_ai": "gemini-2.0-flash",
        "g4f": "gemini-2.0-flash"
    },
    "Gemini 1.5 Flash": {
        "cerebras": "SKIP_PROVIDER",
        "groq": "SKIP_PROVIDER",
        "google_ai": "gemini-1.5-flash",
        "g4f": "gemini-1.5-flash"
    },
    "Gemini 1.5 Pro": {
        "cerebras": "SKIP_PROVIDER",
        "groq": "SKIP_PROVIDER",
        "google_ai": "gemini-1.5-pro",
        "g4f": "gemini-1.5-pro"
    },
    "Gemini 2.0 Flash Lite": {
        "cerebras": "SKIP_PROVIDER",
        "groq": "SKIP_PROVIDER",
        "google_ai": "gemini-2.0-flash-lite",
        "g4f": "gemini-2.0-flash-lite"
    }
}

def update_performance_csv(force_update=False):
    """Update the performance CSV with latest data from the leaderboard."""
    print("--- Checking for latest performance data updates ---")
    
    # Paths for the leaderboard data
    leaderboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLM-Performance-Leaderboard')
    leaderboard_latest_path = os.path.join(leaderboard_dir, 'llm_leaderboard_latest.csv')
    
    # Check if leaderboard data exists
    if not os.path.exists(leaderboard_latest_path):
        print(f"--- Warning: Latest leaderboard CSV not found at {leaderboard_latest_path} ---")
        return False
    
    try:
        # Check if we need to update (compare file modification times)
        csv_mod_time = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0
        leaderboard_mod_time = os.path.getmtime(leaderboard_latest_path)
        
        if not force_update and leaderboard_mod_time <= csv_mod_time:
            print("--- Performance CSV is up to date ---")
            return True
        
        print(f"--- Updating performance CSV from {leaderboard_latest_path} ---")
        convert_leaderboard_to_provider_performance(leaderboard_latest_path, CSV_PATH)
        return True
        
    except Exception as e:
        print(f"--- Error updating performance CSV: {e} ---")
        return False

def refresh_performance_data():
    """Force refresh of performance data (update CSV and reload into memory)."""
    print("--- Force refreshing performance data ---")
    success = update_performance_csv(force_update=True)
    if success:
        load_performance_data()
        print(f"--- Performance data refreshed: {len(PERFORMANCE_DATA)} entries loaded ---")
        return True
    return False

def convert_leaderboard_to_provider_performance(leaderboard_path, output_path):
    """Convert the leaderboard data to provider_performance.csv format."""
    print(f"Converting {leaderboard_path} to {output_path}...")
    
    # Read the leaderboard data
    leaderboard_data = []
    with open(leaderboard_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            leaderboard_data.append(row)
    
    # Convert to provider_performance.csv format
    provider_performance_data = []
    for row in leaderboard_data:
        provider_name = row.get('API Provider', '')
        model_name = row.get('Model', '')
        
        if not provider_name or not model_name:
            continue
        
        # Parse context window
        context_window = row.get('ContextWindow', '0')
        if context_window.lower().endswith('k'):
            context_window = int(float(context_window[:-1]) * 1000)
        elif context_window.lower().endswith('m'):
            context_window = int(float(context_window[:-1]) * 1000000)
        else:
            try:
                context_window = int(context_window)
            except (ValueError, TypeError):
                context_window = 0
        
        # Parse intelligence index
        intelligence_index = row.get('Artificial AnalysisIntelligence Index', 'N/A')
        if intelligence_index != 'N/A':
            try:
                intelligence_index = float(intelligence_index)
            except (ValueError, TypeError):
                intelligence_index = 0
        else:
            intelligence_index = 0
        
        # Parse response time
        response_time = row.get('Total Response (s)', 'N/A')
        if response_time != 'N/A':
            try:
                response_time = float(response_time)
            except (ValueError, TypeError):
                response_time = 999
        else:
            response_time = 999
        
        # Parse tokens per second
        tokens_per_s = row.get('MedianTokens/s', 'N/A')
        if tokens_per_s != 'N/A':
            tokens_per_s = tokens_per_s.replace(',', '')
            try:
                tokens_per_s = float(tokens_per_s)
            except (ValueError, TypeError):
                tokens_per_s = 0
        else:
            tokens_per_s = 0
        
        # Create a record
        record = {
            'provider_name_scraped': provider_name,
            'model_name_scraped': model_name,
            'context_window': context_window,
            'intelligence_index': intelligence_index,
            'response_time_s': response_time,
            'tokens_per_s': tokens_per_s,
            'source_url': 'https://artificialanalysis.ai/leaderboards/providers',
            'last_updated_utc': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'is_free_source': 'true'
        }
        
        provider_performance_data.append(record)
    
    # Write to provider_performance.csv
    fieldnames = [
        'provider_name_scraped', 'model_name_scraped', 'context_window',
        'intelligence_index', 'response_time_s', 'tokens_per_s',
        'source_url', 'last_updated_utc', 'is_free_source'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(provider_performance_data)
    
    print(f"--- Converted {len(provider_performance_data)} models to {output_path} ---")

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
            'gemini-2.0-flash-lite',
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
    normalized = normalized.replace('-versatile', '').replace('-preview', '')
    normalized = normalized.replace('_', ' ').replace('-', ' ')
    # Remove extra spaces
    normalized = ' '.join(normalized.split())
    return normalized

def find_performance_data_for_model(model_name):
    """Find the best performance data for a model across all providers."""
    normalized_model = normalize_model_name(model_name)
    
    best_intelligence = 0
    best_response_time = 999
    found_match = False
    
    for key, data in PERFORMANCE_DATA.items():
        stored_model = normalize_model_name(data['model'])
        
        # Check for exact match or partial match
        if (normalized_model == stored_model or 
            normalized_model in stored_model or 
            stored_model in normalized_model or
            # Check for key components match (like "llama 3.3 70b")
            all(part in stored_model for part in normalized_model.split() if len(part) > 1)):
            
            found_match = True
            intelligence = data['intelligence']
            response_time = data['response_time']
            
            # Keep the best intelligence score, or if tied, the fastest response time
            if intelligence > best_intelligence or (intelligence == best_intelligence and response_time < best_response_time):
                best_intelligence = intelligence
                best_response_time = response_time
    
    if found_match:
        return best_intelligence, best_response_time
    else:
        return 0, 999

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
                'display_name': get_canonical_model_name(model),
                'providers': [],
                'tier': get_model_tier(model)
            }
        if 'cerebras' not in all_models[model]['providers']:
            all_models[model]['providers'].append('cerebras')
    
    for model in groq_models:
        if model not in all_models:
            all_models[model] = {
                'name': model,
                'display_name': get_canonical_model_name(model),
                'providers': [],
                'tier': get_model_tier(model)
            }
        if 'groq' not in all_models[model]['providers']:
            all_models[model]['providers'].append('groq')
    
    for model in google_models:
        if model not in all_models:
            all_models[model] = {
                'name': model,
                'display_name': get_canonical_model_name(model),
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
                'display_name': get_canonical_model_name(model),
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
    
    # Deduplicate by display name - keep the best performing version of each model
    display_name_map = {}
    for model_data in all_models.values():
        display_name = model_data['display_name']
        
        if display_name not in display_name_map:
            display_name_map[display_name] = model_data
        else:
            # Keep the one with better performance (lower response time)
            existing = display_name_map[display_name]
            if model_data['best_response_time'] < existing['best_response_time']:
                # Merge providers from both
                merged_providers = list(set(existing['providers'] + model_data['providers']))
                model_data['providers'] = merged_providers
                display_name_map[display_name] = model_data
            else:
                # Keep existing but merge providers
                merged_providers = list(set(existing['providers'] + model_data['providers']))
                existing['providers'] = merged_providers
    
    # Convert to list and sort
    model_list = list(display_name_map.values())
    
    # Sort by tier, then by best response time
    model_list.sort(key=lambda x: (x['tier'], x['best_response_time']))
    
    # Cache results
    CACHED_MODELS = model_list
    CACHED_TIMESTAMP = time.time()
    
    print(f"--- Successfully loaded {len(model_list)} models total (after deduplication) ---")
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

def get_canonical_model_name(model_name):
    """Get canonical display name for a model, handling variants."""
    model_lower = model_name.lower()
    
    # Llama 4 variants
    if 'llama-4-maverick' in model_lower or 'llama 4 maverick' in model_lower:
        return 'Llama 4 Maverick'
    elif 'llama-4-scout' in model_lower or 'llama 4 scout' in model_lower:
        return 'Llama 4 Scout'
    
    # Llama 3.3 variants
    elif 'llama-3.3-70b' in model_lower or 'llama 3.3 70b' in model_lower:
        return 'Llama 3.3 70B'
    elif 'llama-3.3-8b' in model_lower or 'llama 3.3 8b' in model_lower:
        return 'Llama 3.3 8B'
    
    # Llama 3.1 variants
    elif 'llama-3.1-70b' in model_lower or 'llama 3.1 70b' in model_lower:
        return 'Llama 3.1 70B'
    elif 'llama-3.1-8b' in model_lower or 'llama 3.1 8b' in model_lower:
        return 'Llama 3.1 8B'
    
    # Llama 3 variants
    elif 'llama-3-70b' in model_lower or 'llama 3 70b' in model_lower:
        return 'Llama 3 70B'
    elif 'llama-3-8b' in model_lower or 'llama 3 8b' in model_lower:
        return 'Llama 3 8B'
    
    # Gemini variants
    elif 'gemini-2.5-flash-lite' in model_lower:
        return 'Gemini 2.5 Flash Lite'
    elif 'gemini-2.5-flash' in model_lower:
        return 'Gemini 2.5 Flash'
    elif 'gemini-2.0-flash-lite' in model_lower:
        return 'Gemini 2.0 Flash Lite'
    elif 'gemini-2.0-flash' in model_lower:
        return 'Gemini 2.0 Flash'
    elif 'gemini-1.5-flash' in model_lower:
        return 'Gemini 1.5 Flash'
    elif 'gemini-1.5-pro' in model_lower:
        return 'Gemini 1.5 Pro'
    
    # DeepSeek variants
    elif 'deepseek-r1-distill-llama-70b' in model_lower:
        return 'DeepSeek R1 Distill Llama 70B'
    elif 'deepseek-r1-distill' in model_lower:
        return 'DeepSeek R1 Distill'
    elif 'deepseek-r1' in model_lower:
        return 'DeepSeek R1'
    
    # Qwen variants
    elif 'qwen-3-32b' in model_lower or 'qwen3-32b' in model_lower:
        if 'reasoning' in model_lower:
            return 'Qwen 3 32B (Reasoning)'
        else:
            return 'Qwen 3 32B'
    elif 'qwq-32b' in model_lower:
        return 'QwQ-32B'
    
    # Claude variants
    elif 'claude-3.5-sonnet' in model_lower:
        return 'Claude 3.5 Sonnet'
    elif 'claude-3-opus' in model_lower:
        return 'Claude 3 Opus'
    elif 'claude-3-sonnet' in model_lower:
        return 'Claude 3 Sonnet'
    elif 'claude-3-haiku' in model_lower:
        return 'Claude 3 Haiku'
    
    # Fallback to formatted name
    return format_model_name(model_name)

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

def get_provider_specific_model_id(model_name, provider_name):
    """
    Find a provider-specific model ID for a given model name and provider.
    
    Args:
        model_name (str): The model name to find a provider-specific ID for
        provider_name (str): The provider name to find a model ID for
        
    Returns:
        str: The provider-specific model ID, or the original model name if none is found
    """
    # Get the canonical display name first
    display_name = get_canonical_model_name(model_name)
    
    # First check the static mapping using the display name
    if display_name in PROVIDER_MODEL_MAP and provider_name.lower() in PROVIDER_MODEL_MAP[display_name]:
        result = PROVIDER_MODEL_MAP[display_name][provider_name.lower()]
        if result == "SKIP_PROVIDER":
            return None
        print(f"--- Found model mapping for {model_name} ({display_name}) on {provider_name}: {result} ---")
        return result
    
    # Also check if any key in PROVIDER_MODEL_MAP contains the model name as a substring
    for key in PROVIDER_MODEL_MAP:
        if model_name.lower() in key.lower() or key.lower() in model_name.lower():
            if provider_name.lower() in PROVIDER_MODEL_MAP[key]:
                result = PROVIDER_MODEL_MAP[key][provider_name.lower()]
                if result == "SKIP_PROVIDER":
                    return None
                print(f"--- Found partial match mapping for {model_name} using {key} on {provider_name}: {result} ---")
                return result
    
    # If no specific mapping found, return the original model name
    return model_name

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
            # Get provider-specific model ID
            provider_model_id = get_provider_specific_model_id(model_name, provider)
            if provider_model_id is None:
                print(f"--- Skipping {provider} for {model_name} (marked as SKIP_PROVIDER) ---")
                continue
                
            if provider == 'cerebras' and CEREBRAS_API_KEY:
                response = make_cerebras_request(provider_model_id, messages, temperature, max_tokens)
                return response, provider, model_name
            elif provider == 'groq' and GROQ_API_KEY:
                response = make_groq_request(provider_model_id, messages, temperature, max_tokens)
                return response, provider, model_name
            elif provider == 'google_ai' and GOOGLE_API_KEY:
                response = make_google_ai_request(provider_model_id, messages, temperature, max_tokens)
                return response, provider, model_name
            elif provider == 'g4f':
                response = make_g4f_request(provider_model_id, messages, temperature, max_tokens)
                return response, provider, model_name
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
        timeout=120
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
        timeout=120
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
    """Make request using g4f with timeout handling."""
    
    # Create a queue to get the result from the thread
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    def g4f_request():
        try:
            response = ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            result_queue.put(response)
        except Exception as e:
            error_queue.put(e)
    
    # Start the request in a separate thread
    thread = threading.Thread(target=g4f_request)
    thread.daemon = True
    thread.start()
    
    # Wait for result with timeout
    thread.join(timeout=90)  # 90 seconds timeout
    
    if thread.is_alive():
        # Thread is still running, request timed out
        raise Exception("G4F request timed out after 90 seconds")
    
    # Check for errors
    if not error_queue.empty():
        raise error_queue.get()
    
    # Check for result
    if not result_queue.empty():
        return result_queue.get()
    else:
        raise Exception("G4F request completed but returned no result")

def perform_web_search(query, max_results=5):
    """
    Perform a web search using multiple fallback methods.
    Tries DuckDuckGo first, then SerpAPI (if available and has credits),
    then Brave Search API (if available), then free fallback methods (Bing, Google scraping).
    """
    results = []
    
    # Truncate query if too long for DuckDuckGo (limit appears to be around 500 characters)
    ddg_query = query
    if len(query) > 200:  # More conservative limit to avoid rate limiting
        ddg_query = query[:200] + "..."
        print(f"--- Truncated long query for DuckDuckGo: {len(query)} -> {len(ddg_query)} chars ---")
    
    # Method 1: Try DuckDuckGo first (most reliable free option)
    try:
        print(f"--- Performing DuckDuckGo search for: {ddg_query} ---")
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(ddg_query, max_results=max_results))
            for result in ddg_results:
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
        print(f"--- DuckDuckGo search completed. Found {len(results)} results. ---")
        if results:
            return results
    except Exception as e:
        print(f"--- DuckDuckGo search failed: {e} ---")
    
    # Method 2: Try SerpAPI if available and configured
    if SERPAPI_API_KEY and SERPAPI_AVAILABLE:
        try:
            print(f"--- Falling back to SerpApi search for: {query} ---")
            search = serpapi.GoogleSearch({
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": max_results
            })
            search_results = search.get_dict()
            
            # Check for error messages indicating quota exceeded
            if 'error' in search_results:
                error_message = search_results['error']
                print(f"--- SerpAPI error: {error_message} ---")
                if 'quota' in error_message.lower() or 'limit' in error_message.lower() or 'credit' in error_message.lower():
                    print("--- SerpAPI quota/credits exhausted, trying other alternatives ---")
                else:
                    print("--- SerpAPI error, trying other alternatives ---")
            elif 'organic_results' in search_results:
                for result in search_results['organic_results'][:max_results]:
                    results.append({
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'snippet': result.get('snippet', '')
                    })
                print(f"--- SerpAPI search completed. Found {len(results)} results. ---")
                if results:
                    return results
        except Exception as e:
            print(f"--- SerpAPI search failed: {e} ---")
    
    # Method 3: Try Brave Search API if available
    if BRAVE_API_KEY:
        try:
            print(f"--- Falling back to Brave Search API for: {query} ---")
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': BRAVE_API_KEY
            }
            params = {
                'q': query,
                'count': max_results
            }
            response = requests.get('https://api.search.brave.com/res/v1/web/search', 
                                  headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'web' in data and 'results' in data['web']:
                    for result in data['web']['results'][:max_results]:
                        results.append({
                            'title': result.get('title', ''),
                            'link': result.get('url', ''),
                            'snippet': result.get('description', '')
                        })
                    print(f"--- Brave Search API completed. Found {len(results)} results. ---")
                    if results:
                        return results
            else:
                print(f"--- Brave Search API error: {response.status_code} ---")
        except Exception as e:
            print(f"--- Brave Search API failed: {e} ---")
    
    print("--- All web search methods failed ---")
    return results

def should_perform_smart_search(prompt):
    """Determine if a prompt warrants a web search in smart mode."""
    prompt_lower = prompt.lower()
    
    # Time-sensitive keywords - clear indicators of needing current information
    time_keywords = [
        "latest", "recent", "today", "current", "now", "present", "currently",
        "this year", "2024", "2025", "yesterday", "last week", "this week",
        "breaking", "news", "update", "updated", "newest", "new", "fresh"
    ]
    
    # Question patterns that likely need current info (more specific)
    question_patterns = [
        "what happened", "who is the current", "who's the current", "what's the current",
        "current president", "current leader", "current ceo", "current price",
        "latest version", "latest release", "how much does", "price of",
        "stock price", "weather in", "time in", "population of", "who is president",
        "who is the president", "what is the current", "what's happening"
    ]
    
    # Technology and current events keywords
    tech_keywords = [
        "version", "release", "announcement", "launched", "discontinued",
        "merged", "acquired", "ipo", "earnings", "quarterly", "revenue"
    ]
    
    # Sports and entertainment keywords
    sports_keywords = [
        "score", "game", "match", "season", "championship", "winner",
        "standings", "tournament", "playoffs", "draft", "trade"
    ]
    
    # Financial and market keywords
    finance_keywords = [
        "stock", "shares", "market", "trading", "exchange rate",
        "crypto", "bitcoin", "inflation", "interest rate", "gdp"
    ]
    
    # Exclusion patterns - things that should NOT trigger search even if they contain keywords
    exclusion_patterns = [
        r'\bwhat is \d+[\+\-\*\/]\d+',  # Basic math like "what is 2+2"
        r'\bwhat is the capital of',     # Geography basics
        r'\bwhat is the meaning of life', # Philosophical questions
        r'\bexplain (how to|what is)',   # General explanations
        r'\bhow to (write|code|program|make|cook|do)', # How-to questions
        r'\bwhat does .* mean',          # Definition questions
        r'\btell me (a joke|about)',     # Entertainment requests
    ]
    
    # Check if query matches exclusion patterns first
    is_excluded = any(re.search(pattern, prompt_lower) for pattern in exclusion_patterns)
    
    if is_excluded:
        return False
    
    # Check all keyword categories
    all_keywords = time_keywords + question_patterns + tech_keywords + sports_keywords + finance_keywords
    
    # Basic keyword matching
    if any(keyword in prompt_lower for keyword in all_keywords):
        return True
    
    # Check for year references (2020-2030)
    if re.search(r'\b(202[0-9])\b', prompt_lower):
        return True
    
    # Check for "when did" or "when was" questions
    if re.search(r'\b(when (did|was|is|will)|how (long|much|many))\b', prompt_lower):
        return True
    
    return False

# Chat management functions
def load_chats():
    """Load all chats from storage."""
    if os.path.exists(CHAT_STORAGE):
        try:
            with open(CHAT_STORAGE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("--- Warning: Chat storage file is corrupted. Starting fresh. ---")
            return {}
        except Exception as e:
            print(f"--- Error loading chats: {e} ---")
            return {}
    return {}

def save_chats(chats):
    """Save all chats to storage."""
    try:
        with open(CHAT_STORAGE, 'w', encoding='utf-8') as f:
            json.dump(chats, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"--- Error saving chats: {e} ---")

def get_current_chat():
    """Get the current chat from session."""
    chat_id = session.get('current_chat')
    if not chat_id:
        # Create new chat
        chat_id = str(uuid.uuid4())
        session['current_chat'] = chat_id
    
    chats = load_chats()
    if chat_id not in chats:
        chats[chat_id] = {
            'name': 'New Chat',
            'history': [],
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat()
        }
        save_chats(chats)
    
    return chats[chat_id]

def update_chat(chat_data):
    """Update the current chat in storage."""
    chat_id = session.get('current_chat')
    if chat_id:
        chats = load_chats()
        chats[chat_id] = chat_data
        chats[chat_id]['last_modified'] = datetime.now().isoformat()
        save_chats(chats)

# Flask routes
@app.route('/')
def index():
    """Main chat interface."""
    current_chat = get_current_chat()
    models = get_available_models()
    
    # Get selected model from session or use first available
    selected_model = session.get('user_selected_model')
    if not selected_model and models:
        selected_model = models[0]['name']
        session['user_selected_model'] = selected_model
    
    # Build model options HTML
    model_options_html = ""
    for model in models:
        selected_attr = 'selected' if model['name'] == selected_model else ''
        
        # Get best performance data for this model
        best_intelligence, best_response_time = find_performance_data_for_model(model['name'])
        
        # Format the display string
        intelligence_str = f"{int(best_intelligence)}" if best_intelligence > 0 else "N/A"
        time_str = f"{best_response_time:.2f}s" if best_response_time < 999 else "N/A"
        model_options_html += f'<option value="{model["name"]}" {selected_attr}>{model["display_name"]} ({intelligence_str}, {time_str})</option>'
    
    # Build chat history HTML
    history_html = ""
    for msg in current_chat.get('history', []):
        # Format timestamp
        try:
            if 'timestamp' in msg:
                timestamp_dt = datetime.fromisoformat(msg['timestamp'])
                timestamp_display = timestamp_dt.strftime("%b %d, %Y - %I:%M %p")
            else:
                timestamp_display = ""
        except:
            timestamp_display = ""
        
        content_display = html.escape(msg["content"])
        
        if msg['role'] == 'user':
            role_display = "You"
            history_html += f'''<div class="message user-message">
                                 <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                    <b>{role_display}</b>
                                    <small>{timestamp_display}</small>
                                 </div>
                                 <div style="white-space: pre-wrap; word-wrap: break-word;">{content_display}</div>
                               </div>'''
        elif msg['role'] == 'assistant':
            role_display = "Assistant"
            # Create message metadata for assistant
            metadata = []
            if msg.get('model', 'N/A') != 'N/A':
                metadata.append(f"Model: {html.escape(msg.get('model', 'N/A'))}")
            if msg.get('provider', 'N/A') != 'N/A':
                metadata.append(f"Provider: {html.escape(msg.get('provider', 'N/A'))}")
            
            metadata_display = f"<small>{' | '.join(metadata)}</small>" if metadata else ""
            
            history_html += f'''<div class="message assistant-message">
                                 <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                    <b>{role_display}</b>
                                    <small>{timestamp_display}</small>
                                 </div>
                                 {f'<div style="margin-bottom: 4px; font-size: 0.85em; color: #666;">{metadata_display}</div>' if metadata_display else ''}
                                 <div style="white-space: pre-wrap; word-wrap: break-word;">{content_display}</div>
                               </div>'''
    
    # Navigation links
    nav_links_html = '''
        <div class="nav-links">
            <a href="/saved_chats" style="padding: 8px 15px; border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333; text-decoration: none;">Saved Chats</a>
            <button type="submit" name="new_chat" value="1" form="chat-form" style="padding: 8px 15px; border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333; cursor: pointer;">New Chat</button>
        </div>
    '''
    
    # Web search controls
    web_search_html = '''
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="margin-right: 10px; font-weight: bold; white-space: nowrap;">Web Search:</span>
            <div style="display: flex; flex-wrap: wrap;">
                <label style="margin-right: 15px; white-space: nowrap;">
                    <input type="radio" name="web_search_mode" value="off"> Off
                </label>
                <label style="margin-right: 15px; white-space: nowrap;">
                    <input type="radio" name="web_search_mode" value="smart" checked> Smart
                </label>
                <label style="white-space: nowrap;">
                    <input type="radio" name="web_search_mode" value="on"> On
                </label>
            </div>
        </div>
    '''
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Hybrid LLM Chat Interface</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0 0 50px 0; background-color: #fff; min-height: 100vh; display: flex; flex-direction: column; }}
        #top-controls {{ position: sticky; top: 0; padding: 8px; background-color: #f8f9fa; border-bottom: 1px solid #ccc; z-index: 100; }}
        #message-container {{ flex: 1; overflow-y: auto; padding: 5px 10px; margin-bottom: 10px; min-height: 50vh; max-height: 60vh; padding-top: 15px; }}
        .message {{ margin-bottom: 15px; }}
        .user-message {{ background-color: #f0f8ff; padding: 10px 12px; border-radius: 8px; border-left: 3px solid #007bff; }}
        .assistant-message {{ background-color: #f9f9f9; padding: 10px 12px; border-radius: 8px; border-left: 3px solid #28a745; }}
        #controls-container {{ position: sticky; bottom: 0; z-index: 90; background-color: #f8f9fa; border-top: 1px solid #ddd; }}
        #model-selector {{ padding: 5px 10px; background-color: #f8f9fa; }}
        #input-area {{ padding: 10px 10px 15px 10px; background-color: #f8f9fa; margin-bottom: 10px; }}
        textarea {{ width: 100%; box-sizing: border-box; height: 60px; font-size: 1em; margin-bottom: 8px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; font-family: inherit; }}
        select {{ width: 100%; padding: 6px; margin: 0; font-size: 0.95em; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; position: relative; z-index: 100; }}
        select option {{ background-color: white; padding: 8px; }}
        input[type="submit"], button {{ padding: 12px; font-size: 1.1em; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; min-height: 48px; }}
        .button-row {{ display: flex; gap: 10px; margin-top: 5px; }}
        .button-row input {{ flex-grow: 1; background-color: #007bff; color: white; border-color: #007bff; }}
        .button-row input[name="regenerate"] {{ background-color: #fd7e14; border-color: #fd7e14; color: white; }}
        .button-row input:hover {{ opacity: 0.9; }}
        label {{ vertical-align: middle; }}
        input[type="radio"] {{ vertical-align: middle; margin-right: 2px; }}
        .nav-links {{ display: flex; justify-content: space-between; margin: 0; align-items: center; }}
        .nav-links a, .nav-links button {{ text-decoration: none; transition: background-color 0.2s; padding: 6px 12px; height: 32px; line-height: 20px; }}
        .nav-links a:hover, .nav-links button:hover {{ background-color: #d3d3d3; }}
        .web-search-controls {{ margin: 0 0 5px 0; font-size: 0.85em; }}
    </style>
</head>
<body>
    <div id="top-controls">
        <div class="nav-links">{nav_links_html}</div>
    </div>
    
    <div id="message-container">{history_html}</div>
    
    <div id="controls-container">
        <div id="model-selector">
            <select name="model" form="chat-form">{model_options_html}</select>
        </div>
        
        <div id="input-area">
            <form id="chat-form" method="post" style="margin: 0;">
                <textarea name="prompt" placeholder="Type your message..." autofocus></textarea>
                <div class="web-search-controls">
                    {web_search_html}
                </div>
                <div class="button-row">
                    <input type="submit" name="send" value="Send">
                    {('<input type="submit" name="regenerate" value="Regenerate">') if current_chat.get("history") else ""}
                </div>
            </form>
        </div>
    </div>
    
    <script>
        // Auto-resize textarea
        const textarea = document.querySelector('textarea');
        textarea.addEventListener('input', function() {{
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        }});
        
        // Scroll the message container to the bottom when the page loads
        window.addEventListener('load', function() {{
            const messageContainer = document.getElementById('message-container');
            messageContainer.scrollTop = messageContainer.scrollHeight;
            
            // Add padding at the bottom on mobile to prevent controls from being hidden
            if (window.innerHeight < 600) {{
                document.body.style.paddingBottom = '30px';
            }}
        }});
        
        // Focus on the textarea when the page loads
        window.addEventListener('load', function() {{
            document.querySelector('textarea').focus();
        }});
    </script>
</body>
</html>'''

@app.route('/', methods=['POST'])
def handle_chat():
    """Handle chat form submissions."""
    try:
        current_chat = get_current_chat()
        
        # Handle new chat request
        if request.form.get('new_chat'):
            session.pop('current_chat', None)
            return redirect(url_for('index'))
        
        # Handle regenerate request
        if request.form.get('regenerate'):
            # Update selected model in session (same as send request)
            model_name = request.form.get('model')
            if model_name:
                session['user_selected_model'] = model_name
                
            if current_chat.get('history'):
                # Remove last assistant message and regenerate
                history = current_chat['history']
                if history and history[-1]['role'] == 'assistant':
                    history.pop()
                    update_chat(current_chat)
                
                # Build full conversation context for regeneration
                if history:
                    model_name = session.get('user_selected_model')
                    
                    if model_name:
                        # Build full conversation context
                        messages = []
                        for msg in history:
                            messages.append({"role": msg['role'], "content": msg['content']})
                        
                        response, provider, used_model = make_api_request(model_name, messages)
                        
                        # Add response to history with metadata (no appended text)
                        provider_display = {
                            'cerebras': 'Cerebras (Direct API)',
                            'groq': 'Groq (Direct API)', 
                            'google_ai': 'Google AI Studio',
                            'g4f': 'G4F'
                        }.get(provider, provider.title())
                        
                        current_chat['history'].append({
                            'role': 'assistant',
                            'content': response,
                            'model': used_model,
                            'provider': provider_display,
                            'timestamp': datetime.now().isoformat()
                        })
                        update_chat(current_chat)
            
            return redirect(url_for('index'))
        
        # Handle send request
        if request.form.get('send'):
            prompt = request.form.get('prompt', '').strip()
            model_name = request.form.get('model')
            
            if not prompt:
                return redirect(url_for('index'))
            
            # Update selected model in session
            if model_name:
                session['user_selected_model'] = model_name
            
            # Build full conversation context for API request
            messages = []
            for msg in current_chat.get('history', []):
                messages.append({"role": msg['role'], "content": msg['content']})
            
            # Add current user message to context
            messages.append({"role": "user", "content": prompt})
            
            # Add user message to history with timestamp
            current_chat['history'].append({
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update chat name if it's the first message
            if current_chat.get('name') == 'New Chat' and len(current_chat['history']) == 1:
                # Use first 50 characters of prompt as chat name
                current_chat['name'] = prompt[:50] + ('...' if len(prompt) > 50 else '')
            
            # Handle web search
            web_search_mode = request.form.get('web_search_mode', 'off')
            search_results_str = ""
            
            if web_search_mode == 'on':
                # Always perform web search
                search_results = perform_web_search(prompt)
                if search_results:
                    search_results_str = "Web search results:\n" + "\n".join([
                        f"• {result['title']}: {result['snippet']}" for result in search_results[:3]
                    ])
                    print(f"--- Web search (ON) found {len(search_results)} results ---")
                else:
                    print("--- Web search (ON) found no results ---")
            elif web_search_mode == 'smart':
                # Smart search - only search if needed
                if should_perform_smart_search(prompt):
                    search_results = perform_web_search(prompt)
                    if search_results:
                        search_results_str = "Web search results:\n" + "\n".join([
                            f"• {result['title']}: {result['snippet']}" for result in search_results[:3]
                        ])
                        print(f"--- Web search (SMART) found {len(search_results)} results ---")
                    else:
                        print("--- Web search (SMART) found no results ---")
                else:
                    print("--- Web search (SMART) determined search not needed ---")
            
            # Modify the prompt to include search results if any
            prompt_to_use = prompt
            if search_results_str:
                prompt_to_use = f"{search_results_str}\n\nUser question: {prompt}"
                # Update the last message in the context with search results
                messages[-1]['content'] = prompt_to_use
            
            # Make API request with full conversation context
            response, provider, used_model = make_api_request(model_name, messages)
            
            # Add response to history with metadata (no appended text)
            provider_display = {
                'cerebras': 'Cerebras (Direct API)',
                'groq': 'Groq (Direct API)', 
                'google_ai': 'Google AI Studio',
                'g4f': 'G4F'
            }.get(provider, provider.title())
            
            current_chat['history'].append({
                'role': 'assistant',
                'content': response,
                'model': used_model,
                'provider': provider_display,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update chat in storage
            update_chat(current_chat)
            
            return redirect(url_for('index'))
        
        return redirect(url_for('index'))
        
    except Exception as e:
        print(f"Error in handle_chat: {e}")
        return redirect(url_for('index'))

@app.route('/saved_chats')
def saved_chats():
    """Display saved chats."""
    chats = load_chats()
    sorted_chat_items = sorted(chats.items(), 
                              key=lambda item: item[1].get('last_modified', item[1].get('created_at', '1970-01-01T00:00:00')), 
                              reverse=True)
    
    chats_html = "".join([f'''<div style="margin: 8px 0; padding: 8px; border-bottom: 1px solid #ddd; background-color: #fff; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                           <a href="/load_chat/{chat_id}" style="text-decoration: none; color: #007bff; font-weight: bold;">{html.escape(chat_data.get('name', 'Unnamed Chat'))}</a><br>
                           <small style="color:#666">
                             Last modified: {datetime.fromisoformat(chat_data.get('last_modified', chat_data.get('created_at', '1970-01-01T00:00:00'))).strftime("%b %d, %Y - %I:%M %p")}
                           </small>
                           <form method="post" action="/delete_saved_chat/{chat_id}" style="display: inline; float: right;">
                               <button type="submit" onclick="return confirm('Delete this chat?');" style="color: #dc3545; background: none; border: none; cursor: pointer; padding: 0 5px;">Delete</button>
                           </form>
                         </div>''' for chat_id, chat_data in sorted_chat_items])
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Saved Chats</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }}
        .header {{ position: fixed; top: 0; left: 0; right: 0; padding: 10px; background-color: #f8f9fa; border-bottom: 1px solid #ccc; z-index: 100; }}
        .content {{ margin-top: 60px; padding: 10px; margin-bottom: 70px; }}
        .footer {{ position: fixed; bottom: 0; left: 0; right: 0; padding: 15px; background-color: #f8f9fa; border-top: 1px solid #ccc; text-align: center; }}
        .back-button {{ padding: 12px 20px; border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333; text-decoration: none; display: inline-block; }}
        h3 {{ margin: 0; padding-bottom: 5px; color: #333; }}
        a {{ color: #007bff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h3>Saved Chats</h3>
    </div>
    <div class="content">
        {chats_html if chats else "<p>No saved chats yet. Start a conversation to create your first chat!</p>"}
    </div>
    <div class="footer">
        <a href="/" class="back-button">← Back to Chat</a>
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
        chats[chat_id]['last_modified'] = datetime.now().isoformat()
        save_chats(chats)
        
        if 'model' in chats[chat_id]:
            session['user_selected_model'] = chats[chat_id]['model']
            print(f"--- Loaded model from chat: {chats[chat_id]['model']} ---")
            
        return redirect(url_for('index'))
    else:
        return redirect(url_for('saved_chats'))

if __name__ == '__main__':
    print("--- Starting Hybrid Enhanced LLM Chat System ---")
    print("--- Updating performance data from leaderboard ---")
    update_performance_csv()
    print("--- Loading performance data from CSV ---")
    load_performance_data()
    print("--- Loading initial model cache ---")
    get_available_models()
    print("--- Flask app ready ---")
    app.run(debug=True, host='0.0.0.0', port=5002)  # Using port 5002 to avoid conflicts