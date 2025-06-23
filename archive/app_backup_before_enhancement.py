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

# Performance Data Config
PROVIDER_PERFORMANCE_URL = "https://artificialanalysis.ai/leaderboards/providers"
PERFORMANCE_CSV_PATH = os.path.join(APP_DIR, "provider_performance.csv")
print(f"--- Performance CSV path set to: {PERFORMANCE_CSV_PATH} ---")

# Free models data - extracted from provider documentation
FREE_MODELS_BY_PROVIDER = {
    "OpenRouter": [
        "Bytedance UI Tars 72B", "DeepCoder 14B Preview", "DeepHermes 3 Llama 3 8B Preview",
        "DeepSeek R1", "DeepSeek R1 Distill Llama 70B", "DeepSeek R1 Distill Qwen 14B",
        "DeepSeek R1 Distill Qwen 32B", "DeepSeek R1 Zero", "DeepSeek V3", "DeepSeek V3 0324",
        "DeepSeek V3 Base", "Dolphin 3.0 Mistral 24B", "Dolphin 3.0 R1 Mistral 24B",
        "Featherless Qwerky 72B", "Gemma 2 9B Instruct", "Gemma 3 12B Instruct",
        "Gemma 3 1B Instruct", "Gemma 3 27B Instruct", "Gemma 3 4B Instruct",
        "Llama 3.1 8B Instruct", "Llama 3.2 11B Vision Instruct", "Llama 3.2 1B Instruct", 
        "Llama 3.2 3B Instruct", "Llama 3.3 70B Instruct", "Llama 4 Maverick", "Llama 4 Scout",
        "Mistral 7B Instruct", "Mistral Nemo", "Qwen 2.5 72B Instruct", "Qwen 2.5 7B Instruct",
        "Qwen QwQ 32B", "meta-llama/llama-3.1-405b", "meta-llama/llama-3.3-8b-instruct"
    ],
    "Google": [
        "Gemini 2.5 Flash", "Gemini 2.0 Flash", "Gemini 2.0 Flash-Lite",
        "Gemini 1.5 Flash", "Gemini 1.5 Flash-8B", "Gemma 3 27B Instruct", 
        "Gemma 3 12B Instruct", "Gemma 3 4B Instruct", "Gemma 3 1B Instruct"
    ],
    "Cerebras": ["Qwen 3 32B", "Llama 4 Scout", "Llama 3.1 8B", "Llama 3.3 70B"],
    "Groq": [
        "Allam 2 7B", "DeepSeek R1 Distill Llama 70B", "Gemma 2 9B Instruct",
        "Llama 3 70B", "Llama 3 8B", "Llama 3.1 8B", "Llama 3.3 70B",
        "Llama 4 Maverick", "Llama 4 Scout", "Mistral Saba 24B", "Qwen QwQ 32B"
    ],
    "Chutes": [
        "DeepCoder 14B Preview", "DeepSeek R1", "DeepSeek R1-Zero", "DeepSeek V3",
        "DeepSeek V3 0324", "DeepSeek V3 Base", "Llama 3.1 Nemotron Ultra 253B v1",
        "Llama 4 Maverick", "Llama 4 Scout", "Mistral Small 3.1 24B Instruct",
        "Qwen 2.5 VL 32B Instruct", "qwen/qwen3-14b", "qwen/qwen3-32b", "qwen/qwen3-8b"
    ],
    "Together": ["Llama 3.2 11B Vision Instruct", "Llama 3.3 70B Instruct", "DeepSeek R1 Distil Llama 70B"],
    "Cloudflare": [
        "DeepSeek R1 Distill Qwen 32B", "Gemma 3 12B Instruct", "Llama 3 8B Instruct",
        "Llama 3.1 8B Instruct", "Llama 3.2 11B Vision Instruct", "Llama 3.2 1B Instruct",
        "Llama 3.2 3B Instruct", "Llama 3.3 70B Instruct", "Llama 4 Scout Instruct",
        "Mistral 7B Instruct", "Mistral Small 3.1 24B Instruct", "Qwen QwQ 32B"
    ]
}

def get_available_models_with_provider_counts():
    """
    Returns a tuple of (sorted_model_list, model_provider_info, provider_class_map)
    where:
    - sorted_model_list: List of tuples (model_name, provider_count, intel_index, resp_time)
    - model_provider_info: Dict mapping model_name to list of provider classes
    - provider_class_map: Dict mapping provider names to provider classes
    """
    print("--- [MODEL_DISCOVERY] Starting model discovery ---")
    
    model_list = []
    model_provider_info = {}
    provider_class_map = {}
    
    try:
        # Get all available providers from g4f with error handling
        try:
            providers = getattr(g4f, 'Provider', None)
            if providers is None:
                print("--- [MODEL_DISCOVERY] Warning: No Provider attribute found in g4f ---")
                return [], {}, {}
            
            provider_list = getattr(providers, '__providers__', [])
            if not provider_list:
                print("--- [MODEL_DISCOVERY] Warning: No providers found in g4f.Provider.__providers__ ---")
                return [], {}, {}
                
            print(f"--- [MODEL_DISCOVERY] Found {len(provider_list)} providers in g4f ---")
        except Exception as e:
            print(f"--- [MODEL_DISCOVERY] Error getting providers: {str(e)} ---")
            return [], {}, {}
        
        # Map provider classes to their names with validation
        for provider in provider_list:
            try:
                if hasattr(provider, '__name__'):
                    provider_name = provider.__name__
                    provider_class_map[provider_name] = provider
                    print(f"--- [PROVIDER] Mapped provider: {provider_name}")
            except Exception as e:
                print(f"--- [PROVIDER] Error processing provider {provider}: {str(e)}")
        
        # Get all available models with error handling
        try:
            models = g4f.models.ModelUtils.convert
            if not models:
                print("--- [MODEL_DISCOVERY] Warning: No models found in g4f.models.ModelUtils.convert ---")
                return [], {}, {}
                
            print(f"--- [MODEL_DISCOVERY] Found {len(models)} models in g4f ---")
        except Exception as e:
            print(f"--- [MODEL_DISCOVERY] Error getting models: {str(e)} ---")
            return [], {}, {}
        
        # Process each model with error handling
        processed_models = 0
        for model_name in models:
            try:
                if not model_name or not isinstance(model_name, str):
                    continue
                    
                # Get providers that support this model
                model_providers = []
                for provider in provider_list:
                    try:
                        if hasattr(provider, 'models') and model_name in provider.models:
                            model_providers.append(provider)
                    except Exception as e:
                        pass
                
                if model_providers:
                    model_provider_info[model_name] = model_providers
                    
                    # Get performance metrics with defaults
                    intel_index = 0
                    resp_time = float('inf')
                    
                    # Try to get performance data from cache if available
                    if PROVIDER_PERFORMANCE_CACHE:
                        for entry in PROVIDER_PERFORMANCE_CACHE:
                            try:
                                if entry.get('model_name_scraped', '').lower() == model_name.lower():
                                    intel_index = int(entry.get('intelligence_index', 0))
                                    resp_time = float(entry.get('response_time_s', float('inf')))
                                    break
                            except (ValueError, TypeError):
                                pass
                    
                    model_list.append((model_name, len(model_providers), intel_index, resp_time))
                    processed_models += 1
            
            except Exception as e:
                continue
        
        # Sort models by priority
        try:
            model_list.sort(key=lambda x: (
                -int(x[1]) if isinstance(x[1], (int, float)) else 0,
                -int(x[2]) if isinstance(x[2], (int, float)) else 0,
                float('inf') if not isinstance(x[3], (int, float)) else float(x[3]),
                str(x[0])
            ))
        except Exception as e:
            model_list.sort(key=lambda x: str(x[0]))
        
        print(f"--- [MODEL_DISCOVERY] Processed {processed_models} models with provider information ---")
        
        return model_list, model_provider_info, provider_class_map
        
    except Exception as e:
        print(f"--- [ERROR] Unhandled exception in get_available_models_with_provider_counts: {str(e)} ---")
        return [], {}, {}

def normalize_model_name(name):
    """Normalize model name for comparison"""
    if not isinstance(name, str):
        return ""
    # Remove version suffixes, providers, and special characters
    name = re.sub(r'\s*\(.*?\)', '', name.lower())
    name = re.sub(r'[^\w\s]', ' ', name)
    name = ' '.join(name.split())
    return name
def extract_parameter_count(name):
    """Extract parameter count from model name (e.g., '7B', '70B')"""
    if not isinstance(name, str):
        return 0
    match = re.search(r'(\d+\.?\d*)\s*[bB]', name)
    if match:
        return float(match.group(1))
    return 0

def is_free_model(model_name, provider_name):
    """Check if a model is free based on provider terms"""
    normalized_model = normalize_model_name(model_name)
    normalized_provider = provider_name.lower()
    
    for provider, models in FREE_MODELS_BY_PROVIDER.items():
        if provider.lower() in normalized_provider:
            for free_model in models:
                if normalize_model_name(free_model) == normalized_model:
                    return True
    return False

def get_dynamic_sorted_models():
    """
    Implement dynamic dropdown ordering as specified:
    1. Start with fastest free model (lowest response_time_s)
    2. Add next fastest free models only if intelligence_index is higher or equal with larger context
    3. Deduplicate same models (name + parameter count)
    4. Append remaining free models by response time
    """
    print("--- [DYNAMIC_SORT] Starting dynamic model sorting ---")
    
    # Get all models with performance data
    all_models = []
    for entry in PROVIDER_PERFORMANCE_CACHE:
        try:
            model_name = entry.get('model_name_scraped', '')
            provider_name = entry.get('provider_name_scraped', '')
            intelligence_index = int(entry.get('intelligence_index', 0))
            response_time = float(entry.get('response_time_s', float('inf')))
            context_window = parse_context_window(entry.get('context_window', '0'))
            
            if model_name and provider_name:
                all_models.append({
                    'model_name': model_name,
                    'provider_name': provider_name,
                    'intelligence_index': intelligence_index,
                    'response_time_s': response_time,
                    'context_window': context_window or 0,
                    'is_free': is_free_model(model_name, provider_name),
                    'normalized_name': normalize_model_name(model_name),
                    'parameter_count': extract_parameter_count(model_name)
                })
        except Exception as e:
            print(f"--- [DYNAMIC_SORT] Error processing entry: {e}")
            continue
    
    # Filter free models only
    free_models = [m for m in all_models if m['is_free']]
    print(f"--- [DYNAMIC_SORT] Found {len(free_models)} free models ---")
    
    if not free_models:
        print("--- [DYNAMIC_SORT] No free models found, falling back to original sort ---")
        return CACHED_AVAILABLE_MODELS_SORTED_LIST
    
    # Sort free models by response time (fastest first)
    free_models.sort(key=lambda x: x['response_time_s'])
    
    # Deduplicate by normalized name + parameter count
    seen_models = {}
    for model in free_models:
        key = (model['normalized_name'], model['parameter_count'])
        if key not in seen_models or model['intelligence_index'] > seen_models[key]['intelligence_index']:
            seen_models[key] = model
    
    deduplicated_free = list(seen_models.values())
    deduplicated_free.sort(key=lambda x: x['response_time_s'])
    
    # Primary ordering: build priority list
    priority_models = []
    if deduplicated_free:
        # Start with fastest free model
        current_model = deduplicated_free[0]
        priority_models.append(current_model)
        last_intelligence = current_model['intelligence_index']
        last_context_window = current_model['context_window']
        
        # Add subsequent models based on intelligence/context criteria
        for model in deduplicated_free[1:]:
            if (model['intelligence_index'] > last_intelligence or 
                (model['intelligence_index'] == last_intelligence and 
                 model['context_window'] > last_context_window)):
                priority_models.append(model)
                last_intelligence = model['intelligence_index']
                last_context_window = model['context_window']
    
    # Secondary ordering: remaining free models by response time
    priority_names = {m['model_name'] for m in priority_models}
    remaining_free = [m for m in deduplicated_free if m['model_name'] not in priority_names]
    remaining_free.sort(key=lambda x: x['response_time_s'])
    
    # Combine priority + remaining
    final_order = priority_models + remaining_free
    
    # Convert to the expected format for dropdown
    sorted_model_list = []
    for model in final_order:
        provider_count = 1  # At least one provider for this model
        sorted_model_list.append((
            model['model_name'],
            provider_count,
            model['intelligence_index'],
            model['response_time_s']
        ))
    
    print(f"--- [DYNAMIC_SORT] Priority models: {len(priority_models)}, Remaining: {len(remaining_free)} ---")
    print(f"--- [DYNAMIC_SORT] Final order (first 5): {[m[0] for m in sorted_model_list[:5]]} ---")
    
    return sorted_model_list

def scrape_provider_performance(url=PROVIDER_PERFORMANCE_URL):
    """Fetches and parses the provider performance table with enhanced context window support."""
    print(f"--- Scraping provider performance data from: {url} ---")
    performance_data = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find the table
        table = soup.find('table')
        if not table:
            main_content = soup.find('main')
            if main_content:
                tables = main_content.find_all('table')
                if tables: 
                    table = tables[0]
        
        if not table:
            print("--- Error: Could not find the performance table on the page. ---")
            return []
        
        tbody = table.find('tbody')
        if not tbody:
            print("--- Error: Found table but could not find tbody. ---")
            return []
        
        rows = tbody.find_all('tr')
        print(f"--- Found {len(rows)} rows in the table body. ---")
        
        # Expected column indices
        PROVIDER_COL = 0
        MODEL_COL = 1
        CONTEXT_WINDOW_COL = 2
        INTELLIGENCE_INDEX_COL = 3
        TOKENS_PER_S_COL = 5
        RESPONSE_TIME_COL = 7
        EXPECTED_COLS = max(PROVIDER_COL, MODEL_COL, INTELLIGENCE_INDEX_COL, TOKENS_PER_S_COL, RESPONSE_TIME_COL) + 1

        for row_index, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= EXPECTED_COLS:
                try:
                    # Extract provider name
                    provider_img = cols[PROVIDER_COL].find('img')
                    provider = provider_img.get('alt', '').replace(' logo', '').strip() if provider_img else cols[PROVIDER_COL].get_text(strip=True)
                    
                    # Extract model name
                    model = cols[MODEL_COL].get_text(strip=True)
                    
                    # Extract context window with enhanced parsing
                    context_window_text = cols[CONTEXT_WINDOW_COL].get_text(strip=True)
                    context_window_numeric = parse_context_window(context_window_text)
                    
                    # Extract other metrics
                    tokens_per_s_str = cols[TOKENS_PER_S_COL].get_text(strip=True)
                    response_time_str = cols[RESPONSE_TIME_COL].get_text(strip=True).lower().replace('s', '').strip()
                    intelligence_index_str = cols[INTELLIGENCE_INDEX_COL].get_text(strip=True)

                    # Convert metrics
                    try: 
                        tokens_per_s = float(tokens_per_s_str) if tokens_per_s_str.lower() != 'n/a' else 0.0
                    except ValueError: 
                        tokens_per_s = 0.0
                    
                    try: 
                        response_time_s = float(response_time_str) if response_time_str.lower() != 'n/a' else float('inf')
                    except ValueError: 
                        response_time_s = float('inf')
                    
                    try: 
                        intelligence_index = int(intelligence_index_str) if intelligence_index_str.isdigit() else 50
                    except ValueError: 
                        intelligence_index = 50

                    if provider and model:
                        # Determine if model is free
                        is_free = is_free_model(model, provider)
                        
                        performance_data.append({
                            'provider_name_scraped': provider,
                            'model_name_scraped': model,
                            'intelligence_index': intelligence_index,
                            'context_window': context_window_text,
                            'context_window_numeric': context_window_numeric,
                            'response_time_s': response_time_s,
                            'tokens_per_s': tokens_per_s,
                            'is_free_source': is_free,
                            'source_url': url,
                            'last_updated_utc': datetime.utcnow().isoformat()
                        })
                except Exception as e:
                    print(f"--- Warning: Could not parse row {row_index}: {e} ---")
            else:
                print(f"--- Warning: Skipping row {row_index} with insufficient columns ({len(cols)} found) ---")
                
    except requests.exceptions.RequestException as e:
        print(f"--- Error fetching performance data URL: {e} ---")
        return []
    except Exception as e:
        print(f"--- Error processing performance data: {e} ---")
        return []

    print(f"--- Successfully scraped {len(performance_data)} performance entries ---")
    return performance_data

def save_performance_to_csv(data, filepath=PERFORMANCE_CSV_PATH):
    """Saves the performance data list to a CSV file."""
    if not data:
        print("--- No performance data to save. ---")
        return False
    
    # Enhanced header with new fields
    header = [
        'provider_name_scraped', 'model_name_scraped', 'context_window', 'context_window_numeric',
        'intelligence_index', 'response_time_s', 'tokens_per_s', 'source_url', 
        'last_updated_utc', 'is_free_source'
    ]
    
    print(f"--- Saving {len(data)} performance entries to {filepath} ---")
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            
            for row_data in data:
                # Ensure all header fields exist in each row
                filtered_row = {}
                for key in header:
                    filtered_row[key] = row_data.get(key, '')
                writer.writerow(filtered_row)
                
        print("--- Performance data saved successfully. ---")
        return True
    except IOError as e:
        print(f"--- Error saving performance data to CSV: {e} ---")
        return False

def load_performance_from_csv(filepath=PERFORMANCE_CSV_PATH):
    """Loads performance data from a CSV file."""
    data = []
    if not os.path.exists(filepath):
        print(f"--- Performance data CSV not found at {filepath}. ---")
        return data
    
    print(f"--- Loading performance data from {filepath} ---")
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            numeric_fields = {
                'intelligence_index': (int, 50),
                'response_time_s': (float, float('inf')),
                'tokens_per_s': (float, 0.0),
                'context_window_numeric': (int, 0)
            }
            
            for row in reader:
                processed_row = {}
                try:
                    # Copy non-numeric fields
                    for key in row:
                        if key not in numeric_fields:
                            processed_row[key] = row.get(key, '')

                    # Process numeric fields
                    for field, (dtype, default_val) in numeric_fields.items():
                        value_str = row.get(field, '')
                        if value_str:
                            try:
                                if field == 'intelligence_index':
                                    processed_row[field] = int(float(value_str))
                                else:
                                    processed_row[field] = dtype(value_str)
                            except (ValueError, TypeError):
                                processed_row[field] = default_val
                        else:
                            processed_row[field] = default_val

                    data.append(processed_row)

                except Exception as e:
                    print(f"--- Warning: Skipping row due to error: {e} ---")

        print(f"--- Loaded {len(data)} performance entries from CSV. ---")
    except IOError as e:
        print(f"--- Error loading performance data from CSV: {e} ---")
    
    return data

def parse_context_window(cw_str):
    """
    Parses context window strings like "32k", "1M", "128k" into integers.
    Returns an integer or None if parsing fails.
    """
    if not isinstance(cw_str, str):
        return None
    cw_str = cw_str.lower().strip()
    if not cw_str or cw_str == 'n/a':
        return None
    try:
        if 'k' in cw_str:
            return int(float(cw_str.replace('k', '')) * 1000)
        elif 'm' in cw_str:
            return int(float(cw_str.replace('m', '')) * 1000000)
        else:
            return int(cw_str)
    except ValueError:
        print(f"--- Warning: Could not parse context window string: {cw_str} ---")
        return None

async def fetch_and_parse_free_llm_apis(timeout=20):
    """Fetch and parse free LLM APIs with enhanced error handling."""
    free_models = []
    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            # This is a placeholder - in real implementation you'd fetch from actual APIs
            # For now, return the static free models data
            for provider, models in FREE_MODELS_BY_PROVIDER.items():
                for model in models:
                    free_models.append({
                        'provider': provider,
                        'model': model,
                        'is_free': True
                    })
    except Exception as e:
        print(f"--- Error fetching free LLM APIs: {e} ---")
    
    return free_models

def load_context_window_data_from_csv():
    """Load context window data from CSV into cache."""
    global MODEL_CONTEXT_WINDOW_CACHE
    try:
        for entry in PROVIDER_PERFORMANCE_CACHE:
            model_name = entry.get('model_name_scraped', '')
            context_window = entry.get('context_window_numeric', 0)
            if model_name and context_window:
                MODEL_CONTEXT_WINDOW_CACHE[model_name] = context_window
        print(f"--- Loaded context window data for {len(MODEL_CONTEXT_WINDOW_CACHE)} models ---")
    except Exception as e:
        print(f"--- Error loading context window data: {e} ---")

def perform_web_search(query, max_results=5):
    """Perform web search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': result.get('title', ''),
                    'body': result.get('body', ''),
                    'url': result.get('href', '')
                })
            return results
    except Exception as e:
        print(f"--- Error performing web search: {e} ---")
        return []

def load_chats():
    """Load chat history from JSON file."""
    chats_file = os.path.join(APP_DIR, 'chats.json')
    try:
        if os.path.exists(chats_file):
            with open(chats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"--- Error loading chats: {e} ---")
        return {}

def save_chats(chats):
    """Save chat history to JSON file."""
    chats_file = os.path.join(APP_DIR, 'chats.json')
    try:
        # Create backup
        if os.path.exists(chats_file):
            backup_path = chats_file + '.backup'
            shutil.copy2(chats_file, backup_path)
        
        with open(chats_file, 'w', encoding='utf-8') as f:
            json.dump(chats, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"--- Error saving chats: {e} ---")
        # Restore backup if save failed
        backup_path = chats_file + '.backup'
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, chats_file)
        return False

def rate_limit(max_calls=60, period=60):
    """Rate limiting decorator."""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            now = time.time()
            # Remove old calls
            while calls and calls[0] <= now - period:
                calls.pop(0)
            
            if len(calls) >= max_calls:
                abort(429)  # Too Many Requests
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapped
    return decorator

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(APP_DIR, 'flask_session')
Session(app)

@app.route('/', methods=['GET', 'POST'])
@rate_limit(max_calls=60, period=60)
def index():
    chats = load_chats()
    
    # Use dynamically sorted model data instead of cached static list
    available_models_sorted_list = get_dynamic_sorted_models()
    if not available_models_sorted_list:
        available_models_sorted_list = CACHED_AVAILABLE_MODELS_SORTED_LIST
    
    print(f"--- [ROUTE] Number of models in dropdown: {len(available_models_sorted_list)} ---")
    
    # Get unique model names
    available_model_names = {name for name, count, intel, rt in available_models_sorted_list}
    
    # Set default model
    if available_models_sorted_list:
        default_model = available_models_sorted_list[0][0]
        print(f"--- Setting default model to: {default_model} (top of dynamic sort) ---")
    else:
        default_model = "gpt-3.5-turbo"
        print("--- Warning: No available models found. Setting default to fallback. ---")
    
    # Initialize or load current chat
    if 'current_chat' not in session or session['current_chat'] not in chats:
        session['current_chat'] = str(uuid.uuid4())
        chats[session['current_chat']] = {"history": [], "model": default_model, "name": "New Chat", "created_at": datetime.now().isoformat()}
        save_chats(chats)
    
    # Handle case where session refers to a deleted chat
    if session['current_chat'] not in chats:
        if chats:
            # Select the most recent chat if current is invalid
            latest_chat_id = sorted(chats.keys(), key=lambda k: chats[k].get('created_at', ''), reverse=True)[0]
            session['current_chat'] = latest_chat_id
        else:
            # Create a new chat if no chats exist
            session['current_chat'] = str(uuid.uuid4())
            chats[session['current_chat']] = {"history": [], "model": default_model, "name": "New Chat", "created_at": datetime.now().isoformat()}
            save_chats(chats)

    current_chat = chats[session['current_chat']]
    current_model = current_chat.get("model", default_model)
    
    # Ensure current model is valid, reset if not
    if current_model not in available_model_names and available_model_names:
        current_model = default_model
        current_chat["model"] = current_model

    if request.method == 'POST':
        # Handle New Chat action
        if 'new_chat' in request.form:
            session['current_chat'] = str(uuid.uuid4())
            chats[session['current_chat']] = {"history": [], "model": default_model, "name": "New Chat", "created_at": datetime.now().isoformat()}
            save_chats(chats)
            return redirect(url_for('index'))
        
        # Handle Delete Chat action
        if 'delete_chat' in request.form:
            chat_to_delete = session.get('current_chat')
            if chat_to_delete and chat_to_delete in chats:
                print(f"--- Deleting chat: {chat_to_delete} ---")
                del chats[chat_to_delete]
                save_chats(chats)
                session.pop('current_chat', None)
                return redirect(url_for('index'))
            else:
                print(f"--- Delete request for invalid/missing chat ID: {chat_to_delete} ---")
                return redirect(url_for('index'))

        # Process message submission or regeneration
        prompt_from_input = request.form.get('prompt', '').strip()
        selected_model_for_request = request.form.get('model', default_model)
        
        # Validate selected model
        if selected_model_for_request not in available_model_names and available_model_names:
            selected_model_for_request = default_model

        current_chat["model"] = selected_model_for_request
        
        # Simple response for demonstration
        if prompt_from_input:
            # Add user message
            current_chat["history"].append({
                "role": "user", 
                "content": prompt_from_input, 
                "timestamp": datetime.now().isoformat()
            })
            
            # Add simple AI response
            current_chat["history"].append({
                "role": "assistant", 
                "content": f"This is a placeholder response using {selected_model_for_request}. The dynamic model ordering is working!", 
                "model": selected_model_for_request, 
                "provider": "Demo Provider", 
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

        # Save chat history after processing
        save_chats(chats)
        return redirect(url_for('index'))

    # Prepare data for rendering the page (GET request or after POST redirect)
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
        # Format performance string, handle missing data
        if resp_time != float('inf') and intel_index > 0:
            perf_str = f"({intel_index}, {resp_time:.2f}s)"
        elif resp_time != float('inf'):
            perf_str = f"(Intel N/A, {resp_time:.2f}s)"
        else:
            perf_str = "(Perf N/A)"
        model_options_html += f'<option value="{model_name}" {selected_attr}>{model_name} {perf_str}</option>'

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Chat - Dynamic Model Ordering</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 0; background-color: #fff; }}
        #message-container {{ height: calc(100vh - 150px); overflow-y: auto; padding: 10px; border-bottom: 1px solid #ccc; }}
        #input-area {{ padding: 10px; background-color: #f0f0f0; border-top: 1px solid #ccc; }}
        textarea {{ width: 100%; box-sizing: border-box; height: 80px; font-size: 1em; margin-bottom: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; }}
        select {{ width: 100%; padding: 8px; margin-bottom: 10px; font-size: 1em; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
        input[type="submit"], button {{ padding: 10px; font-size: 1em; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; cursor: pointer; }}
        .button-row {{ display: flex; gap: 10px; margin-top: 10px; }}
        .button-row input {{ flex-grow: 1; background-color: #4CAF50; color: white; border-color: #4CAF50; }}
        .nav-links {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
        .nav-links button {{ padding: 8px 15px; border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333; cursor: pointer; }}
    </style>
</head>
<body>
    <div id="message-container">{history_html}</div>
    <div id="input-area">
        <form method="post" style="margin: 0;">
            <select name="model">{model_options_html}</select>
            <div class="nav-links">
                <button type="submit" name="new_chat" value="1">New Chat</button>
                <button type="submit" name="delete_chat" value="1">Delete Chat</button>
            </div>
            <textarea name="prompt" placeholder="Type your message..." autofocus></textarea>
            <div class="button-row">
                <input type="submit" name="send" value="Send">
            </div>
        </form>
    </div>
    <script>
        var messageContainer = document.getElementById('message-container');
        messageContainer.scrollTop = messageContainer.scrollHeight;
    </script>
</body>
</html>'''

def initialize_model_cache():
    """Initialize the global model cache by fetching available models and their providers."""
    global CACHED_AVAILABLE_MODELS_SORTED_LIST, CACHED_MODEL_PROVIDER_INFO, CACHED_PROVIDER_CLASS_MAP
    
    print("--- [CACHE] Starting model cache initialization ---")
    start_time = time.time()
    
    try:
        list_data, info_data, map_data = get_available_models_with_provider_counts()
        
        if not list_data or not isinstance(list_data, list):
            raise ValueError("No valid model data returned from g4f")
            
        print(f"--- [CACHE] Successfully retrieved {len(list_data)} models from g4f")
        
        CACHED_AVAILABLE_MODELS_SORTED_LIST = list_data
        CACHED_MODEL_PROVIDER_INFO = info_data if info_data and isinstance(info_data, dict) else {}
        CACHED_PROVIDER_CLASS_MAP = map_data if map_data and isinstance(map_data, dict) else {}
        
        elapsed = time.time() - start_time
        print(f"--- [CACHE] Model cache initialized in {elapsed:.2f}s with {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)} models ---")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"--- [CACHE] Error initializing model cache after {elapsed:.2f}s: {str(e)}")
        print("--- [CACHE] Falling back to default models")
        
        # Set default values
        default_models = [
            ("gpt-3.5-turbo", 1, 85, 0.5),
            ("gpt-4", 1, 95, 1.0),
            ("claude-3-opus", 1, 98, 1.2),
            ("gemini-pro", 1, 93, 0.6),
            ("llama-2-70b", 1, 92, 1.5)
        ]
        CACHED_AVAILABLE_MODELS_SORTED_LIST = default_models
        CACHED_MODEL_PROVIDER_INFO = {model[0]: [] for model in default_models}
        CACHED_PROVIDER_CLASS_MAP = {}
        
        print(f"--- [CACHE] Initialized with {len(default_models)} default models ---")

def initialize_application():
    """Initialize the application with proper error handling and logging."""
    print("\n" + "="*60)
    print("  INITIALIZING APPLICATION")
    print("="*60)
    
    # Ensure required directories exist
    required_dirs = [
        os.path.join(APP_DIR, 'chats'),
        os.path.join(APP_DIR, 'logs'),
        os.path.join(APP_DIR, 'data'),
        os.path.join(APP_DIR, 'flask_session')
    ]
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"--- [INIT] Created directory: {dir_path}")
        except Exception as e:
            print(f"--- [ERROR] Failed to create directory {dir_path}: {e}")
    
    # Initialize model cache
    print("\n--- [INIT] Initializing model cache...")
    initialize_model_cache()
    
    # Load context window data
    try:
        print("\n--- [INIT] Loading context window data...")
        load_context_window_data_from_csv()
        print(f"--- [INIT] Loaded context window data for {len(MODEL_CONTEXT_WINDOW_CACHE)} models")
    except Exception as e:
        print(f"--- [ERROR] Failed to load context window data: {e}")
    
    # Load performance data if available
    try:
        print("\n--- [INIT] Loading performance data...")
        global PROVIDER_PERFORMANCE_CACHE
        PROVIDER_PERFORMANCE_CACHE = load_performance_from_csv()
        print(f"--- [INIT] Loaded performance data for {len(PROVIDER_PERFORMANCE_CACHE)} entries")
        
        # If no performance data, try to scrape it
        if not PROVIDER_PERFORMANCE_CACHE:
            print("--- [INIT] No performance data found, attempting to scrape...")
            scraped_data = scrape_provider_performance()
            if scraped_data:
                save_performance_to_csv(scraped_data)
                PROVIDER_PERFORMANCE_CACHE = scraped_data
                print(f"--- [INIT] Scraped and saved {len(scraped_data)} performance entries")
    except Exception as e:
        print(f"--- [WARNING] Could not load performance data: {e}")
    
    print("\n" + "="*60)
    print("  APPLICATION INITIALIZATION COMPLETE")
    print("="*60 + "\n")

if __name__ == '__main__':
    try:
        # Initialize application components
        initialize_application()
        
        # Start the Flask development server
        host = '0.0.0.0'
        port = 5000
        print(f"\n{'='*60}")
        print(f"  STARTING FLASK APPLICATION")
        print(f"  - Host: {host}")
        print(f"  - Port: {port}")
        print(f"  - Models available: {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)}")
        print(f"{'='*60}\n")
        
        # Run the application
        app.run(host=host, port=port, debug=True, use_reloader=False)
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print("  FATAL ERROR DURING APPLICATION STARTUP")
        print(f"{'!'*60}")
        print(f"Error: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        print(f"\n{'!'*60}\n")