import g4f
from flask_session import Session
from g4f import ChatCompletion
# Import necessary g4f components for dynamic model loading
from g4f.models import ModelUtils, IterListProvider
# Import specific providers we might want to target
try:
    from g4f.Provider import Groq # Try importing the Groq provider
    GROQ_PROVIDER_CLASS = Groq
except ImportError:
    print("Warning: Could not import g4f.Provider.Groq. Groq-specific provider targeting disabled.")
    GROQ_PROVIDER_CLASS = None
# Import ProviderUtils to get the list of providers
# Assuming the correct path is directly under g4f.Provider
try:
    from g4f.Provider import ProviderUtils
except ImportError:
    print("Error: Could not import ProviderUtils from g4f.Provider. Provider mapping disabled.")
    ProviderUtils = None

# Try importing Cerebras Provider directly
try:
    from g4f.Provider import Cerebras
    CEREBRAS_PROVIDER_CLASS = Cerebras
except ImportError:
    print("Warning: Could not import g4f.Provider.Cerebras. Cerebras-specific provider targeting disabled.")
    CEREBRAS_PROVIDER_CLASS = None

import os
import json
import uuid
from datetime import datetime
import asyncio
import platform
import html # Imported html
from dotenv import load_dotenv # Import load_dotenv
import requests
from bs4 import BeautifulSoup
import csv # Import csv module
from duckduckgo_search import DDGS # Import for web search
import aiohttp # For Chutes AI API
import re
import sys
import shutil
import time
from functools import wraps
from flask import Flask, request, session, redirect, url_for, abort

# Initialize global variables
CACHED_AVAILABLE_MODELS_SORTED_LIST = []
CACHED_MODEL_PROVIDER_INFO = {}
CACHED_PROVIDER_CLASS_MAP = {}
PROVIDER_PERFORMANCE_CACHE = []
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CONTEXT_WINDOW_CACHE = {}

# --- Function to get available models with provider counts ---
def get_available_models_with_provider_counts():
    """
    Returns a tuple of (sorted_model_list, model_provider_info, provider_class_map)
    where:
    - sorted_model_list: List of tuples (model_name, provider_count, intel_index, resp_time)
    - model_provider_info: Dict mapping model_name to list of provider classes
    - provider_class_map: Dict mapping provider names to provider classes
    """
    print("--- [MODEL_DISCOVERY] Starting model discovery ---")
    
    # Initialize return values with empty defaults
    model_list = []
    model_provider_info = {}
    provider_class_map = {}
    
    try:
        # Get all available providers from g4f with error handling
        try:
            providers = g4f.Provider.__providers__
            if not providers:
                print("--- [MODEL_DISCOVERY] Warning: No providers found in g4f.Provider.__providers__ ---")
                return [], {}, {}
                
            print(f"--- [MODEL_DISCOVERY] Found {len(providers)} providers in g4f ---")
            if len(providers) > 0:
                print(f"--- [MODEL_DISCOVERY] First 5 providers: {[p.__name__ for p in providers[:5] if hasattr(p, '__name__')]}...")
        except Exception as e:
            print(f"--- [MODEL_DISCOVERY] Error getting providers: {str(e)} ---")
            return [], {}, {}
        
        # Map provider classes to their names with validation
        for provider in providers:
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
            if len(models) > 0:
                print(f"--- [MODEL_DISCOVERY] First 5 models: {list(models)[:5]}...")
        except Exception as e:
            print(f"--- [MODEL_DISCOVERY] Error getting models: {str(e)} ---")
            return [], {}, {}
        
        # Process each model with error handling
        processed_models = 0
        for model_name in models:
            try:
                if not model_name or not isinstance(model_name, str):
                    print(f"--- [MODEL] Skipping invalid model name: {model_name} ---")
                    continue
                    
                # Get providers that support this model
                model_providers = []
                for provider in providers:
                    try:
                        if hasattr(provider, 'models') and model_name in provider.models:
                            model_providers.append(provider)
                    except Exception as e:
                        print(f"--- [MODEL] Error checking model {model_name} in provider {getattr(provider, '__name__', str(provider))}: {str(e)}")
                
                if model_providers:
                    # Store provider info for this model
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
                            except (ValueError, TypeError) as e:
                                print(f"--- [PERF] Error parsing performance data for {model_name}: {str(e)}")
                    
                    # Add to sorted list with provider count and performance metrics
                    model_list.append((model_name, len(model_providers), intel_index, resp_time))
                    processed_models += 1
                    
                    if processed_models <= 5:  # Only log first few for brevity
                        print(f"--- [MODEL] Added: {model_name} ({len(model_providers)} providers, intel: {intel_index}, resp: {resp_time:.2f}s)")
                    elif processed_models == 6:
                        print("--- [MODEL] ... (additional models not shown) ---")
            
            except Exception as e:
                print(f"--- [MODEL] Error processing model {model_name}: {str(e)} ---")
                continue
        
        # Sort models by priority:
        # 1. Number of providers (descending)
        # 2. Intelligence index (descending)
        # 3. Response time (ascending)
        # 4. Model name (ascending)
        try:
            model_list.sort(key=lambda x: (
                -int(x[1]) if isinstance(x[1], (int, float)) else 0,  # Provider count
                -int(x[2]) if isinstance(x[2], (int, float)) else 0,  # Intelligence index
                float('inf') if not isinstance(x[3], (int, float)) else float(x[3]),  # Response time
                str(x[0])  # Model name
            ))
        except Exception as e:
            print(f"--- [SORT] Error sorting models: {str(e)} ---")
            # Fallback to simple sorting by name if complex sort fails
            model_list.sort(key=lambda x: str(x[0]))
        
        print(f"--- [MODEL_DISCOVERY] Processed {processed_models} models with provider information ---")
        if model_list:
            print(f"--- [MODEL_DISCOVERY] Top 5 models: {[m[0] for m in model_list[:5]]}...")
        
        return model_list, model_provider_info, provider_class_map
        
    except Exception as e:
        print(f"--- [ERROR] Unhandled exception in get_available_models_with_provider_counts: {str(e)} ---")
        import traceback
        print(f"--- [TRACEBACK] {traceback.format_exc()}")
        return [], {}, {}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session
app.config['SESSION_TYPE'] = 'filesystem'
# Ensure the session directory exists
SESSION_DIR = os.path.join(os.path.dirname(__file__), 'flask_session')
if not os.path.exists(SESSION_DIR):
    os.makedirs(SESSION_DIR)
app.config['SESSION_FILE_DIR'] = SESSION_DIR
Session(app)

# Load environment variables from .env file
# Load environment variables from the parent directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')  # Updated path to parent directory
load_dotenv(dotenv_path=dotenv_path)
print(f"--- Loading .env from: {dotenv_path} ---")  # Debug print

# Fix async event loop handling for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Configuration ---
# Path to store chat histories
CHAT_STORAGE = os.path.join(os.path.dirname(__file__), "chats.json") # NEW PATH - Save inside /app
# API Keys loaded from .env
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"--- GOOGLE_API_KEY Loaded: {'Exists' if GOOGLE_API_KEY else 'Not Found'} ---") # API Key Load Check
# Add checks for other keys
print(f"--- CEREBRAS_API_KEY Loaded: {'Exists' if CEREBRAS_API_KEY else 'Not Found'} ---")
print(f"--- GROQ_API_KEY Loaded: {'Exists' if GROQ_API_KEY else 'Not Found'} ---")

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
print(f"--- SERPAPI_API_KEY Loaded: {'Exists' if SERPAPI_API_KEY else 'Not Found'} ---")

# Map user-facing model names to provider-specific names if they differ
# Based on Groq error message and user list
GROQ_MODEL_NAME_MAP = {
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "llama-3.3-70b": "llama-3.3-70b-versatile", # Corrected mapping
    "gemma-2-9b-it": "gemma2-9b-it", # g4f seems to use gemma2
    "llama-3-8b": "llama3-8b-8192",
    "llama-3-70b": "llama3-70b-8192",
    # Add other mappings as needed based on Groq's valid list vs user selection
}
# Models we KNOW should target specific providers if selected (using user-facing names)
# Ensure keys from the map are included if they are also display names
# GROQ_TARGET_MODELS is now replaced by dynamic GROQ_MODELS_CACHE (Removed line)
CEREBRAS_TARGET_MODELS = {"llama-3.1-8b", "llama-3.3-70b", "llama-4-scout"} # Keep Cerebras hardcoded for now
# Update Google models based on user feedback and potential scraped names
GOOGLE_TARGET_MODELS = {
    "Gemini 2.5 Flash (AI_Studio)", # Match a name from CSV for context window
    "Gemini 2.0 Flash (AI Studio)", # Match a name from CSV
    "Gemini 2.0 Flash Lite (Preview) (AI Studio)" # Match a name from CSV
} # Corrected Google AI Studio models

# --- Add Display Name Mapping (Moved Higher) ---
# Ensure these display names are also updated if the internal names changed
MODEL_DISPLAY_NAME_MAP = {
    "llama-4-maverick": "Llama 4 Maverick",
    "llama4maverick": "Llama 4 Maverick",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "qwen-qwq-32b": "QwQ-32B", # Added display name for QwQ
    # Add mappings for Gemini variants to ensure consistent display
    "Gemini 2.5 Flash": "Gemini 2.5 Flash",
    "Gemini 2.0 Flash": "Gemini 2.0 Flash",
    "Gemini 2.0 Flash Lite": "Gemini 2.0 Flash Lite"
}
# --- End Display Name Mapping ---

# Provider names as they might appear in the scraped data (case-insensitive check recommended)
SCRAPED_PROVIDER_NAME_CEREBRAS = "Cerebras"
SCRAPED_PROVIDER_NAME_GROQ = "Groq"


# Chutes AI Config
CHUTES_API_URL = "https://llm.chutes.ai/v1"
CHUTES_MODELS_CACHE = [] # Cache for models fetched from Chutes API

# Groq API Config
GROQ_API_URL = "https://api.groq.com/openai/v1"
GROQ_MODELS_CACHE = [] # Cache for models fetched from Groq API
# Performance Data Config
PROVIDER_PERFORMANCE_URL = "https://artificialanalysis.ai/leaderboards/providers"
# --- Corrected Performance Data Path ---
# Get the directory where app.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Define CSV path relative to the app directory
PERFORMANCE_CSV_PATH = os.path.join(APP_DIR, "provider_performance.csv")
print(f"--- Performance CSV path set to: {PERFORMANCE_CSV_PATH} ---") # Add print for verification
# --- End Corrected Path ---
# --- End Configuration ---

DEFAULT_MAX_OUTPUT_TOKENS = 8000 # Default for max output tokens

# --- Global Cache for Available Models ---
CACHED_AVAILABLE_MODELS_SORTED_LIST = []
CACHED_MODEL_PROVIDER_INFO = {}
CACHED_PROVIDER_CLASS_MAP = {}
# --- End Global Cache ---

# --- Performance Data Handling ---

def scrape_provider_performance(url=PROVIDER_PERFORMANCE_URL):
    """Fetches and parses the provider performance table."""
    print(f"--- Scraping provider performance data from: {url} ---")
    performance_data = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table')
        if not table:
             main_content = soup.find('main')
             if main_content:
                 tables = main_content.find_all('table')
                 if tables: table = tables[0]
        if not table:
            print("--- Error: Could not find the performance table on the page. ---")
            return []
        tbody = table.find('tbody')
        if not tbody:
            print("--- Error: Found table but could not find tbody. ---")
            return []
        rows = tbody.find_all('tr')
        print(f"--- Found {len(rows)} rows in the table body. ---")
        # Expected column indices (adjust if the table layout changes)
        PROVIDER_COL = 0
        MODEL_COL = 1
        CONTEXT_WINDOW_COL = 2
        INTELLIGENCE_INDEX_COL = 3
        TOKENS_PER_S_COL = 5      # Assuming 6th column
        RESPONSE_TIME_COL = 7   # Assuming 8th column
        EXPECTED_COLS = max(PROVIDER_COL, MODEL_COL, INTELLIGENCE_INDEX_COL, TOKENS_PER_S_COL, RESPONSE_TIME_COL) + 1

        for row_index, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= EXPECTED_COLS:
                try:
                    provider_img = cols[PROVIDER_COL].find('img')
                    provider = provider_img['alt'].replace(' logo', '').strip() if provider_img and provider_img.has_attr('alt') else cols[PROVIDER_COL].get_text(strip=True)
                    model = cols[MODEL_COL].get_text(strip=True)
                    tokens_per_s_str = cols[TOKENS_PER_S_COL].get_text(strip=True)
                    response_time_str = cols[RESPONSE_TIME_COL].get_text(strip=True).lower().replace('s', '').strip()
                    intelligence_index_str = cols[INTELLIGENCE_INDEX_COL].get_text(strip=True)

                    # Convert tokens per second
                    try: tokens_per_s = float(tokens_per_s_str) if tokens_per_s_str.lower() != 'n/a' else 0.0
                    except ValueError: tokens_per_s = 0.0
                    # Convert response time
                    try: response_time_s = float(response_time_str) if response_time_str.lower() != 'n/a' else float('inf')
                    except ValueError: response_time_s = float('inf')
                    # Convert intelligence index
                    try: intelligence_index = int(intelligence_index_str) if intelligence_index_str.isdigit() else 0 # Default to 0 if not a digit
                    except ValueError: intelligence_index = 0 # Default to 0 on conversion error

                    if provider and model:
                        performance_data.append({
                            'provider_name_scraped': provider,
                            'model_name_scraped': model,
                            'intelligence_index': intelligence_index if intelligence_index > 0 else 50,
                            'context_window': cols[CONTEXT_WINDOW_COL].get_text(strip=True).replace('k', '000').replace('m', '000000'),
                            'response_time_s': response_time_s,
                            'tokens_per_s': tokens_per_s
                        })
                except IndexError as e:
                     print(f"--- Warning: Skipping row {row_index} due to missing columns (IndexError): {row}. Error: {e} ---")
                except Exception as e:
                    print(f"--- Warning: Could not parse row {row_index} content: {row}. Error: {e} ---")
            else:
                 print(f"--- Warning: Skipping row {row_index} with insufficient columns ({len(cols)} found): {row} ---")
    except requests.exceptions.RequestException as e:
        print(f"--- Error fetching performance data URL: {e} ---")
        return []
    except Exception as e:
        print(f"--- Error processing performance data: {e} ---")
        return []

    if not performance_data:
         print("--- Warning: Scraping finished but no performance data was extracted. ---")
    else:
        print(f"--- Successfully scraped {len(performance_data)} performance entries (including intelligence index). ---")
    return performance_data

def save_performance_to_csv(data, filepath=PERFORMANCE_CSV_PATH):
    """Saves the performance data list to a CSV file."""
    if not data:
        print("--- No performance data to save. --- ")
        return False
    # Ensure all expected keys are present in the first row for the header
    # Add default values if keys are missing in the first row
    default_entry = {'provider_name_scraped': '', 'model_name_scraped': '', 'intelligence_index': 0, 'response_time_s': float('inf'), 'tokens_per_s': 0.0}
    header = list(default_entry.keys()) # Use defined order
    print(f"--- Saving {len(data)} performance entries to {filepath} with header: {header} ---")
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            # Write data, ensuring all header fields exist in each row
            for row_data in data:
                # Create a new dict with default values, update with actual row data
                row_to_write = default_entry.copy()
                row_to_write.update(row_data)
                # Ensure only keys defined in the header are written
                filtered_row = {key: row_to_write.get(key, default_entry[key]) for key in header}
                writer.writerow(filtered_row)
        print("--- Performance data saved successfully. ---")
        return True
    except IOError as e:
        print(f"--- Error saving performance data to CSV: {e} ---")
        return False
    except Exception as e:
        print(f"--- Unexpected error saving performance data: {e} ---")
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
            # Define expected numeric fields and their defaults/types
            numeric_fields = {
                'intelligence_index': (int, 0),
                'response_time_s': (float, float('inf')),
                'tokens_per_s': (float, 0.0)
            }
            for row in reader:
                processed_row = {} # Store processed data for this row
                try:
                    # Copy non-numeric fields first
                    for key in row:
                        if key not in numeric_fields:
                            processed_row[key] = row.get(key, '') # Default to empty string if somehow missing

                    # Process numeric fields with type conversion and error handling
                    for field, (dtype, default_val) in numeric_fields.items():
                        value_str = row.get(field) # Get the string value from CSV
                        if value_str:
                            try:
                                if field == 'intelligence_index':
                                    # Attempt to convert to float first, then to int
                                    processed_row[field] = int(float(value_str))
                                else:
                                    # Original behavior for other numeric fields
                                    processed_row[field] = dtype(value_str)
                            except (ValueError, TypeError):
                                # Custom warning for intelligence_index, generic for others
                                if field == 'intelligence_index':
                                    print(f"--- Warning: Could not convert '{field}' value '{value_str}' to int. Using default {default_val}. Row: {row} ---")
                                else:
                                    print(f"--- Warning: Could not convert '{field}' value '{value_str}' to {dtype.__name__}. Using default {default_val}. Row: {row} ---")
                                processed_row[field] = default_val
                        else:
                            # Handle missing column or empty value
                            # Temporarily comment out this specific warning to reduce log noise
                            # print(f"--- Warning: Missing or empty value for '{field}'. Using default {default_val}. Row: {row} ---")
                            processed_row[field] = default_val

                    data.append(processed_row)

                except Exception as e:
                    # Catch unexpected errors during row processing
                    print(f"--- Warning: Skipping row due to unexpected error: {row}. Error: {e} ---")

        print(f"--- Loaded {len(data)} performance entries from CSV. ---")
    except IOError as e:
        print(f"--- Error loading performance data from CSV: {e} ---")
    except Exception as e:
        print(f"--- Unexpected error loading performance data: {e} ---")
    return data

# --- End Performance Data Handling ---

# --- Context Window Parsing Utility ---
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
            return int(cw_str) # Assume it's already an integer
    except ValueError:
        print(f"--- Warning: Could not parse context window string: {cw_str} ---")
        return None
# --- End Context Window Parsing Utility ---

# --- Free LLM API Provider Info ---
FREE_LLM_API_README_URL = "https://github.com/xtekky/gpt4free/blob/main/README.md"
CACHED_FREE_API_PROVIDERS = {} # Cache for {provider_name: {model_name: {details}}}

async def fetch_and_parse_free_llm_apis(url=FREE_LLM_API_README_URL):
