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
    """
    Fetches and parses the Markdown file for free LLM API providers.
    Populates CACHED_FREE_API_PROVIDERS.
    This is a placeholder and will need robust Markdown parsing.
    """
    global CACHED_FREE_API_PROVIDERS
    print(f"--- Fetching free LLM API info from: {url} ---")
    try:
        # Using aiohttp for async request, similar to other fetches in the app
        async with aiohttp.ClientSession() as http_session:
            async with http_session.get(url, timeout=20) as response:
                response.raise_for_status()
                markdown_content = await response.text()
        
        print(f"--- Successfully fetched README.md content ({len(markdown_content)} chars). Parsing content... ---")
        parsed_data = {}
        current_provider_name = None
        
        # Regex to identify provider headings (##, ###, ####)
        provider_heading_re = re.compile(r"^#{2,4}\s*(?:\[([^\]]+)\]\(.*?\)|([^#\n(]+?))\s*(?:\(.*?\))?\s*$", re.IGNORECASE)
        
        # Regex for list items (potential models)
        list_item_re = re.compile(r"^\s*-\s+(.+?)(?:\s*[-–—:]\s*(.+))?$", re.IGNORECASE)
        
        # Regex to extract model name(s) from an item, possibly from a link: [Model](link) or just Model Name
        # It tries to capture full model names including versions or special characters.
        model_name_re = re.compile(r"(?:\[([^\]]+?)\]\(.+?\)|([\w\./\s()'-]+(?:[\w\s()'-]+)?))", re.IGNORECASE)

        # Keywords for payment requirement
        no_payment_keywords = [
            "no payment method", "no credit card", "no cc required", "without payment",
            "doesn't require payment", "no card needed", "payment not required"
        ]
        requires_payment_keywords = [
            "payment method required", "credit card required", "cc required",
            "needs payment", "requires payment", "card required"
        ]

        lines = markdown_content.splitlines()
        
        for line_number, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped: # Skip empty lines
                continue

            provider_match = provider_heading_re.match(line_stripped)
            if provider_match:
                name_from_link = provider_match.group(1)
                name_plain = provider_match.group(2)
                current_provider_name = (name_from_link or name_plain).strip()
                if current_provider_name and current_provider_name not in parsed_data:
                    parsed_data[current_provider_name] = {}
                # print(f"DEBUG: New provider section: {current_provider_name}")
                continue

            if current_provider_name:
                list_item_match = list_item_re.match(line_stripped)
                if list_item_match:
                    item_text = list_item_match.group(1).strip()
                    item_description = (list_item_match.group(2) or "").strip()
                    full_text_for_payment_check = (item_text + " " + item_description).lower()

                    extracted_models = []
                    model_matches_from_re = model_name_re.findall(item_text)
                    if model_matches_from_re:
                        for m_link, m_plain in model_matches_from_re:
                            model_name = (m_link or m_plain).strip()
                            if model_name and len(model_name) > 1 and not model_name.lower().startswith("http") and "free tier" not in model_name.lower() and "api" not in model_name.lower():
                                extracted_models.append(model_name)
                    
                    # If no models extracted via regex, but line seems relevant (e.g. contains 'model', 'API')
                    # this part can be risky and might need more specific heuristics if used.
                    # For now, rely on model_name_re.

                    if not extracted_models:
                        # Could be a general note under the provider, not a model listing.
                        # print(f"DEBUG: No models extracted by regex from: '{item_text}' for provider {current_provider_name}")
                        continue # Skip if no clear model name is found by regex

                    requires_payment = None
                    for kw in requires_payment_keywords:
                        if kw in full_text_for_payment_check:
                            requires_payment = True
                            break
                    if requires_payment is None:
                        for kw in no_payment_keywords:
                            if kw in full_text_for_payment_check:
                                requires_payment = False
                                break
                    
                    if requires_payment is None:
                        requires_payment = True # Default to needing payment if not explicitly stated otherwise

                    for model_name_to_add in extracted_models:
                        # Normalize model name slightly (e.g. remove trailing dots if any)
                        model_name_to_add = model_name_to_add.rstrip('.').strip()
                        if not model_name_to_add: continue

                        if model_name_to_add not in parsed_data[current_provider_name]:
                            parsed_data[current_provider_name][model_name_to_add] = {
                                "requires_payment_method": requires_payment,
                                "notes": item_description,
                                "original_line": line_stripped
                            }
                            # print(f"DEBUG: Added model: '{model_name_to_add}' for {current_provider_name}, Payment: {requires_payment}, Notes: '{item_description}'")
                        # else:
                            # print(f"DEBUG: Model '{model_name_to_add}' already exists for {current_provider_name}")
        
        CACHED_FREE_API_PROVIDERS = parsed_data

    except aiohttp.ClientError as e:
        print(f"--- Error fetching free LLM API README (ClientError): {e} ---")
        CACHED_FREE_API_PROVIDERS = {}
    except asyncio.TimeoutError:
        print(f"--- Error fetching free LLM API README: Request timed out. ---")
        CACHED_FREE_API_PROVIDERS = {}
    except Exception as e:
        print(f"--- Unexpected error fetching or parsing free LLM API README: {e} ---")
        CACHED_FREE_API_PROVIDERS = {}

    if not CACHED_FREE_API_PROVIDERS:
        print("--- No free LLM API provider data parsed or an error occurred. ---")
    else:
        print(f"--- Successfully parsed {len(CACHED_FREE_API_PROVIDERS)} free API providers. ---")

# --- End Free LLM API Provider Info ---

# --- Context Window Data Handling from LLM Leaderboard CSV ---
CONTEXT_WINDOW_CSV_PATH = os.path.join(APP_DIR, "data", "context_windows.csv")
MODEL_CONTEXT_WINDOW_CACHE = {} # Cache for {(provider_name.lower(), model_name.lower()): context_window_int}

def load_context_window_data_from_csv(filepath=CONTEXT_WINDOW_CSV_PATH):
    """
    Loads context window data from the specified CSV file.
    Uses the parse_context_window utility.
    Populates MODEL_CONTEXT_WINDOW_CACHE.
    """
    global MODEL_CONTEXT_WINDOW_CACHE
    MODEL_CONTEXT_WINDOW_CACHE = {} # Reset cache

    if not os.path.exists(filepath):
        print(f"--- Context window CSV not found at {filepath}. Skipping load. ---")
        return
    
    print(f"--- Loading context window data from {filepath} ---")
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Expected column names (case-sensitive from the file)
            provider_col = 'API Provider'
            model_col = 'Model'
            context_col = 'ContextWindow' # Corrected based on CSV header 'ContextWindow'

            if not all(col in reader.fieldnames for col in [provider_col, model_col, context_col]):
                print(f"--- Error: CSV missing one or more required columns: '{provider_col}', '{model_col}', '{context_col}'. Found: {reader.fieldnames} ---")
                return

            for row_num, row in enumerate(reader):
                try:
                    provider_name = row.get(provider_col, '').strip()
                    model_name_csv = row.get(model_col, '').strip() # Raw model name from CSV
                    context_str = row.get(context_col, '').strip()

                    if not provider_name or not model_name_csv:
                        # print(f"--- Warning: Skipping row {row_num+2} in context CSV due to missing provider or model name. ---")
                        continue

                    context_int = parse_context_window(context_str)
                    # Removed [LOAD_CTX_DEBUG] print for CSV Row

                    if context_int is not None:
                        provider_key = provider_name.lower()
                        # Normalize provider key further for Google AI Studio variants
                        provider_key = provider_key.replace('(ai_studio)', '(ai studio)')
                        
                        model_key_normalized = model_name_csv.lower().replace(' ', '-')
                        MODEL_CONTEXT_WINDOW_CACHE[(provider_key, model_key_normalized)] = context_int
                        # Removed [LOAD_CTX_DEBUG] print for Stored key1
                        
                        model_key_simple_lower = model_name_csv.lower()
                        if model_key_simple_lower != model_key_normalized:
                            MODEL_CONTEXT_WINDOW_CACHE[(provider_key, model_key_simple_lower)] = context_int
                            # Removed [LOAD_CTX_DEBUG] print for Stored key2
                    # else:
                        # Removed [LOAD_CTX_DEBUG] print for CtxInt is None
                        # print(f"--- Info: Could not parse context window '{context_str}' for {provider_name} - {model_name_csv} in context CSV. ---")

                except Exception as e:
                    print(f"--- Warning: Error processing row {row_num+2} in context CSV: {row}. Error: {e} ---")
        
        print(f"--- Loaded {len(MODEL_CONTEXT_WINDOW_CACHE)} context window entries from CSV. ---")

    except IOError as e:
        print(f"--- Error loading context window data from CSV: {e} ---")
    except Exception as e:
        print(f"--- Unexpected error loading context window data: {e} ---")

# --- End Context Window Data Handling ---

# --- Web Search Functionality ---
def perform_web_search(query, num_results=3):
    """Performs a web search using SerpAPI, Gemini grounding, then DuckDuckGo as fallback."""
    print(f"--- Performing web search for: {query} ---")
    results_str = "Web search results:\n"
    # Try SerpAPI
    if SERPAPI_API_KEY:
        try:
            params = {'engine': 'google', 'q': query, 'api_key': SERPAPI_API_KEY, 'num': num_results}
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
            data = resp.json()
            organic = data.get('organic_results', [])[:num_results]
            if organic:
                for idx, item in enumerate(organic, 1):
                    title = item.get('title', 'No Title')
                    snippet = item.get('snippet', 'No Snippet')
                    link = item.get('link', 'N/A')
                    results_str += f"{idx}. {title}\n   Snippet: {snippet}\n   Source: {link}\n"
                print("--- SerpAPI search complete. ---")
                return results_str
            else:
                print("--- SerpAPI returned no results, falling back. ---")
        except Exception as e:
            print(f"--- SerpAPI error: {e} ---")
    # Try Gemini grounding via Google Generative AI
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            gen_config = genai.types.GenerationConfig(max_output_tokens=512)
            messages = [{'role': 'user', 'content': f'Search the web and provide the top {num_results} concise results for: {query}'}]
            for model_id in ['gemini-2.5-pro-grounding', 'gemini-2.5-flash-grounding']:
                try:
                    model = genai.GenerativeModel(model_id)
                    response = model.generate_content(messages, generation_config=gen_config)
                    if response and hasattr(response, 'text') and response.text:
                        print(f"--- Gemini grounding ({model_id}) complete. ---")
                        return 'Web search results (Gemini grounding):\n' + response.text.strip()
                    else:
                        print(f"--- Gemini grounding with {model_id} returned no text, trying next... ---")
                except Exception as e2:
                    print(f"--- Gemini grounding error with {model_id}: {e2} ---")
        except Exception as e:
            print(f"--- Google Generative AI setup error: {e} ---")
    # DuckDuckGo fallback
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=num_results))
            if not search_results:
                results_str += "No results found.\n"
            else:
                for i, result in enumerate(search_results, 1):
                    title = result.get('title', 'No Title')
                    body = result.get('body', 'No Snippet')
                    href = result.get('href', 'N/A')
                    results_str += f"{i}. {title}\n   Snippet: {body}\n   Source: {href}\n"
    except Exception as e:
        print(f"--- DuckDuckGo search error: {e} ---")
        results_str += "Search failed.\n"
    print("--- Web search complete. ---")
    return results_str
# --- End Web Search ---


# Function to load chat history
def load_chats():
    """Loads chat histories from the JSON storage file."""
    if os.path.exists(CHAT_STORAGE):
        try:
            with open(CHAT_STORAGE, "r", encoding='utf-8') as f:
                content = f.read()
                if not content: 
                    return {}
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in {CHAT_STORAGE}. Creating backup and returning empty dict.")
                    # Create backup of corrupted file
                    backup_path = f"{CHAT_STORAGE}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with open(backup_path, "w", encoding='utf-8') as backup:
                        backup.write(content)
                    return {}
        except Exception as e:
            print(f"Error loading chats from {CHAT_STORAGE}: {e}")
            return {}
    return {}

def save_chats(chats):
    """Saves the chat histories to the JSON storage file."""
    try:
        # Create backup of existing file if it exists
        if os.path.exists(CHAT_STORAGE):
            backup_path = f"{CHAT_STORAGE}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(CHAT_STORAGE, backup_path)
        
        # Save new content
        with open(CHAT_STORAGE, "w", encoding='utf-8') as f:
            json.dump(chats, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        print(f"--- Chat save successful for {len(chats)} chats. ---")
    except Exception as e:
        print(f"Error saving chats to {CHAT_STORAGE}: {e}")
        # Try to restore from backup if save failed
        if 'backup_path' in locals():
            try:
                shutil.copy2(backup_path, CHAT_STORAGE)
                print(f"Restored from backup: {backup_path}")
            except Exception as restore_error:
                print(f"Failed to restore from backup: {restore_error}")

# Add rate limiting for chat operations
from functools import wraps
from flask import request, abort
import time

RATE_LIMIT_WINDOW = 60  # 1 minute window
RATE_LIMIT_MAX_REQUESTS = 30  # Maximum requests per window

def rate_limit():
    """Rate limiting decorator for chat operations."""
    def decorator(f):
        request_history = {}
        
        @wraps(f)
        def wrapped(*args, **kwargs):
            now = time.time()
            client_ip = request.remote_addr
            
            # Clean up old entries
            request_history[client_ip] = [t for t in request_history.get(client_ip, []) 
                                        if now - t < RATE_LIMIT_WINDOW]
            
            # Check rate limit
            if len(request_history.get(client_ip, [])) >= RATE_LIMIT_MAX_REQUESTS:
                abort(429)  # Too Many Requests
            
            # Add current request
            request_history.setdefault(client_ip, []).append(now)
            
            return f(*args, **kwargs)
        return wrapped
    return decorator

# Apply rate limiting to chat routes
@app.route('/', methods=['GET', 'POST'])
@rate_limit()
def index():
    chats = load_chats()
    # Use cached model data
    available_models_sorted_list = CACHED_AVAILABLE_MODELS_SORTED_LIST
    print(f"--- [ROUTE] Number of models in dropdown: {len(available_models_sorted_list)} ---")
    print(f"--- [ROUTE] First few models: {[m[0] for m in available_models_sorted_list[:5]]} ---")
    
    # model_provider_info and provider_class_map will be accessed via their global cached versions
    # directly in the POST handler where needed.

    # Get unique model names
    available_model_names = {name for name, count, intel, rt in available_models_sorted_list}
    print(f"--- [ROUTE] Number of unique model names: {len(available_model_names)} ---")
    print(f"--- [ROUTE] First few unique model names: {list(available_model_names)[:5]} ---")

    # --- Set Default Model ---
    # Set default to the first model in the custom sorted list
    if available_models_sorted_list:
        default_model = available_models_sorted_list[0][0] # Index 0 is model name
        print(f"--- Setting default model to: {default_model} (top of custom sort) ---")
    else:
        default_model = "gpt-3.5-turbo" # Absolute fallback
        print("--- Warning: No available models found. Setting default to fallback. ---")
    # --- End Default Model ---

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
    # Add debug logs for loaded data
    print(f"--- Index GET: Loaded {len(chats)} chats. Current chat ID: {session['current_chat']} ---")
    print(f"--- Index GET: Current chat history length: {len(current_chat.get('history', []))} ---")
    print(f"--- Index GET: Current chat model: {current_model} ---")
    # Ensure current model is valid, reset if not
    if current_model not in available_model_names and available_model_names:
        current_model = default_model
        current_chat["model"] = current_model

    # Define web_search_mode with a default for GET requests
    web_search_mode = 'smart' # Default value

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
                session.pop('current_chat', None) # Remove from session
                return redirect(url_for('index')) # Redirect to potentially new default chat
            else:
                 print(f"--- Delete request for invalid/missing chat ID: {chat_to_delete} ---")
                 return redirect(url_for('index'))

        # Process message submission or regeneration
        prompt_from_input = request.form.get('prompt', '').strip()
        selected_model_for_request = request.form.get('model', default_model)
        # Validate selected model
        if selected_model_for_request not in available_model_names and available_model_names:
            selected_model_for_request = default_model

        # Get web_search_mode from form for POST requests, overwriting the default
        web_search_mode = request.form.get('web_search_mode', 'smart') # Default to smart if not provided
        search_results_str = "" # Initialize search results

        prompt_to_use = prompt_from_input
        is_regeneration = 'regenerate' in request.form and current_chat["history"]
        provider_used_str = "Unknown"
        response_content = None
        final_error_message = None
        current_chat["model"] = selected_model_for_request # Update chat model immediately

        # Prepare message history for API call
        temp_history_for_api = list(current_chat["history"])
        remove_last_ai_msg_from_actual_history = False

        if is_regeneration:
            # Find the last user prompt to regenerate from
            last_user_prompt = next((msg["content"] for msg in reversed(temp_history_for_api) if msg["role"] == "user"), None)
            if last_user_prompt:
                prompt_to_use = last_user_prompt
                # Remove the last assistant message if it exists
                if temp_history_for_api and temp_history_for_api[-1]["role"] == "assistant":
                    temp_history_for_api.pop()
                    remove_last_ai_msg_from_actual_history = True
            else:
                prompt_to_use = "" # Cannot regenerate if no prior user prompt
        elif prompt_to_use:
            # Add current user prompt to temporary history for the API call
            temp_history_for_api.append({"role": "user", "content": prompt_to_use, "timestamp": datetime.now().isoformat()})
        else:
            prompt_to_use = "" # Ensure prompt_to_use is empty if input was empty

        # Only proceed if there's a valid prompt (either new or from regeneration)
        if prompt_to_use:
            # --- Web Search Logic ---
            if web_search_mode == 'on':
                app.logger.info(f"--- Web search explicitly enabled (mode='on') for: {prompt_to_use[:50]}... ---")
                search_results_str = perform_web_search(prompt_to_use)
            elif web_search_mode == 'smart':
                # Enhanced keyword check for smart search trigger
                smart_search_keywords = [
                    # Time-based keywords
                    "latest", "recent", "today", "current", "now", "present", "currently",
                    # News-style keywords
                    "current news", "what happened", "breaking news", "update",
                    # Question patterns that likely need current info
                    "who is", "what is", "where is", "when did", "how many",
                    # Position/status keywords
                    "current president", "current leader", "current status", "current state",
                    # Event-related keywords
                    "recent event", "recent development", "recent change", "recent update",
                    # Version/iteration keywords
                    "latest version", "current version", "newest", "most recent"
                ]
                
                # Check for keywords in prompt
                prompt_lower = prompt_to_use.lower()
                should_search = any(keyword in prompt_lower for keyword in smart_search_keywords)
                
                # Additional pattern matching for current information needs
                if not should_search:
                    # Check for questions about current positions/status
                    position_patterns = [
                        r"who is the current .+",
                        r"what is the current .+",
                        r"where is the current .+",
                        r"when did the current .+",
                        r"how many current .+"
                    ]
                    import re
                    should_search = any(re.search(pattern, prompt_lower) for pattern in position_patterns)
                
                if should_search:
                    app.logger.info(f"--- Smart search triggered for: {prompt_to_use[:50]}... ---")
                    search_results_str = perform_web_search(prompt_to_use)
                else:
                    app.logger.info(f"--- Smart search not triggered for prompt. ---")
            # --- End Web Search Logic ---

            # Prepare messages for API call (convert full history to API format)
            api_messages = [{"role": msg["role"], "content": msg["content"]} for msg in temp_history_for_api]

            # Prepend search results if they exist
            if search_results_str:
                print("--- Prepending web search results to API messages ---")
                # Modify the last user message content
                if api_messages and api_messages[-1]["role"] == "user":
                     original_prompt_content = api_messages[-1]["content"]
                     api_messages[-1]["content"] = f"Web Search Results:\n{search_results_str}\n\nOriginal User Prompt:\n{original_prompt_content}"
                else:
                     # Fallback: Add as a system message if history is weird
                     api_messages.insert(0, {"role": "system", "content": f"Context from web search:\n{search_results_str}"}) # Insert at beginning

            # Final check for valid messages before calling API
            if not api_messages or not any(msg['role'] == 'user' for msg in api_messages):
                 print("--- Error: No user messages to send to API after potential search modification ---")
                 final_error_message = "Error: Cannot send empty message history."
                 provider_used_str = "Internal Error"
            else:
                # --- Provider Selection Logic (Refactored based on Plan v2) ---
                provider_used_str = "None Attempted"
                response_content = None
                # Seed attempted_direct_providers to skip the previous provider on regeneration
                if is_regeneration:
                    attempted_direct_providers = set()
                    last_history = current_chat.get("history", [])
                    if last_history and last_history[-1].get("role") == "assistant":
                        last_provider = last_history[-1].get("provider", "")
                        lp = last_provider.lower()
                        if "groq" in lp:
                            attempted_direct_providers.add("groq")
                        elif "cerebras" in lp:
                            attempted_direct_providers.add("cerebras")
                        elif "google" in lp:
                            attempted_direct_providers.add("google")
                        elif "chutes" in lp:
                            attempted_direct_providers.add("chutes")
                else:
                    attempted_direct_providers = set()
                provider_errors = {} # Dictionary to store errors per provider
                max_retries = 2  # Maximum number of retries with web search
                retry_count = 0
                should_retry = True
                max_tokens_for_request = DEFAULT_MAX_OUTPUT_TOKENS # Use the defined constant

                while should_retry and retry_count < max_retries:
                    # Helper function to get performance score (lower is better, e.g., response time)
                    def get_scraped_performance_metric(provider_identifier, model_name):
                        # Handles both provider class objects and string identifiers
                        try:
                            provider_name_lower = ""
                            provider_name_for_print = "" # For logging
                            if isinstance(provider_identifier, str):
                                # Map string identifiers to searchable names (case-insensitive)
                                provider_name_for_print = provider_identifier
                                id_lower = provider_identifier.lower()
                                if "groq" in id_lower:
                                    provider_name_lower = "groq"
                                elif "cerebras" in id_lower:
                                    provider_name_lower = "cerebras"
                                elif "chutes" in id_lower:
                                    provider_name_lower = "chutesai" # Match scraped name if possible
                                elif "google" in id_lower:
                                     provider_name_lower = "google" # Match scraped name if possible
                                else:
                                    # Try to extract a usable name from g4f provider strings
                                    if "provider" in id_lower:
                                        # Extract name between '.' and 'Provider' if possible
                                        match = re.search(r'\\.(.*?)Provider', provider_identifier)
                                        if match:
                                            provider_name_lower = match.group(1).lower()
                                        else:
                                            provider_name_lower = id_lower.replace("provider", "").strip()
                                    else:
                                        provider_name_lower = id_lower # Fallback to the identifier itself
                            elif hasattr(provider_identifier, '__name__'):
                                provider_name_for_print = provider_identifier.__name__
                                provider_name_lower = provider_identifier.__name__.lower()
                            else:
                                print(f"--- Warning: Cannot determine provider name from identifier: {provider_identifier} ---")
                                return float('inf') # Return worst score if name unknown

                            if not provider_name_lower:
                                return float('inf')

                            best_score = float('inf')
                            model_name_lower = model_name.lower()
                            # Search cache for matching provider name (case-insensitive) and model name
                            for entry in PROVIDER_PERFORMANCE_CACHE:
                                scraped_provider = entry.get('provider_name_scraped', '').lower()
                                scraped_model = entry.get('model_name_scraped', '').lower()
                                # Check if the determined provider name is part of the scraped name, or vice-versa,
                                # or if it's a known direct match (like Groq)
                                provider_match = (provider_name_lower in scraped_provider or
                                                  scraped_provider in provider_name_lower or
                                                  (provider_name_lower == 'groq' and scraped_provider == 'groq') or
                                                  (provider_name_lower == 'google' and scraped_provider == 'google') or
                                                  (provider_name_lower == 'cerebras' and scraped_provider == 'cerebras'))

                                # Basic model name matching (can be improved)
                                # Allow partial match for models like llama-4-scout vs meta-llama/llama-4-scout...
                                model_match = (model_name_lower == scraped_model or
                                               model_name_lower in scraped_model or
                                               scraped_model in model_name_lower)

                                if provider_match and model_match:
                                    score = entry.get('response_time_s', float('inf'))
                                    if score < best_score:
                                        best_score = score
                            # Return None if no match found, or the best score
                            final_score = best_score if best_score != float('inf') else float('inf')
                            # print(f"--- Perf Metric: Provider='{provider_name_for_print}' ({provider_name_lower}), Model='{model_name}', Score={final_score} ---")
                            return final_score
                        except Exception as e:
                            print(f"Error getting performance metric for {provider_identifier}, {model_name}: {e}")
                            return float('inf') # Return worst score on error

                    # Define model mapping (user-facing -> provider-specific)
                    MODEL_PROVIDER_MAP = {
                        "llama-4-scout": {
                             # Map to the actual ID Groq reported
                            "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
                             # Cerebras doesn't seem to list it based on logs, keep None
                            "cerebras": None
                        },
                        # Add other mappings if needed - example:
                        # "some-other-model": { "groq": "actual-groq-id", "cerebras": "actual-cerebras-id" }
                    }
                    # Import re for the helper function
                    import re

                    # Function to get the provider-specific model ID
                    def get_provider_model_id(user_model_name, provider_key):
                        provider_key_lower = provider_key.lower()
                        mapping = MODEL_PROVIDER_MAP.get(user_model_name)
                        if mapping:
                            # Check direct key match first (e.g., 'groq')
                            if provider_key_lower in mapping:
                                return mapping[provider_key_lower]
                            # Check partial match for strings like 'GroqAPIProvider'
                            for map_key, model_id in mapping.items():
                                if map_key in provider_key_lower:
                                    return model_id
                        # If no mapping or provider not found in map, return original name
                        return user_model_name

                    # Step 1: Try Direct Cerebras
                    cerebras_model_id = get_provider_model_id(selected_model_for_request, "cerebras")
                    # Try if the original model is in targets OR if a mapping exists (and isn't None)
                    should_try_cerebras = (selected_model_for_request in CEREBRAS_TARGET_MODELS or (cerebras_model_id is not None and cerebras_model_id != selected_model_for_request)) and CEREBRAS_API_KEY

                    if should_try_cerebras:
                        provider_name_for_attempt = "Cerebras (Direct API)"
                        attempted_direct_providers.add("cerebras")
                        # Use the directly imported class (if available) instead of map
                        # cerebras_provider_cls = provider_class_map.get("cerebras")
                        cerebras_provider_cls = CEREBRAS_PROVIDER_CLASS # Use directly imported class
                        # Use the mapped ID if it exists and is not None, otherwise use the original selection
                        model_to_use_cerebras = cerebras_model_id if cerebras_model_id else selected_model_for_request

                        if cerebras_provider_cls and model_to_use_cerebras:
                            current_args = {
                                "model": model_to_use_cerebras, # Use mapped or original ID
                                "messages": api_messages,
                                "provider": cerebras_provider_cls,
                                "api_key": CEREBRAS_API_KEY,
                                "max_tokens": max_tokens_for_request # Added max_tokens
                            }
                            try:
                                print(f"--- Attempting Priority Provider: {provider_name_for_attempt} ({model_to_use_cerebras}) ---")
                                response_content = ChatCompletion.create(**current_args)
                                # Corrected error handling block:
                                if response_content and response_content.strip():
                                    content_str = response_content.strip()
                                    low = content_str.lower()
                                    if not (low.startswith("error:") or low.startswith("you have reached") or "challenge error" in low or "rate limit" in low or "no provider found" in low or "no providers found" in low or "context_length_exceeded" in low or "request entity too large" in low or "model_not_found" in low or "token" in low): # Enhanced error check including 'token'
                                        provider_used_str = provider_name_for_attempt
                                        print(f"--- Provider {provider_name_for_attempt} succeeded! ---")
                                    else:
                                        error_msg = f"Provider {provider_name_for_attempt} returned error string: {content_str}"
                                        print(f"--- {error_msg} ---")
                                        provider_errors[provider_name_for_attempt] = content_str
                                        response_content = None
                                else:
                                    error_msg = f"Provider {provider_name_for_attempt} returned empty response."
                                    print(f"--- {error_msg} ---")
                                    provider_errors[provider_name_for_attempt] = "Returned empty response"
                                    response_content = None
                            except Exception as e:
                                error_msg_str = str(e).lower()
                                if "token" in error_msg_str and ("limit" in error_msg_str or "quota" in error_msg_str or "rate" in error_msg_str):
                                    error_msg = f"Provider {provider_name_for_attempt} failed due to token rate limit: {e}"
                                    print(f"--- {error_msg} ---")
                                    provider_errors[provider_name_for_attempt] = f"Token rate limit: {str(e)}"
                                else:
                                    error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                                    print(f"--- {error_msg} ---")
                                    provider_errors[provider_name_for_attempt] = str(e)
                                response_content = None
                        else:
                             # Corrected print statement:
                             print(f"--- Skipping Cerebras: Provider class found: {cerebras_provider_cls is not None}, Model ID to use: {model_to_use_cerebras} ---")
                             provider_errors[provider_name_for_attempt] = f"Provider class or model ID missing for Cerebras ({selected_model_for_request})"
                             response_content = None

                    # Step 2: Try Direct Groq (if Cerebras failed or wasn't applicable)
                    groq_model_id = get_provider_model_id(selected_model_for_request, "groq")
                    # Try if the mapped ID is in cache OR the original selection is in cache
                    should_try_groq = ( (groq_model_id is not None and groq_model_id in GROQ_MODELS_CACHE) or selected_model_for_request in GROQ_MODELS_CACHE ) and GROQ_API_KEY

                    if response_content is None and should_try_groq:
                        provider_name_for_attempt = "Groq (Direct API)"
                        attempted_direct_providers.add("groq")
                        # Use the mapped ID if it exists and is in the cache, otherwise fallback to original selection if that's in cache
                        model_to_use_groq = groq_model_id if (groq_model_id is not None and groq_model_id in GROQ_MODELS_CACHE) else selected_model_for_request

                        if GROQ_PROVIDER_CLASS and model_to_use_groq in GROQ_MODELS_CACHE: # Double check model is valid for Groq
                            current_args = {"api_key": GROQ_API_KEY, "provider": GROQ_PROVIDER_CLASS, "model": model_to_use_groq, "messages": api_messages, "max_tokens": max_tokens_for_request} # Added max_tokens
                            try:
                                print(f"--- Attempting Priority Provider: {provider_name_for_attempt} ({model_to_use_groq}) via g4f ---")
                                response_content = ChatCompletion.create(**current_args)
                                # Corrected error handling block:
                                if response_content and response_content.strip():
                                    content_str = response_content.strip()
                                    low = content_str.lower()
                                    if not (low.startswith("error:") or low.startswith("you have reached") or "challenge error" in low or "rate limit" in low or "no provider found" in low or "no providers found" in low or "context_length_exceeded" in low or "request entity too large" in low or "model_not_found" in low or "token" in low): # Enhanced error check including 'token'
                                        provider_used_str = provider_name_for_attempt
                                        print(f"--- Provider {provider_name_for_attempt} succeeded! ---")
                                    else:
                                        error_msg = f"Provider {provider_name_for_attempt} returned error string: {content_str}"
                                        print(f"--- {error_msg} ---")
                                        provider_errors[provider_name_for_attempt] = content_str
                                        response_content = None
                                else:
                                    error_msg = f"Provider {provider_name_for_attempt} returned empty response."
                                    print(f"--- {error_msg} ---")
                                    provider_errors[provider_name_for_attempt] = "Returned empty response"
                                    response_content = None
                            except Exception as e:
                                # --- MODIFIED EXCEPTION HANDLING ---
                                error_msg_lower = str(e).lower()
                                if "rate_limit_exceeded" in error_msg_lower and ("tokens per minute" in error_msg_lower or "tpm" in error_msg_lower or "request too large" in error_msg_lower or "token" in error_msg_lower): # Added generic token check
                                    # Specifically handle request size limit errors
                                    print(f"--- Provider {provider_name_for_attempt} failed due to request size/token limit: {e} ---")
                                    provider_errors[provider_name_for_attempt] = f"Request size/token limit exceeded: {str(e)}"
                                else:
                                    # Handle other errors
                                    error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                                    print(f"--- {error_msg} ---")
                                    provider_errors[provider_name_for_attempt] = str(e)
                                response_content = None # Ensure reset before next iteration
                                # --- END MODIFIED EXCEPTION HANDLING ---
                        else:
                            error_reason = "Groq provider class not imported" if not GROQ_PROVIDER_CLASS else f"Model '{model_to_use_groq}' not found in Groq cache"
                            print(f"--- Skipping Groq (Direct API via g4f): {error_reason}. ---")
                            provider_errors[provider_name_for_attempt] = error_reason
                            response_content = None

                    # Step 3: Try Google AI Studio
                    # Use the corrected names from GOOGLE_TARGET_MODELS
                    if response_content is None and selected_model_for_request in GOOGLE_TARGET_MODELS and GOOGLE_API_KEY:
                        provider_name_for_attempt = "Google AI Studio"
                        attempted_direct_providers.add("google") # Mark as attempted
                        try:
                            print(f"--- Attempting Priority Provider: {provider_name_for_attempt} with model: {selected_model_for_request} ---")
                            # Import Google AI library only when needed
                            import google.generativeai as genai
                            genai.configure(api_key=GOOGLE_API_KEY)

                            # Use the model name directly from GOOGLE_TARGET_MODELS
                            google_model_name = selected_model_for_request
                            print(f"--- Using Google AI Model ID: {google_model_name} ---") # Log the ID being used
                            
                            # Map the display name to the actual model ID
                            model_id_mapping = {
                                "Gemini 2.5 Flash": "gemini-2.5-flash-preview-04-17",
                                "Gemini 2.0 Flash": "gemini-2.0-flash",
                                "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite"
                            }
                            
                            actual_model_id = model_id_mapping.get(google_model_name)
                            if not actual_model_id:
                                raise ValueError(f"Unknown model name: {google_model_name}")
                                
                            print(f"--- Using actual Google AI Model ID: {actual_model_id} ---")
                            
                            try:
                                model = genai.GenerativeModel(actual_model_id)
                                actual_model_used = actual_model_id  # Store the actual model ID being used
                            except Exception as model_error:
                                print(f"--- Error initializing model {actual_model_id}: {model_error} ---")
                                # Try fallback models in order of preference
                                fallback_models = [
                                    "gemini-2.0-flash",
                                    "gemini-2.0-flash-lite"
                                ]
                                
                                for fallback_id in fallback_models:
                                    if fallback_id != actual_model_id:  # Don't retry the same model
                                        print(f"--- Attempting fallback to {fallback_id} ---")
                                        try:
                                            model = genai.GenerativeModel(fallback_id)
                                            actual_model_used = fallback_id  # Store the fallback model ID
                                            print(f"--- Successfully initialized fallback model {fallback_id} ---")
                                            break
                                        except Exception as fallback_error:
                                            print(f"--- Fallback to {fallback_id} failed: {fallback_error} ---")
                                            continue
                                else:
                                    raise ValueError(f"Failed to initialize any Gemini model. Last error: {model_error}")

                            # Convert message history to Google AI format
                            generation_config = genai.types.GenerationConfig(
                                max_output_tokens=8192, # Example config
                                temperature=0.9       # Example config
                            )
                            google_messages = []
                            system_instruction_parts = None # Store parts of system instruction
                            for msg in api_messages:
                                role = 'user' if msg['role'] == 'user' else 'model'
                                if msg['role'] == 'system':
                                    # Store system instruction content but don't add to list yet
                                    system_instruction_parts = [genai.types.Part(text=msg['content'])]
                                    print("--- Found system instruction for Google AI ---")
                                    continue
                                google_messages.append({'role': role, 'parts': [msg['content']]})

                            # Prepend system instruction if found
                            if system_instruction_parts:
                                print("--- Prepending system instruction to Google messages ---")
                                # Format according to Content structure if needed by the specific model/library version
                                # [{role: "system"...}] might not be valid, depends on API.
                                # Often, just the content parts are expected first, implicitly as system.
                                # Let's try prepending a Content object with user role (as per some examples) or just parts
                                # Safer approach: Prepend as a separate Content object if library supports it, otherwise
                                # merge with first user message or just pass parts. Let's just pass parts first.
                                # google_messages.insert(0, genai.types.Content(parts=system_instruction_parts, role="system")) # May not be valid
                                # Alternative: Prepend as the first item without role (implicit system?)
                                # google_messages.insert(0, {'parts': system_instruction_parts}) # Might work?
                                # Simplest: Just pass the parts list to the model constructor? No, needs history.
                                # Let's try combining with the first user message or sending it separately if possible.
                                # For now, let's stick to the documented way for most models: list of dicts.
                                # We will NOT pass system_instruction kwarg anymore.
                                # If system message needs special handling, requires more specific logic per model/API.

                                # Option: Add as first 'user' message (might confuse model)
                                # google_messages.insert(0, {'role': 'user', 'parts': system_instruction_parts})

                                # Option: Add as first 'model' message (less ideal)
                                # google_messages.insert(0, {'role': 'model', 'parts': system_instruction_parts})

                                # Let's assume for now the API handles a simple list and ignore system message if kwarg fails.
                                print("--- Note: System instruction kwarg failed. System message may be ignored by Google AI. ---")

                            # Remove last 'model' message if it exists (common if regenerating)
                            if google_messages and google_messages[-1]['role'] == 'model':
                                print("--- Removing last 'model' message before sending to Google AI ---")
                                google_messages.pop()

                            if not any(m['role'] == 'user' for m in google_messages):
                                # This code *should* be inside the 'if not any' block
                                print("--- Error: No user messages found for Google AI after processing. Skipping. ---")
                                provider_errors[provider_name_for_attempt] = "No user messages to send"
                                response_content = None
                            else:
                                # This code *should* be inside the 'else' block
                                print(f"--- Sending {len(google_messages)} messages to Google AI ({google_model_name}) ---")
                                # REMOVED system_instruction kwarg
                                response = model.generate_content(
                                    google_messages,
                                    generation_config=generation_config
                                )
                                # Check for response content safely
                                if response and hasattr(response, 'text') and response.text:
                                    response_content = response.text.strip()
                                    if response_content:
                                        provider_used_str = provider_name_for_attempt
                                        # Use the actual model ID that was used (including fallbacks)
                                        model_display_name = MODEL_DISPLAY_NAME_MAP.get(actual_model_used, actual_model_used)
                                        print(f"--- Provider {provider_name_for_attempt} succeeded using model: {model_display_name} ---")
                                    else:
                                        error_msg = f"Provider {provider_name_for_attempt} returned empty response text."
                                        print(f"--- {error_msg} ---")
                                        provider_errors[provider_name_for_attempt] = "Returned empty response text"
                                        response_content = None
                                else:
                                    error_msg = f"Provider {provider_name_for_attempt} returned unexpected or empty response structure: {response}"
                                    print(f"--- {error_msg} ---")
                                    provider_errors[provider_name_for_attempt] = f"Unexpected/empty response structure: {str(response)[:100]}"
                                    response_content = None

                        except ImportError:
                             print("--- Google AI Studio library not installed. Skipping. Run: pip install google-generativeai ---")
                             provider_errors[provider_name_for_attempt] = "Library not installed"
                             response_content = None
                        except Exception as e:
                            # Catch specific Google API errors if possible
                            # --- MODIFIED EXCEPTION HANDLING ---
                            error_msg_lower = str(e).lower()
                            # Google might use different phrasing, e.g. "resource has been exhausted" for quota, or specific error codes.
                            # We'll check for common patterns and specific keywords like "tokens"
                            is_token_rate_limit = (
                                (("rate_limit_exceeded" in error_msg_lower or "resource_exhausted" in error_msg_lower or "quota" in error_msg_lower) and "token" in error_msg_lower) or
                                ("request entity too large" in error_msg_lower) or # Can sometimes relate to token limits over time
                                ("token limit" in error_msg_lower)
                            )
                            if is_token_rate_limit:
                                print(f"--- Provider {provider_name_for_attempt} failed due to token rate/size limit: {e} ---")
                                provider_errors[provider_name_for_attempt] = f"Token rate/size limit exceeded: {str(e)}"
                            else:
                                error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                                print(f"--- {error_msg} ---")
                                provider_errors[provider_name_for_attempt] = str(e)
                            response_content = None
                            # --- END MODIFIED EXCEPTION HANDLING ---

                    # Step 4: Try Chutes AI (if previous steps failed)
                    # Check if the selected model is potentially available via Chutes and API key exists
                    chutes_model_to_try = None
                    if selected_model_for_request in CHUTES_MODELS_CACHE:
                        chutes_model_to_try = selected_model_for_request
                    elif selected_model_for_request == "deepseek-v3" and "deepseek-ai/DeepSeek-V3-0324" in CHUTES_MODELS_CACHE:
                        # Explicit mapping if user selects 'deepseek-v3' but Chutes has the specific one
                        chutes_model_to_try = "deepseek-ai/DeepSeek-V3-0324"
                        print(f"--- Mapping selected model '{selected_model_for_request}' to '{chutes_model_to_try}' for Chutes AI ---")

                    if response_content is None and chutes_model_to_try and CHUTES_API_KEY:
                        provider_name_for_attempt = "Chutes AI"
                        attempted_direct_providers.add("chutes") # Mark as attempted
                        try:
                            print(f"--- Attempting Priority Provider: {provider_name_for_attempt} ({chutes_model_to_try}) ---")

                            async def call_chutes_api():
                                """Helper async function to call Chutes API."""
                                nonlocal response_content, provider_used_str # Allow modification of outer scope variables
                                chutes_headers = {
                                    "Authorization": f"Bearer {CHUTES_API_KEY}",
                                    "Content-Type": "application/json"
                                }
                                # Construct Chutes API body
                                chutes_body = {
                                    "model": chutes_model_to_try, # Use the determined model name
                                    "messages": api_messages, # Assuming Chutes uses OpenAI message format
                                    "stream": False, # Request non-streaming response for simplicity here
                                    # Add other parameters like max_tokens, temperature if needed
                                    "max_tokens": max_tokens_for_request, # uncommented and set
                                    # "temperature": 0.7
                                }
                                chutes_url = f"{CHUTES_API_URL}/chat/completions"

                                async with aiohttp.ClientSession() as session:
                                    async with session.post(chutes_url, headers=chutes_headers, json=chutes_body, timeout=30) as response:
                                        if response.status == 200:
                                            data = await response.json()
                                            # Extract content (adjust based on actual Chutes response structure)
                                            # Assuming response like: {'choices': [{'message': {'content': '...'}}]}
                                            if 'choices' in data and data['choices'] and 'message' in data['choices'][0] and 'content' in data['choices'][0]['message']:
                                                extracted_content = data['choices'][0]['message']['content']
                                                if extracted_content and extracted_content.strip():
                                                    response_content = extracted_content.strip()
                                                    provider_used_str = provider_name_for_attempt
                                                    print(f"--- Provider {provider_name_for_attempt} succeeded! ---")
                                                    return True # Indicate success
                                                else:
                                                    print(f"--- Provider {provider_name_for_attempt} returned empty content. ---")
                                                    provider_errors[provider_name_for_attempt] = "Returned empty content"
                                            else:
                                                print(f"--- Provider {provider_name_for_attempt} returned unexpected JSON structure: {data} ---")
                                                provider_errors[provider_name_for_attempt] = f"Unexpected JSON structure: {str(data)[:100]}"
                                        else:
                                            error_text = await response.text()
                                            print(f"--- Provider {provider_name_for_attempt} failed with status {response.status}: {error_text[:200]} ---")
                                            provider_errors[provider_name_for_attempt] = f"Status {response.status}: {error_text[:100]}"
                                return False # Indicate failure

                            # Run the async helper function in the current event loop if possible,
                            # or create a new one if needed (asyncio.run does this).
                            try:
                                # If Flask is running with an async framework (like Quart or uvicorn with loop='asyncio'),
                                # we might be able to await directly or use asyncio.create_task.
                                # However, standard Flask runs synchronously. asyncio.run() is a common way
                                # to bridge sync/async but has implications (starts/stops loop).
                                # Consider loop = asyncio.get_event_loop(); loop.run_until_complete(call_chutes_api())
                                # if managing the loop explicitly.
                                if not asyncio.run(call_chutes_api()):
                                    # Explicitly set response_content to None if call_chutes_api returned False (failure)
                                    response_content = None
                            except RuntimeError as e:
                                 if "cannot run nested event loops" in str(e):
                                     # This happens if Flask/Gunicorn/Uvicorn is already running an event loop.
                                     # Try running in the existing loop.
                                     print("--- Detected existing event loop. Running Chutes call within it. ---")
                                     loop = asyncio.get_event_loop()
                                     if not loop.run_until_complete(call_chutes_api()):
                                         response_content = None
                                 else:
                                     raise # Re-raise other runtime errors
                        except Exception as e:
                            # Catch errors during the asyncio.run or the API call itself
                            # --- MODIFIED EXCEPTION HANDLING ---
                            error_msg_lower = str(e).lower()
                            is_size_limit_error = (
                                ("rate_limit_exceeded" in error_msg_lower and ("tokens" in error_msg_lower or "tpm" in error_msg_lower)) or
                                ("request entity too large" in error_msg_lower) or # Common HTTP error for size
                                ("context_length_exceeded" in error_msg_lower) or # Potential OpenAI/compatible error
                                ("token limit" in error_msg_lower) or # General token limit
                                (("quota" in error_msg_lower or "limit" in error_msg_lower) and "token" in error_msg_lower) # More generic token quota/limit
                            )
                            if is_size_limit_error:
                                print(f"--- Provider {provider_name_for_attempt} failed due to request size/token limit: {e} ---")
                                provider_errors[provider_name_for_attempt] = f"Request size/token limit exceeded: {str(e)}"
                            else:
                                error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                                print(f"--- {error_msg} ---")
                                provider_errors[provider_name_for_attempt] = str(e)
                            response_content = None # Ensure reset on failure
                            # --- END MODIFIED EXCEPTION HANDLING ---

                    # Step 5: Try g4f Providers (Sorted by Scraped Performance or Default)
                    if response_content is None:
                        # Get the list of potential providers (classes or strings)
                        # Use the cached model provider info
                        potential_providers = CACHED_MODEL_PROVIDER_INFO.get(selected_model_for_request, [])

                        if not potential_providers:
                            # ... (existing fallback logic) ...
                            print(f"--- No default g4f providers found for model {selected_model_for_request}. Falling back to generic g4f provider. ---")
                            # ...
                        else:
                            # Create list of (provider, score) tuples using the updated metric function
                            scored_providers = []
                            for p_id in potential_providers:
                                # Skip providers already tried directly
                                provider_key = ""
                                if isinstance(p_id, str):
                                     if "groq" in p_id.lower() and "groq" in attempted_direct_providers: continue
                                     if "cerebras" in p_id.lower() and "cerebras" in attempted_direct_providers: continue
                                     if "google" in p_id.lower() and "google" in attempted_direct_providers: continue
                                     if "chutes" in p_id.lower() and "chutes" in attempted_direct_providers: continue
                                elif hasattr(p_id, '__name__'):
                                     p_name_lower = p_id.__name__.lower()
                                     if "groq" in p_name_lower and "groq" in attempted_direct_providers: continue
                                     if "cerebras" in p_name_lower and "cerebras" in attempted_direct_providers: continue
                                     # Add others if needed
                                else:
                                     continue # Skip unknown identifier types

                                score = get_scraped_performance_metric(p_id, selected_model_for_request)
                                scored_providers.append((p_id, score))

                            # Sort by score (lower response time is better, float('inf') places unknowns last)
                            scored_providers.sort(key=lambda x: x[1])

                            print(f"--- g4f providers sorted by performance: {[( (p[0].__name__ if hasattr(p[0], '__name__') else p[0]), p[1]) for p in scored_providers]} ---")

                            # Iterate through the sorted providers
                            for provider_identifier, score in scored_providers:
                                # Determine provider class and name for attempt
                                provider_class_to_use = None
                                provider_name_for_attempt = "Unknown"
                                api_key_to_use = None # Reset API key

                                if isinstance(provider_identifier, str):
                                    # This case shouldn't happen often for g4f list, primarily for direct mapped ones
                                    # But if it does, we might need to map string back to class object if possible
                                    # For now, we mainly expect class objects here from model_provider_info
                                    print(f"--- Warning: Skipping string identifier '{provider_identifier}' in g4f sorted list. ---")
                                    continue
                                elif hasattr(provider_identifier, '__name__'):
                                    provider_class_to_use = provider_identifier
                                    provider_name_for_attempt = f"{provider_class_to_use.__name__} (g4f List, Perf: {score})"
                                else:
                                    print(f"--- Warning: Skipping unknown identifier '{provider_identifier}' in g4f sorted list. ---")
                                    continue

                                # Get the correct model ID for this specific provider
                                model_id_for_g4f_attempt = get_provider_model_id(selected_model_for_request, provider_class_to_use.__name__)

                                # Use max_completion_tokens for o3 model, max_tokens for others
                                if selected_model_for_request.lower() == "o3":
                                    current_args = {"provider": provider_class_to_use, "model": model_id_for_g4f_attempt, "messages": api_messages, "max_completion_tokens": max_tokens_for_request}
                                else:
                                    current_args = {"provider": provider_class_to_use, "model": model_id_for_g4f_attempt, "messages": api_messages, "max_tokens": max_tokens_for_request}

                                # Add API keys if required by the specific g4f provider (Optional - g4f might handle some internally)
                                # Example: if provider_class_to_use == SomeProviderRequiringKey: current_args["api_key"] = SOME_KEY

                                try:
                                    print(f"--- Attempting provider: {provider_name_for_attempt} with model {model_id_for_g4f_attempt} ---")
                                    # Special handling for o3 model which requires max_completion_tokens
                                    if model_id_for_g4f_attempt == "o3":
                                        if "max_tokens" in current_args:
                                            current_args["max_completion_tokens"] = current_args.pop("max_tokens")
                                    response_content = ChatCompletion.create(**current_args)
                                    # ... (rest of try/except/check block remains the same) ...
                                    if response_content and response_content.strip():
                                        content_str = response_content.strip()
                                        low = content_str.lower()
                                        # Use a simpler error check for g4f providers
                                        if not (low.startswith("error:") or "unable to fetch" in low or "invalid key" in low or "no provider found" in low or "no providers found" in low or "context_length_exceeded" in low or "model_not_found" in low or "token" in low or "limit" in low): # Enhanced error check for tokens/limits
                                            provider_used_str = provider_name_for_attempt # Use the detailed name
                                            print(f"--- Provider {provider_name_for_attempt} succeeded! ---")
                                            break # Exit loop on first success
                                        else:
                                            error_msg = f"Provider {provider_name_for_attempt} returned error string: {content_str}"
                                            print(f"--- {error_msg} ---")
                                            provider_errors[provider_name_for_attempt] = content_str
                                            response_content = None
                                    else:
                                        error_msg = f"Provider {provider_name_for_attempt} returned empty response."
                                        print(f"--- {error_msg} ---")
                                        provider_errors[provider_name_for_attempt] = "Returned empty response"
                                        response_content = None
                                except Exception as e:
                                    # --- MODIFIED EXCEPTION HANDLING ---
                                    error_msg_lower = str(e).lower()
                                    is_size_limit_error = (
                                        ("rate_limit_exceeded" in error_msg_lower and ("tokens" in error_msg_lower or "tpm" in error_msg_lower)) or
                                        ("request entity too large" in error_msg_lower) or
                                        ("context_length_exceeded" in error_msg_lower) or # g4f might pass this through
                                        ("token limit" in error_msg_lower) or
                                        ("prompt is too long" in error_msg_lower) or # Another possible phrasing
                                        ("maximum context length" in error_msg_lower) or
                                        (("quota" in error_msg_lower or "limit" in error_msg_lower) and "token" in error_msg_lower) # More generic token quota/limit for g4f
                                    )
                                    if is_size_limit_error:
                                        print(f"--- Provider {provider_name_for_attempt} failed due to request size/token limit: {e} ---")
                                        provider_errors[provider_name_for_attempt] = f"Request size/token limit exceeded: {str(e)}"
                                    else:
                                        error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                                        print(f"--- {error_msg} ---")
                                        provider_errors[provider_name_for_attempt] = str(e)
                                    response_content = None # Ensure reset before next iteration
                                    # --- END MODIFIED EXCEPTION HANDLING ---

                    # Check final outcome after all attempts
                    if response_content is None and provider_used_str == "None Attempted":
                         final_error_message = f"Error: No suitable providers found or attempted for model {selected_model_for_request}."
                         provider_used_str = "None Found/Attempted"
                    elif response_content is None:
                         error_details = "; ".join([f"{p}: {e}" for p, e in provider_errors.items()])
                         final_error_message = f"Error: All attempted providers failed for model {selected_model_for_request}. Details: {error_details}"
                         provider_used_str = "All Failed"
                    # --- End Provider Selection ---

                    # After getting response_content, check if it indicates outdated information
                    if response_content and retry_count < max_retries - 1:
                        response_lower = response_content.lower()
                        outdated_indicators = [
                            "my knowledge cutoff",
                            "my training data only goes up to",
                            "i don't have information beyond",
                            "i don't have access to information after",
                            "i don't have real-time information",
                            "i don't have current information",
                            "i don't have the latest information",
                            "i don't have up-to-date information"
                        ]
                        
                        if any(indicator in response_lower for indicator in outdated_indicators):
                            print(f"--- LLM response indicates outdated information, retrying with web search (attempt {retry_count + 1}) ---")
                            search_results_str = perform_web_search(prompt_to_use)
                            if search_results_str:
                                # Modify the last user message content with web search results
                                if api_messages and api_messages[-1]["role"] == "user":
                                    original_prompt_content = api_messages[-1]["content"]
                                    api_messages[-1]["content"] = f"Web Search Results:\n{search_results_str}\n\nOriginal User Prompt:\n{original_prompt_content}"
                                else:
                                    api_messages.insert(0, {"role": "system", "content": f"Context from web search:\n{search_results_str}"})
                            retry_count += 1
                            should_retry = True
                        else:
                            should_retry = False
                    else:
                        should_retry = False

                # Process successful response or final error
                # Handle regeneration: remove previous AI message FIRST if needed
                if remove_last_ai_msg_from_actual_history:
                    if current_chat["history"] and current_chat["history"][-1]["role"] == "assistant":
                        print("--- Removing previous assistant message for regeneration ---")
                        current_chat["history"].pop()

                if response_content is not None:
                    # Add user prompt to actual history if it wasn't a regeneration
                    if not is_regeneration:
                        current_chat["history"].append({"role": "user", "content": prompt_from_input, "timestamp": datetime.now().isoformat()}) # Use original prompt here
                    # Add the successful assistant response with the actual model used
                    current_chat["history"].append({
                        "role": "assistant", 
                        "content": response_content, 
                        "model": actual_model_used if 'actual_model_used' in locals() else selected_model_for_request, 
                        "provider": provider_used_str, 
                        "timestamp": datetime.now().isoformat()
                    })
                    # Auto-name chat if it's new
                    if current_chat.get("name") == "New Chat" and any(msg["role"] == "user" for msg in current_chat["history"]):
                         first_user_prompt = next((msg["content"] for msg in current_chat["history"] if msg["role"] == "user"), None)
                         if first_user_prompt:
                             clean_prompt = ''.join(c for c in ' '.join(first_user_prompt.split()[:6]) if c.isalnum() or c.isspace()).strip()
                             response_timestamp = current_chat["history"][-1].get("timestamp") if current_chat["history"] else datetime.now().isoformat()
                             timestamp_str = datetime.fromisoformat(response_timestamp).strftime("%b %d, %I:%M%p")
                             chat_name = f"{clean_prompt[:30]}... ({timestamp_str})" if clean_prompt else f"Chat ({timestamp_str})"
                             current_chat["name"] = chat_name
                elif final_error_message:
                     # Add user prompt to history only if it was a NEW prompt that failed
                     # (Don't re-add the user prompt if regeneration failed)
                     if not is_regeneration:
                         current_chat["history"].append({"role": "user", "content": prompt_from_input, "timestamp": datetime.now().isoformat()}) # Use original prompt
                     # Add error message as assistant response
                     current_chat["history"].append({"role": "assistant", "content": final_error_message, "model": selected_model_for_request, "provider": provider_used_str, "timestamp": datetime.now().isoformat()})

        # Save chat history after processing response/error
        save_chats(chats)
        # DEBUG: Re-load and check history immediately after save
        try:
            reloaded_chats = load_chats()
            reloaded_current_chat = reloaded_chats.get(session['current_chat'], {})
            print(f"--- POST Save Check: Reloaded history length: {len(reloaded_current_chat.get('history', []))} ---")
        except Exception as e:
            print(f"--- POST Save Check: Error reloading chats immediately after save: {e} ---")
        # END DEBUG
        return redirect(url_for('index')) # Redirect after processing POST

    # --- Prepare data for rendering the page (GET request or after POST redirect) ---
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

    # Use MODEL_DISPLAY_NAME_MAP and performance data for dropdown text
    model_options_html = ''
    seen_display_names = set()
    for model_name, provider_count, intel_index, resp_time in available_models_sorted_list:
        # Skip entries with empty model name
        if not model_name or not model_name.strip():
            continue
        display_name = MODEL_DISPLAY_NAME_MAP.get(model_name.lower().replace('-','').replace(' ',''), model_name).strip()
        # Merge duplicate display names
        if not display_name or display_name in seen_display_names:
            continue
        seen_display_names.add(display_name)
        selected_attr = "selected" if model_name == current_model else ""
        # Format performance string, handle missing data (resp_time == inf)
        if resp_time != float('inf') and intel_index > 0:
            perf_str = f"({intel_index}, {resp_time:.2f}s)"
        elif resp_time != float('inf'):
             perf_str = f"(Intel N/A, {resp_time:.2f}s)"
        else:
            perf_str = "(Perf N/A)"
        model_options_html += f'<option value="{model_name}" {selected_attr}>{display_name} {perf_str}</option>'

    # Determine checked state for web search (default to smart)
    # Use the web_search_mode defined earlier for GET requests
    web_search_html = f'''
        <div style="margin-bottom: 10px;">
            <label style="margin-right: 10px; font-weight: bold;">Web Search:</label>
            <label style="margin-right: 15px;">
                <input type="radio" name="web_search_mode" value="off" {'checked' if web_search_mode == 'off' else ''}> Off
            </label>
            <label style="margin-right: 15px;">
                <input type="radio" name="web_search_mode" value="smart" {'checked' if web_search_mode == 'smart' else ''}> Smart
            </label>
            <label>
                <input type="radio" name="web_search_mode" value="on" {'checked' if web_search_mode == 'on' else ''}> On
            </label>
        </div>
    '''

    # Navigation buttons moved higher
    nav_links_html = f'''
        <div class="nav-links" style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <a href="/saved_chats" style="padding: 8px 15px; border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333; text-decoration: none;">Saved Chats</a>
            <button type="submit" name="new_chat" value="1" style="padding: 8px 15px; border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333; cursor: pointer;">New Chat</button>
            <button type="submit" name="delete_chat" value="1" style="padding: 8px 15px; border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333; cursor: pointer;">Delete Chat</button>
        </div>
    '''

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 0; background-color: #fff; }}
        #message-container {{ height: calc(100vh - 150px); overflow-y: auto; padding: 10px; border-bottom: 1px solid #ccc; }}
        #message-container > div {{ margin:4px 0; padding:6px; border-bottom:1px solid #eee; border-radius: 4px; }}
        #message-container small {{ color: #555; font-size: 0.8em; margin-left: 5px; }}
        #input-area {{ padding: 10px; background-color: #f0f0f0; border-top: 1px solid #ccc; flex-shrink: 0; }}
        textarea {{ width: 100%; box-sizing: border-box; height: 80px; font-size: 1em; margin-bottom: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; }} /* Increased height slightly */
        select {{ width: 100%; padding: 8px; margin-bottom: 10px; font-size: 1em; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }} /* Added margin-bottom */
        input[type="submit"], button {{ padding: 10px; font-size: 1em; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; cursor: pointer; }} /* Slightly larger padding */
        .button-row {{ display: flex; gap: 10px; margin-top: 10px; }} /* Added margin-top */
        .button-row input {{ flex-grow: 1; background-color: #4CAF50; color: white; border-color: #4CAF50; }} /* Style Send button */
        .button-row input[name="regenerate"] {{ background-color: #ff9800; border-color: #ff9800; color: white; }} /* Style Regenerate */
        label {{ vertical-align: middle; }}
        input[type="radio"] {{ vertical-align: middle; margin-right: 2px; }}
        /* Styles for nav-links are now inline */
    </style>
</head>
<body>
    <div id="message-container">{history_html}</div>
    <div id="input-area">
        <form method="post" style="margin: 0;">
            <select name="model">{model_options_html}</select>
            {nav_links_html}
            {web_search_html}
            <textarea name="prompt" placeholder="Type your message..." autofocus></textarea>
            <div class="button-row">
                <input type="submit" name="send" value="Send">
                {('<input type="submit" name="regenerate" value="Regenerate">') if current_chat.get("history") else ""}
            </div>
        </form>
    </div>
    <script>
        // Scroll to bottom on page load/update
        var messageContainer = document.getElementById('message-container');
        messageContainer.scrollTop = messageContainer.scrollHeight;
    </script>
</body>
</html>'''

@app.route('/saved_chats')
@rate_limit()
def saved_chats():
    chats = load_chats()
    # Sort chats by creation date, most recent first
    sorted_chat_items = sorted(chats.items(), key=lambda item: item[1].get('created_at', '1970-01-01T00:00:00'), reverse=True)
    chats_html = "".join([f'''<div style="margin: 8px 0; padding: 8px; border-bottom: 1px solid #ddd; background-color: #fff; border-radius: 4px;">
                           <a href="/load_chat/{chat_id}" style="text-decoration: none; color: #007bff; font-weight: bold;">{html.escape(chat_data.get('name', 'Unnamed Chat'))}</a><br>
                           <small style="color:#666">Created: {datetime.fromisoformat(chat_data.get('created_at', '1970-01-01T00:00:00')).strftime("%b %d, %Y - %I:%M %p")}</small>
                           <form method="post" action="/delete_saved_chat/{chat_id}" style="display: inline; float: right;">
                               <button type="submit" onclick="return confirm('Delete this chat?');" style="color: red; background: none; border: none; cursor: pointer; padding: 0 5px;">Delete</button>
                           </form>
                         </div>''' for chat_id, chat_data in sorted_chat_items])
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Saved Chats</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: sans-serif; margin: 10px; background-color: #f4f4f4; }}
        h3 {{ border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
        a {{ color: #007bff; text-decoration: none; }}
        .back-link {{ margin-top: 15px; display: inline-block; }}
    </style>
</head>
<body>
    <h3>Saved Chats:</h3>
    {chats_html if chats else "<p>No saved chats yet.</p>"}
    <p class="back-link"><a href="/">< Back to Chat</a></p>
</body>
</html>'''

@app.route('/delete_saved_chat/<chat_id>', methods=['POST'])
@rate_limit()
def delete_saved_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        del chats[chat_id]
        save_chats(chats)
    return redirect(url_for('saved_chats'))

@app.route('/load_chat/<chat_id>')
@rate_limit()
def load_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        session['current_chat'] = chat_id
        return redirect(url_for('index')) # Load the chat in the main view
    else:
        # If chat ID is invalid, redirect to saved chats list
        return redirect(url_for('saved_chats'))

# Add error handlers
@app.errorhandler(404)
def not_found_error(error):
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Page Not Found</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; padding: 50px; }}
        h1 {{ color: #333; }}
        a {{ color: #007bff; text-decoration: none; }}
    </style>
</head>
<body>
    <h1>Page Not Found</h1>
    <p>The page you're looking for doesn't exist.</p>
    <p><a href="/">Return to Chat</a></p>
</body>
</html>''', 404

@app.errorhandler(429)
def too_many_requests(error):
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Too Many Requests</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; padding: 50px; }}
        h1 {{ color: #333; }}
        a {{ color: #007bff; text-decoration: none; }}
    </style>
</head>
<body>
    <h1>Too Many Requests</h1>
    <p>Please wait a moment before trying again.</p>
    <p><a href="/">Return to Chat</a></p>
</body>
</html>''', 429

@app.errorhandler(500)
def internal_error(error):
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Internal Server Error</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; padding: 50px; }}
        h1 {{ color: #333; }}
        a {{ color: #007bff; text-decoration: none; }}
    </style>
</head>
<body>
    <h1>Internal Server Error</h1>
    <p>Something went wrong. Please try again later.</p>
    <p><a href="/">Return to Chat</a></p>
</body>
</html>''', 500

# --- Chutes AI Integration ---

async def fetch_chutes_models():
    """Fetches the list of available models from the Chutes AI API."""
    global CHUTES_MODELS_CACHE
    if not CHUTES_API_KEY:
        print("--- Chutes API Key not found, skipping model fetch. ---")
        CHUTES_MODELS_CACHE = []
        return

    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Accept": "application/json"
    }
    models_url = f"{CHUTES_API_URL}/models"
    print(f"--- Fetching models from Chutes AI: {models_url} ---")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(models_url, headers=headers, timeout=15) as response:
                response.raise_for_status()
                data = await response.json()
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                    CHUTES_MODELS_CACHE = [model.get('id') for model in data['data'] if model.get('id')]
                    print(f"--- Found {len(CHUTES_MODELS_CACHE)} models from Chutes AI: {CHUTES_MODELS_CACHE[:5]}... ---")
                else:
                    print(f"--- Unexpected format received from Chutes AI /models endpoint: {data} ---")
                    CHUTES_MODELS_CACHE = []

    except Exception as e:
        print(f"--- Error fetching models from Chutes AI: {e} ---")
        CHUTES_MODELS_CACHE = []

# --- Groq API Integration ---

async def fetch_groq_models():
    """Fetches the list of available models from the Groq API."""
    global GROQ_MODELS_CACHE
    if not GROQ_API_KEY:
        print("--- Groq API Key not found, skipping model fetch. ---")
        GROQ_MODELS_CACHE = []
        return

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Accept": "application/json"
    }
    models_url = f"{GROQ_API_URL}/models"
    print(f"--- Fetching models from Groq API: {models_url} ---")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(models_url, headers=headers, timeout=15) as response:
                response.raise_for_status()
                data = await response.json()
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                    GROQ_MODELS_CACHE = [model.get('id') for model in data['data'] if model.get('id')]
                    print(f"--- Found {len(GROQ_MODELS_CACHE)} models from Groq API: {GROQ_MODELS_CACHE[:5]}... ---")
                else:
                    print(f"--- Unexpected format received from Groq API /models endpoint: {data} ---")
                    GROQ_MODELS_CACHE = []

    except Exception as e:
        print(f"--- Error fetching models from Groq API: {e} ---")
        GROQ_MODELS_CACHE = []

# --- End Groq API Integration ---

# --- End Chutes AI Integration ---

# --- Global variable to store scraped performance data ---
PROVIDER_PERFORMANCE_CACHE = {}

# --- Load or Scrape performance data (Moved Here) ---
print("--- [MODULE LOAD] Entering performance data handling ---")
print(f"--- [MODULE LOAD] Checking for CSV at: {PERFORMANCE_CSV_PATH} ---")
PROVIDER_PERFORMANCE_CACHE = load_performance_from_csv(PERFORMANCE_CSV_PATH)
print(f"--- [MODULE LOAD] load_performance_from_csv returned cache size: {len(PROVIDER_PERFORMANCE_CACHE)} ---")

if not PROVIDER_PERFORMANCE_CACHE:
    print("--- [MODULE LOAD] Cache empty or load failed. Attempting scrape... ---")
    scraped_data = scrape_provider_performance()
    print(f"--- [MODULE LOAD] scrape_provider_performance returned data size: {len(scraped_data)} ---")
    if scraped_data:
        print("--- [MODULE LOAD] Scrape successful. Attempting save... ---")
        save_succeeded = save_performance_to_csv(scraped_data, PERFORMANCE_CSV_PATH)
        print(f"--- [MODULE LOAD] save_performance_to_csv returned: {save_succeeded} ---")
        if save_succeeded:
            PROVIDER_PERFORMANCE_CACHE = scraped_data # Update cache only if save succeeded
            print("--- [MODULE LOAD] Cache updated with scraped data. ---")
        else:
             print("--- [MODULE LOAD] CSV save failed. Cache remains empty. ---")
    else:
        print("--- [MODULE LOAD] Scraping failed or returned no data. Cache remains empty. ---")
        PROVIDER_PERFORMANCE_CACHE = [] # Ensure it's an empty list if scraping fails
else:
     print(f"--- [MODULE LOAD] Initialization complete using existing cached data (size: {len(PROVIDER_PERFORMANCE_CACHE)}). ---")
# --- End Moved Data Loading ---


# --- Fetch External Models on Module Load ---
# --- Fetch Chutes Models (Async) ---
try:
    print("--- [MODULE LOAD] Initializing: Fetching Chutes AI models... ---")
    # Run the async function to populate CHUTES_MODELS_CACHE
    # Handle potential RuntimeError if an event loop is already running (less common in standard WSGI)
    try:
        asyncio.run(fetch_chutes_models())
    except RuntimeError as e:
        if "cannot run nested event loops" in str(e):
            print("--- [MODULE LOAD] Warning: Detected existing event loop for Chutes fetch. Attempting to use it. ---")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(fetch_chutes_models())
        else:
            raise # Re-raise other runtime errors
    print(f"--- [MODULE LOAD] Chutes models fetched: {len(CHUTES_MODELS_CACHE)} ---")
except Exception as e:
    print(f"--- [MODULE LOAD] Error running fetch_chutes_models at startup: {e} ---")
    CHUTES_MODELS_CACHE = [] # Ensure it's empty on error
# --- End Chutes Fetch ---

# --- Fetch Groq Models (Async) ---
try:
    print("--- [MODULE LOAD] Initializing: Fetching Groq API models... ---")
    # Handle potential RuntimeError if an event loop is already running
    try:
        asyncio.run(fetch_groq_models())
    except RuntimeError as e:
        if "cannot run nested event loops" in str(e):
            print("--- [MODULE LOAD] Warning: Detected existing event loop for Groq fetch. Attempting to use it. ---")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(fetch_groq_models())
        else:
            raise # Re-raise other runtime errors
    print(f"--- [MODULE LOAD] Groq models fetched: {len(GROQ_MODELS_CACHE)} ---")
except Exception as e:
    print(f"--- [MODULE LOAD] Error running fetch_groq_models at startup: {e} ---")
    GROQ_MODELS_CACHE = [] # Ensure it's empty on error
# --- End Groq Fetch ---

# --- Fetch Free LLM API Info (Async) ---
try:
    print("--- [MODULE LOAD] Initializing: Fetching Free LLM API Info... ---")
    try:
        asyncio.run(fetch_and_parse_free_llm_apis())
    except RuntimeError as e:
        if "cannot run nested event loops" in str(e):
            print("--- [MODULE LOAD] Warning: Detected existing event loop for Free LLM API fetch. Attempting to use it. ---")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(fetch_and_parse_free_llm_apis())
        else:
            raise
    print("--- [MODULE LOAD] Free LLM API info fetched/parsed:", len(CACHED_FREE_API_PROVIDERS) if CACHED_FREE_API_PROVIDERS else "None or Error")
except Exception as e:
    print(f"--- [MODULE LOAD] Error running fetch_and_parse_free_llm_apis at startup: {e} ---")
    CACHED_FREE_API_PROVIDERS = {}
# --- End Free LLM API Info Fetch ---

# --- Load Context Window Data from CSV ---
try:
    print("--- [MODULE LOAD] Initializing: Loading Context Window Data from CSV... ---")
    load_context_window_data_from_csv()
    print("--- [MODULE LOAD] Context window data loaded:", len(MODEL_CONTEXT_WINDOW_CACHE) if MODEL_CONTEXT_WINDOW_CACHE else "None or Error")
except Exception as e:
    print(f"--- [MODULE LOAD] Error during initial context window data load: {e} ---")
    MODEL_CONTEXT_WINDOW_CACHE = {}
# --- End Load Context Window Data ---

# --- End Fetch External Models ---

# --- New Global List for Known Free Model Offerings ---
KNOWN_FREE_MODEL_OFFERINGS = [
    {
        "provider_key": "groq", # Matches direct_provider_to_context_csv_name or provider_ref.__name__.lower()
        "model_key": "llama3-8b-8192", # Matches model_name.lower() or normalized display name
        "is_genuinely_free": True,
        "notes": "Groq Llama3 8B (e.g., from GROQ_MODELS_CACHE)"
    },
    {
        "provider_key": "google (ai studio)", # Matches direct_provider_to_context_csv_name
        "model_key": "gemini 2.5 flash (ai_studio)", # Matches model_name.lower() for "Gemini 2.5 Flash (AI_Studio)"
        "is_genuinely_free": True,
        "notes": "Google Gemini 2.5 Flash via AI Studio (e.g., from GOOGLE_TARGET_MODELS)"
    },
    {
        "provider_key": "groq",
        "model_key": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "is_genuinely_free": True,
        "notes": "Groq Llama 4 Maverick 17B (as per free-llm-api-resources)"
    },
    # Add more entries as needed, for example:
    # {
    #     "provider_key": "someg4fprovidername", # This would be SomeG4fProvider.__name__.lower()
    #     "model_key": "g4f-model-x-name-lower", # This would be model.name.lower() from g4f
    #     "is_genuinely_free": True,
    #     "notes": "A specific free model from a g4f provider"
    # }
]
# --- End New Global List ---

# --- Function to initialize model cache ---
def normalize_model_name(name):
    # Remove version numbers and parameters
    return re.sub(r'[\d\.]+[bBmMkK]?[-_]?', '', name).lower().replace(' ', '').replace('-', '')

def initialize_model_cache():
    """
    Initializes the global model cache by fetching available models and their providers.
    Falls back to default models if there are any errors.
    """
    global CACHED_AVAILABLE_MODELS_SORTED_LIST, CACHED_MODEL_PROVIDER_INFO, CACHED_PROVIDER_CLASS_MAP
    
    # Define default models with their properties
    default_models = [
        ("gpt-3.5-turbo", 1, 85, 0.5),      # OpenAI
        ("gpt-4", 1, 95, 1.0),              # OpenAI
        ("claude-3-opus", 1, 98, 1.2),      # Anthropic
        ("claude-3-sonnet", 1, 96, 0.8),    # Anthropic
        ("claude-3-haiku", 1, 94, 0.4),     # Anthropic
        ("gemini-pro", 1, 93, 0.6),         # Google
        ("llama-2-70b", 1, 92, 1.5),        # Meta
        ("llama-2-13b", 1, 90, 1.0),        # Meta
        ("llama-2-7b", 1, 88, 0.8),         # Meta
        ("mistral-7b", 1, 89, 0.7),         # Mistral AI
        ("mixtral-8x7b", 1, 91, 1.1),       # Mistral AI
        ("qwen-72b", 1, 93, 1.3),           # Alibaba
        ("qwen-14b", 1, 90, 0.9),           # Alibaba
        ("qwen-7b", 1, 88, 0.7),            # Alibaba
        ("deepseek-coder-33b", 1, 92, 1.2),  # DeepSeek
        ("deepseek-coder-6.7b", 1, 89, 0.8), # DeepSeek
        ("deepseek-llm-67b", 1, 93, 1.4),    # DeepSeek
        ("deepseek-llm-7b", 1, 90, 0.9),     # DeepSeek
        ("yi-34b", 1, 91, 1.2),             # 01.AI
        ("yi-6b", 1, 88, 0.7)               # 01.AI
    ]
    
    print("\n=== [CACHE] Starting model cache initialization ===")
    start_time = time.time()
    
    try:
        # Try to get models from g4f with timeout
        print("--- [CACHE] Fetching models from g4f...")
        list_data, info_data, map_data = get_available_models_with_provider_counts()
        
        # Validate the returned data
        if not list_data or not isinstance(list_data, list):
            raise ValueError("No valid model data returned from g4f")
            
        print(f"--- [CACHE] Successfully retrieved {len(list_data)} models from g4f")
        
        # Initialize caches with fetched data
        CACHED_AVAILABLE_MODELS_SORTED_LIST = list_data
        CACHED_MODEL_PROVIDER_INFO = info_data if info_data and isinstance(info_data, dict) else {}
        CACHED_PROVIDER_CLASS_MAP = map_data if map_data and isinstance(map_data, dict) else {}
        
        # Add any default models that aren't already in the list
        existing_models = {model[0].lower() for model in CACHED_AVAILABLE_MODELS_SORTED_LIST}
        added_defaults = 0
        
        for default_model in default_models:
            model_name = default_model[0]
            if model_name.lower() not in existing_models:
                CACHED_AVAILABLE_MODELS_SORTED_LIST.append(default_model)
                CACHED_MODEL_PROVIDER_INFO[model_name] = []
                added_defaults += 1
                
        if added_defaults > 0:
            print(f"--- [CACHE] Added {added_defaults} default models not found in g4f")
        
        # Sort the final list with error handling
        try:
            CACHED_AVAILABLE_MODELS_SORTED_LIST.sort(
                key=lambda x: (
                    -int(x[1]) if isinstance(x[1], (int, float)) else 0,  # Provider count
                    -int(x[2]) if isinstance(x[2], (int, float)) else 0,  # Intelligence index
                    float('inf') if not isinstance(x[3], (int, float)) else float(x[3]),  # Response time
                    str(x[0])  # Model name
                )
            )
        except Exception as sort_error:
            print(f"--- [CACHE] Error sorting models, using default order: {sort_error}")
            CACHED_AVAILABLE_MODELS_SORTED_LIST.sort(key=lambda x: str(x[0]))
        
        # Log success
        elapsed = time.time() - start_time
        print(f"=== [CACHE] Model cache initialized in {elapsed:.2f}s ===")
        print(f"=== [CACHE] Total models: {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)}")
        print(f"=== [CACHE] Top 5 models: {[m[0] for m in CACHED_AVAILABLE_MODELS_SORTED_LIST[:5]]}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        # Fall back to default models on any error
        elapsed = time.time() - start_time
        print(f"--- [CACHE] Error initializing model cache after {elapsed:.2f}s: {str(e)}")
        print("--- [CACHE] Falling back to default models")
        import traceback
        print(f"--- [CACHE] Traceback: {traceback.format_exc()}")
        
        # Set default values
        CACHED_AVAILABLE_MODELS_SORTED_LIST = default_models
        CACHED_MODEL_PROVIDER_INFO = {model[0]: [] for model in default_models}
        CACHED_PROVIDER_CLASS_MAP = {}
        
        print(f"=== [CACHE] Initialized with {len(default_models)} default models ===\n")

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
    start_time = time.time()
    initialize_model_cache()
    elapsed = time.time() - start_time
    print(f"--- [INIT] Model cache initialized in {elapsed:.2f} seconds")
    
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
    except Exception as e:
        print(f"--- [WARNING] Could not load performance data: {e}")
    
    print("\n" + "="*60)
    print(f"  APPLICATION INITIALIZATION COMPLETE")
    print("="*60 + "\n")

# Initialize the application
if __name__ == '__main__':
    try:
        # Initialize application components
        initialize_application()
        
        # Start the Flask development server
        host = '0.0.0.0'  # Listen on all network interfaces
        port = 5000
        print(f"\n{'='*60}")
        print(f"  STARTING FLASK APPLICATION")
        print(f"  - Host: {host}")
        print(f"  - Port: {port}")
        print(f"  - Debug: {True}")
        print(f"  - Models available: {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)}")
        print(f"  - Working directory: {os.getcwd()}")
        print(f"  - Python version: {sys.version}")
        print(f"  - g4f version: {getattr(g4f, '__version__', 'unknown')}")
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
        
        # Try to provide helpful troubleshooting steps
        print("\nTroubleshooting steps:")
        print("1. Check if all required environment variables are set")
        print("2. Verify internet connection for model downloads")
        print("3. Check if port 5000 is available")
        print("4. Try running with a clean virtual environment")
        print("5. Check the logs above for specific error messages\n")

def initialize_async_data():
    """Initialize async data at module level"""
    try:
        print("--- [STARTUP] Starting async data initialization... ---")
        # Run the async functions to populate the caches
        asyncio.run(fetch_chutes_models())
        asyncio.run(fetch_groq_models())
        print("--- [STARTUP] Async data initialization complete ---")
    except Exception as e:
        print(f"--- [STARTUP] Error initializing async data: {e} ---")
        import traceback
        print(f"--- [STARTUP] Traceback: {traceback.format_exc()} ---")

if __name__ == '__main__':
    # Initialize async data before starting the Flask app
    initialize_async_data()
    app.run(host='192.168.0.11', port=5000, debug=True)
