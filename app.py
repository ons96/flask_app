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


# Load environment variables from .env file
# Load environment variables from the application directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Load from same directory as app.py
load_dotenv(dotenv_path=dotenv_path)

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from flask import Flask, request, session, redirect, url_for

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
GOOGLE_TARGET_MODELS = {"Gemini 2.5 Flash", "Gemini 2.5 Pro"} # Corrected Google AI Studio models

# --- Add Display Name Mapping (Moved Higher) ---
MODEL_DISPLAY_NAME_MAP = {
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "qwen-qwq-32b": "QwQ-32B", # Added display name for QwQ
    # Add mappings for Gemini variants to ensure consistent display
    "Gemini 2.5 Flash": "Gemini 2.5 Flash",
    "Gemini 2.0 Flash (AI Studio)": "Gemini 2.5 Flash", # Map potential internal name
    "Gemini 2.5 Pro": "Gemini 2.5 Pro",
    "Gemini 1.5 Pro (AI Studio)": "Gemini 2.5 Pro"  # Map potential internal name
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
        INTELLIGENCE_INDEX_COL = 3 # Assuming 4th column
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
                            'intelligence_index': intelligence_index, # Added
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
                                processed_row[field] = dtype(value_str)
                            except (ValueError, TypeError):
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

# --- Web Search Functionality ---
def perform_web_search(query, num_results=3):
    """Performs a web search using DuckDuckGo and returns formatted results."""
    print(f"--- Performing web search for: {query} ---")
    results_str = "Web search results:\n" # Changed separator
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=num_results))
            if not search_results:
                results_str += "No results found.\n"
            else:
                for i, result in enumerate(search_results):
                    title = result.get('title', 'No Title')
                    body = result.get('body', 'No Snippet')
                    href = result.get('href', 'N/A')
                    results_str += f"{i+1}. {title}\n   Snippet: {body}\n   Source: {href}\n" # Improved formatting
    except Exception as e:
        print(f"--- Error during web search: {e} ---")
        results_str += "Search failed.\n"
    print("--- Web search complete. --- ")
    return results_str

# --- End Web Search ---


# Function to load chat history
def load_chats():
    """Loads chat histories from the JSON storage file."""
    if os.path.exists(CHAT_STORAGE):
        try:
            with open(CHAT_STORAGE, "r", encoding='utf-8') as f:
                content = f.read()
                if not content: return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {CHAT_STORAGE}. Returning empty.")
            return {}
        except Exception as e:
            print(f"Error loading chats from {CHAT_STORAGE}: {e}")
            return {}
    return {}

def save_chats(chats):
    """Saves the chat histories to the JSON storage file."""
    try:
        with open(CHAT_STORAGE, "w", encoding='utf-8') as f:
            json.dump(chats, f, indent=4)
    except Exception as e:
        print(f"Error saving chats to {CHAT_STORAGE}: {e}")

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
        "Accept": "application/json" # Explicitly ask for JSON
    }
    models_url = f"{CHUTES_API_URL}/models"
    print(f"--- Fetching models from Chutes AI: {models_url} ---")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(models_url, headers=headers, timeout=15) as response:
                response.raise_for_status() # Raise error for bad status codes
                data = await response.json()
                # Assuming the structure is like OpenAI's: {'data': [{'id': 'model-name', ...}, ...]}
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                    CHUTES_MODELS_CACHE = [model.get('id') for model in data['data'] if model.get('id')]
                    print(f"--- Found {len(CHUTES_MODELS_CACHE)} models from Chutes AI: {CHUTES_MODELS_CACHE[:5]}... ---")
                else:
                    print(f"--- Unexpected format received from Chutes AI /models endpoint: {data} ---")
                    CHUTES_MODELS_CACHE = []

    except aiohttp.ClientError as e:
        print(f"--- Error fetching models from Chutes AI (ClientError): {e} ---")
        CHUTES_MODELS_CACHE = []
    except asyncio.TimeoutError:
        print("--- Error fetching models from Chutes AI: Request timed out. ---")
        CHUTES_MODELS_CACHE = []
    except json.JSONDecodeError as e:
        print(f"--- Error decoding JSON response from Chutes AI /models: {e} ---")
        CHUTES_MODELS_CACHE = []
    except Exception as e:
        print(f"--- Unexpected error fetching models from Chutes AI: {e} ---")

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
                # Assuming structure {'data': [{'id': 'model-name', ...}, ...]}
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                    GROQ_MODELS_CACHE = [model.get('id') for model in data['data'] if model.get('id')]
                    print(f"--- Found {len(GROQ_MODELS_CACHE)} models from Groq API: {GROQ_MODELS_CACHE[:5]}... ---")
                else:
                    print(f"--- Unexpected format received from Groq API /models endpoint: {data} ---")
                    GROQ_MODELS_CACHE = []

    except aiohttp.ClientError as e:
        print(f"--- Error fetching models from Groq API (ClientError): {e} ---")
        GROQ_MODELS_CACHE = []
    except asyncio.TimeoutError:
        print("--- Error fetching models from Groq API: Request timed out. ---")
        GROQ_MODELS_CACHE = []
    except json.JSONDecodeError as e:
        print(f"--- Error decoding JSON response from Groq API /models: {e} ---")
        GROQ_MODELS_CACHE = []
    except Exception as e:
        print(f"--- Unexpected error fetching models from Groq API: {e} ---")
        GROQ_MODELS_CACHE = []

# --- End Groq API Integration ---

# --- End Chutes AI Integration ---

# --- Global variable to store scraped performance data ---
PROVIDER_PERFORMANCE_CACHE = [] # Initialize cache at module level

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
# --- End Fetch External Models ---

app = Flask(__name__) # Flask automatically looks for 'static' folder
app.secret_key = "your-secret-key" # Replace with a strong secret key
app.config["SESSION_TYPE"] = "filesystem"
# Ensure the session directory exists
SESSION_DIR = './flask_session' # Relative to app.py location
if not os.path.exists(SESSION_DIR):
    os.makedirs(SESSION_DIR)
app.config["SESSION_FILE_DIR"] = SESSION_DIR
Session(app)

# Function to dynamically get models and provider counts, sorted by priority
def get_available_models_with_provider_counts():
    """
    Dynamically retrieves available models, combines with performance data,
    and sorts them based on a predefined priority list and response time.
    Returns:
        list: A sorted list of tuples: [(model_name, provider_count), ...]
        dict: A dictionary mapping model names to a list of their provider identifiers (classes or strings).
        dict: A map of lowercase provider names to provider class objects.
    """
    available_models_dict = {}
    model_provider_info = {}
    provider_class_map = {} # Initialize map
    try:
        print("--- Fetching available models and providers --- ")
        if ProviderUtils:
            provider_classes = ProviderUtils.convert.values()
            provider_class_map = {prov.__name__.lower(): prov for prov in provider_classes}
        else:
            print("--- ProviderUtils not available, cannot map provider names. --- ")
            # Proceed without class map if needed, direct calls might fail

        for model in ModelUtils.convert.values():
            working_providers_list = []
            if isinstance(model.best_provider, IterListProvider):
                working_providers_list = [p for p in model.best_provider.providers if p.working]
            elif model.best_provider is not None and model.best_provider.working:
                working_providers_list = [model.best_provider]

            if working_providers_list:
                available_models_dict[model.name] = len(working_providers_list)
                model_provider_info[model.name] = working_providers_list
        print(f"--- Finished fetching g4f models. Found {len(available_models_dict)} models with working providers. ---")
    except Exception as e:
        print(f"Error dynamically loading g4f models: {e}")

    # --- Integrate external/direct providers --- 
    # Chutes AI
    print(f"--- Integrating Chutes models ({len(CHUTES_MODELS_CACHE)} found)... ---")
    for chutes_model_name in CHUTES_MODELS_CACHE:
        provider_id = "ChutesAIProvider"
        if chutes_model_name not in available_models_dict:
            available_models_dict[chutes_model_name] = 1
            model_provider_info[chutes_model_name] = [provider_id]
        elif provider_id not in model_provider_info[chutes_model_name]:
            if isinstance(model_provider_info[chutes_model_name], list):
                model_provider_info[chutes_model_name].append(provider_id)
            else: # Convert single provider to list
                model_provider_info[chutes_model_name] = [model_provider_info[chutes_model_name], provider_id]
            available_models_dict[chutes_model_name] += 1

    # Groq API
    print(f"--- Integrating Groq models ({len(GROQ_MODELS_CACHE)} found)... ---")
    for groq_model_name in GROQ_MODELS_CACHE:
        provider_id = "GroqAPIProvider"
        # Check if the *actual Groq model ID* is already a key
        if groq_model_name not in available_models_dict:
            available_models_dict[groq_model_name] = 1
            model_provider_info[groq_model_name] = [provider_id]
        # Also check if the *user-facing name* might exist from g4f and add Groq as provider
        # Example: user selects "llama-4-scout", g4f has it, Groq has "meta-llama/llama-4-scout..."
        # We need to handle this potential mismatch during provider selection, not necessarily here.
        # For now, just add the explicit Groq model ID if new.
        # Existing logic for adding provider to existing model seems okay.
        elif provider_id not in model_provider_info[groq_model_name]:
             if isinstance(model_provider_info[groq_model_name], list):
                 model_provider_info[groq_model_name].append(provider_id)
             else:
                 model_provider_info[groq_model_name] = [model_provider_info[groq_model_name], provider_id]
             available_models_dict[groq_model_name] += 1

    # Cerebras API
    print(f"--- Integrating Cerebras models ({len(CEREBRAS_TARGET_MODELS)} specified)... ---")
    for cerebras_model_name in CEREBRAS_TARGET_MODELS:
        provider_id = "CerebrasAPIProvider"
        if cerebras_model_name not in available_models_dict:
            available_models_dict[cerebras_model_name] = 1
            model_provider_info[cerebras_model_name] = [provider_id]
        elif provider_id not in model_provider_info[cerebras_model_name]:
             if isinstance(model_provider_info[cerebras_model_name], list):
                 model_provider_info[cerebras_model_name].append(provider_id)
             else:
                 model_provider_info[cerebras_model_name] = [model_provider_info[cerebras_model_name], provider_id]
             available_models_dict[cerebras_model_name] += 1

    # Google AI Studio
    print(f"--- Integrating Google AI Studio models ({len(GOOGLE_TARGET_MODELS)} specified)... ---")
    for google_model_name in GOOGLE_TARGET_MODELS:
        provider_id = "GoogleAIProvider"
        if google_model_name not in available_models_dict:
            available_models_dict[google_model_name] = 1
            model_provider_info[google_model_name] = [provider_id]
        elif provider_id not in model_provider_info[google_model_name]:
             if isinstance(model_provider_info[google_model_name], list):
                 model_provider_info[google_model_name].append(provider_id)
             else:
                 model_provider_info[google_model_name] = [model_provider_info[google_model_name], provider_id]
             available_models_dict[google_model_name] += 1
    # --- End Integrate --- 

    # --- New Sorting Logic --- 
    print("--- Applying custom sorting based on user priority and performance ---")

    # Define the user's explicit priority list (map to internal names if needed)
    # User table order: Llama 3.3 70B (cerebras), Llama 4 Scout (cerebras/groq), Llama 4 Maverick (groq), QwQ-32B (groq), Gemini 2.5 Flash (google), Gemini 2.5 Pro (google)
    user_priority_sequence = [
        # Mapped internal names based on likely targets/caches:
        "llama-3.3-70b",           # Corresponds to Cerebras Llama 3.3 70B
        "llama-4-scout",           # Corresponds to Cerebras/Groq Llama 4 Scout
        "meta-llama/llama-4-maverick-17b-128e-instruct", # Specific Groq ID for Llama 4 Maverick
        "qwen-qwq-32b",            # Specific Groq ID for QwQ-32B
        "Gemini 2.5 Flash",        # Corrected name
        "Gemini 2.5 Pro"           # Corrected name
    ]
    # Filter the priority sequence to only include models actually available
    filtered_priority_sequence = [name for name in user_priority_sequence if name in available_models_dict]

    final_display_list = []
    processed_models = set()

    # 1. Add models from the filtered user priority sequence
    print(f"--- Adding user priority models: {filtered_priority_sequence} ---")
    for model_name in filtered_priority_sequence:
        if model_name in available_models_dict:
            # Fetch performance for priority models
            best_response_time = float('inf')
            best_intel_index = 0
            found_perf_match = False # Debug flag
            display_name_lower = MODEL_DISPLAY_NAME_MAP.get(model_name, model_name).lower()
            for entry in PROVIDER_PERFORMANCE_CACHE:
                scraped_model_lower = entry.get('model_name_scraped', '').lower()
                model_name_lower = model_name.lower()
                # Check internal name OR display name against scraped name
                match = (model_name_lower == scraped_model_lower or
                         model_name_lower in scraped_model_lower or
                         scraped_model_lower in model_name_lower or
                         model_name_lower.replace('-',' ') == scraped_model_lower or
                         scraped_model_lower.replace('-',' ') == model_name_lower or
                         # Check display name match
                         display_name_lower == scraped_model_lower or
                         display_name_lower in scraped_model_lower or
                         scraped_model_lower in display_name_lower)

                if match:
                    found_perf_match = True # Mark as found
                    response_time = entry.get('response_time_s', float('inf'))
                    # Prioritize lower response time, then higher intelligence index
                    current_intel_index = entry.get('intelligence_index', 0)
                    if response_time < best_response_time:
                        best_response_time = response_time
                        best_intel_index = current_intel_index
                    elif response_time == best_response_time and current_intel_index > best_intel_index:
                        # If times are equal, prefer the one with higher intelligence
                        best_intel_index = current_intel_index
            # Debug print if no match found for a priority model
            if not found_perf_match:
                print(f"--- [Perf Debug] No performance match found in cache for priority model: {model_name} ---")

            final_display_list.append((model_name, available_models_dict[model_name], best_intel_index, best_response_time))
            processed_models.add(model_name)

    # 2. Gather remaining models and their best response times/intel
    remaining_models_with_perf = []
    print("--- Gathering remaining models and performance scores ---")
    for model_name, count in available_models_dict.items():
        if model_name not in processed_models:
            best_response_time = float('inf')
            best_intel_index = 0
            display_name_lower = MODEL_DISPLAY_NAME_MAP.get(model_name, model_name).lower()
            # Search performance cache for this model
            for entry in PROVIDER_PERFORMANCE_CACHE:
                scraped_model_lower = entry.get('model_name_scraped', '').lower()
                model_name_lower = model_name.lower()
                # Use the same enhanced matching here, including display name
                match = (model_name_lower == scraped_model_lower or
                         model_name_lower in scraped_model_lower or
                         scraped_model_lower in model_name_lower or
                         model_name_lower.replace('-',' ') == scraped_model_lower or
                         scraped_model_lower.replace('-',' ') == model_name_lower or
                         # Check display name match
                         display_name_lower == scraped_model_lower or
                         display_name_lower in scraped_model_lower or
                         scraped_model_lower in display_name_lower)

                if match:
                    response_time = entry.get('response_time_s', float('inf'))
                    current_intel_index = entry.get('intelligence_index', 0)
                    if response_time < best_response_time:
                        best_response_time = response_time
                        best_intel_index = current_intel_index
                    elif response_time == best_response_time and current_intel_index > best_intel_index:
                        best_intel_index = current_intel_index
            remaining_models_with_perf.append((model_name, count, best_intel_index, best_response_time))

    # 3. Sort remaining models by response time (ascending)
    remaining_models_with_perf.sort(key=lambda item: item[3]) # Sort by response_time (index 3)
    print(f"--- Sorted {len(remaining_models_with_perf)} remaining models by response time. ---")

    # 4. Append sorted remaining models to the final list
    for model_data in remaining_models_with_perf:
        final_display_list.append(model_data) # Append tuple with perf data

    print(f"--- Final sorted model list contains {len(final_display_list)} models. Top 5 (with perf): {final_display_list[:5]} ---")
    # Return list including performance data
    return final_display_list, model_provider_info, provider_class_map
    # --- End New Sorting Logic ---

@app.route('/', methods=['GET', 'POST'])
def index():
    chats = load_chats()
    # Get available models dynamically, sorted by new logic, plus provider info/map
    # Now includes performance data: [(name, count, intel_idx, resp_time), ...]
    available_models_sorted_list, model_provider_info, provider_class_map = get_available_models_with_provider_counts()
    available_model_names = {name for name, count, _, _ in available_models_sorted_list} # Unpack name only

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
                attempted_direct_providers = set() # Keep track of direct Cerebras/Groq attempts
                provider_errors = {} # Dictionary to store errors per provider
                max_retries = 2  # Maximum number of retries with web search
                retry_count = 0
                should_retry = True

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
                                "api_key": CEREBRAS_API_KEY
                            }
                            try:
                                print(f"--- Attempting Priority Provider: {provider_name_for_attempt} ({model_to_use_cerebras}) ---")
                                response_content = ChatCompletion.create(**current_args)
                                # Corrected error handling block:
                                if response_content and response_content.strip():
                                    content_str = response_content.strip()
                                    low = content_str.lower()
                                    if not (low.startswith("error:") or low.startswith("you have reached") or "challenge error" in low or "rate limit" in low):
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
                            current_args = {"api_key": GROQ_API_KEY, "provider": GROQ_PROVIDER_CLASS, "model": model_to_use_groq, "messages": api_messages}
                            try:
                                print(f"--- Attempting Priority Provider: {provider_name_for_attempt} ({model_to_use_groq}) via g4f ---")
                                response_content = ChatCompletion.create(**current_args)
                                # Corrected error handling block:
                                if response_content and response_content.strip():
                                    content_str = response_content.strip()
                                    low = content_str.lower()
                                    if not (low.startswith("error:") or low.startswith("you have reached") or "challenge error" in low or "rate limit" in low):
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
                                error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                                print(f"--- {error_msg} ---")
                                provider_errors[provider_name_for_attempt] = str(e)
                                response_content = None # Ensure reset before next iteration
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
                            model = genai.GenerativeModel(google_model_name)

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
                                        print(f"--- Provider {provider_name_for_attempt} succeeded! ---")
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
                            error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                            print(f"--- {error_msg} ---")
                            provider_errors[provider_name_for_attempt] = str(e)
                            response_content = None

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
                                    # "max_tokens": 1024,
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
                            error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                            print(f"--- {error_msg} ---")
                            provider_errors[provider_name_for_attempt] = str(e)
                            response_content = None # Ensure reset on failure

                    # Step 5: Try g4f Providers (Sorted by Scraped Performance or Default)
                    if response_content is None:
                        # Get the list of potential providers (classes or strings)
                        potential_providers = model_provider_info.get(selected_model_for_request, [])

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

                                current_args = {"provider": provider_class_to_use, "model": model_id_for_g4f_attempt, "messages": api_messages}

                                # Add API keys if required by the specific g4f provider (Optional - g4f might handle some internally)
                                # Example: if provider_class_to_use == SomeProviderRequiringKey: current_args["api_key"] = SOME_KEY

                                try:
                                    print(f"--- Attempting provider: {provider_name_for_attempt} with model {model_id_for_g4f_attempt} ---")
                                    response_content = ChatCompletion.create(**current_args)
                                    # ... (rest of try/except/check block remains the same) ...
                                    if response_content and response_content.strip():
                                        content_str = response_content.strip()
                                        low = content_str.lower()
                                        # Use a simpler error check for g4f providers
                                        if not low.startswith("error:") and "unable to fetch" not in low and "invalid key" not in low:
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
                                    error_msg = f"Provider {provider_name_for_attempt} failed: {e}"
                                    print(f"--- {error_msg} ---")
                                    provider_errors[provider_name_for_attempt] = str(e)
                                    response_content = None # Ensure reset before next iteration

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
                    # Add the successful assistant response
                    current_chat["history"].append({"role": "assistant", "content": response_content, "model": selected_model_for_request, "provider": provider_used_str, "timestamp": datetime.now().isoformat()})
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
    for model_name, provider_count, intel_index, resp_time in available_models_sorted_list:
        display_name = MODEL_DISPLAY_NAME_MAP.get(model_name, model_name) # Use mapped name or fallback to original
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
def delete_saved_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        print(f"--- Deleting chat: {chat_id} ---")
        del chats[chat_id]
        save_chats(chats)
    # If the deleted chat was the current one, clear it from session
    if session.get('current_chat') == chat_id:
        session.pop('current_chat', None)
    return redirect(url_for('saved_chats')) # Redirect back to saved_chats page


@app.route('/load_chat/<chat_id>')
def load_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        session['current_chat'] = chat_id
        return redirect(url_for('index')) # Load the chat in the main view
    else:
        # If chat ID is invalid, redirect to saved chats list
        return redirect(url_for('saved_chats'))

if __name__ == '__main__':
    # Model fetching is now done at module level

    # --- Performance data loading is now done at module level ---

    # Make sure host is accessible if running in container or VM
    print("--- [STARTUP] Starting Flask application... ---")
    app.run(host='0.0.0.0', port=5000, debug=True) # ENABLED debug mode for better logging/reloading