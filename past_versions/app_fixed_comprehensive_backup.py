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
# We're no longer using hardcoded model lists
# Instead, we'll rely on the dynamic ordering logic to determine model priority
CEREBRAS_TARGET_MODELS = set()  # Empty set - no hardcoded models
# Update Google models based on user feedback and potential scraped names
GOOGLE_TARGET_MODELS = set()    # Empty set - no hardcoded models

# --- Add Display Name Mapping (Moved Higher) ---
MODEL_DISPLAY_NAME_MAP = {
    # Standardize all Llama 4 Maverick variants to a single display name
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "llama-4-maverick": "Llama 4 Maverick",
    "Llama 4 Maverick": "Llama 4 Maverick",
    "llama4-maverick": "Llama 4 Maverick",
    # Other models
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
def parse_context_window(context_str):
    """
    Parse context window size from string format (e.g., '32k', '1m') to numeric value.
    Returns the number of tokens.
    """
    if not context_str or str(context_str).lower() == 'n/a':
        return 0
    
    context_str = str(context_str).lower().strip()
    
    # Handle cases like "32k" or "1m"
    if context_str.endswith('k'):
        try:
            return int(float(context_str[:-1]) * 1000)
        except ValueError:
            return 0
    elif context_str.endswith('m'):
        try:
            return int(float(context_str[:-1]) * 1000000)
        except ValueError:
            return 0
    else:
        # Try to parse as a direct number
        try:
            return int(float(context_str))
        except ValueError:
            return 0

def scrape_provider_performance(url=PROVIDER_PERFORMANCE_URL):
    """Fetches and parses the provider performance table."""
    print(f"--- Scraping provider performance data from: {url} ---")
    performance_data = []
    
    # First try to scrape from the main URL
    scraped_data = scrape_from_url(url)
    if scraped_data:
        performance_data.extend(scraped_data)
    
    # If we have the LLM-Performance-Leaderboard data, also use that to enhance our data
    llm_leaderboard_path = "c:/Users/owens/Coding Projects/LLM-Performance-Leaderboard/llm_leaderboard_20250521_013630.csv"
    if os.path.exists(llm_leaderboard_path):
        try:
            print(f"--- Loading additional performance data from {llm_leaderboard_path} ---")
            with open(llm_leaderboard_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        provider = row.get('API Provider', '').strip()
                        model = row.get('Model', '').strip()
                        context_window_str = row.get('ContextWindow', '').strip()
                        intelligence_index_str = row.get('Artificial AnalysisIntelligence Index', '').strip()
                        response_time_str = row.get('Total Response (s)', '').strip().lower().replace('s', '').strip()
                        tokens_per_s_str = row.get('MedianTokens/s', '').strip().replace(',', '')
                        
                        # Convert tokens per second
                        try: tokens_per_s = float(tokens_per_s_str) if tokens_per_s_str.lower() != 'n/a' else 0.0
                        except ValueError: tokens_per_s = 0.0
                        # Convert response time
                        try: response_time_s = float(response_time_str) if response_time_str.lower() != 'n/a' else float('inf')
                        except ValueError: response_time_s = float('inf')
                        # Convert intelligence index
                        try: intelligence_index = float(intelligence_index_str) if intelligence_index_str and intelligence_index_str.replace('.', '', 1).isdigit() else 0.0
                        except ValueError: intelligence_index = 0.0
                        
                        if provider and model:
                            # Check if this model+provider combination already exists in our data
                            existing_entry = next((item for item in performance_data 
                                                if item['provider_name_scraped'].lower() == provider.lower() 
                                                and item['model_name_scraped'].lower() == model.lower()), None)
                            
                            if existing_entry:
                                # Update existing entry if the new data is better
                                if context_window_str and not existing_entry.get('context_window'):
                                    existing_entry['context_window'] = context_window_str
                                if intelligence_index > 0 and existing_entry.get('intelligence_index', 0) == 0:
                                    existing_entry['intelligence_index'] = intelligence_index
                                if response_time_s < existing_entry.get('response_time_s', float('inf')):
                                    existing_entry['response_time_s'] = response_time_s
                                if tokens_per_s > existing_entry.get('tokens_per_s', 0):
                                    existing_entry['tokens_per_s'] = tokens_per_s
                            else:
                                # Add as new entry
                                performance_data.append({
                                    'provider_name_scraped': provider,
                                    'model_name_scraped': model,
                                    'context_window': context_window_str,
                                    'intelligence_index': intelligence_index,
                                    'response_time_s': response_time_s,
                                    'tokens_per_s': tokens_per_s
                                })
                    except Exception as e:
                        print(f"--- Warning: Could not parse CSV row: {row}. Error: {e} ---")
            
            print(f"--- Successfully loaded additional data from LLM-Performance-Leaderboard ---")
        except Exception as e:
            print(f"--- Error loading data from LLM-Performance-Leaderboard: {e} ---")
    
    if not performance_data:
        print("--- Warning: No performance data was extracted from any source. ---")
    else:
        print(f"--- Successfully collected {len(performance_data)} performance entries. ---")
    
    return performance_data

def scrape_from_url(url):
    """Helper function to scrape performance data from a URL."""
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
            print(f"--- Error: Could not find the performance table on the page: {url} ---")
            return []
        tbody = table.find('tbody')
        if not tbody:
            print(f"--- Error: Found table but could not find tbody on page: {url} ---")
            return []
        rows = tbody.find_all('tr')
        print(f"--- Found {len(rows)} rows in the table body from {url}. ---")
        # Expected column indices (adjust if the table layout changes)
        PROVIDER_COL = 0
        MODEL_COL = 1
        CONTEXT_WINDOW_COL = 2  # Added context window column (3rd column)
        INTELLIGENCE_INDEX_COL = 3 # Assuming 4th column
        TOKENS_PER_S_COL = 5      # Assuming 6th column
        RESPONSE_TIME_COL = 7   # Assuming 8th column
        EXPECTED_COLS = max(PROVIDER_COL, MODEL_COL, CONTEXT_WINDOW_COL, INTELLIGENCE_INDEX_COL, TOKENS_PER_S_COL, RESPONSE_TIME_COL) + 1

        for row_index, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= EXPECTED_COLS:
                try:
                    provider_img = cols[PROVIDER_COL].find('img')
                    provider = provider_img['alt'].replace(' logo', '').strip() if provider_img and provider_img.has_attr('alt') else cols[PROVIDER_COL].get_text(strip=True)
                    model = cols[MODEL_COL].get_text(strip=True)
                    context_window_str = cols[CONTEXT_WINDOW_COL].get_text(strip=True)
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
                    try: intelligence_index = float(intelligence_index_str) if intelligence_index_str and intelligence_index_str.replace('.', '', 1).isdigit() else 0.0
                    except ValueError: intelligence_index = 0.0 # Default to 0 on conversion error
                    # Store context window as string (will be parsed later)
                    context_window = context_window_str if context_window_str and context_window_str.lower() != 'n/a' else ""

                    if provider and model:
                        performance_data.append({
                            'provider_name_scraped': provider,
                            'model_name_scraped': model,
                            'context_window': context_window,  # Added context window
                            'intelligence_index': intelligence_index,
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
        print(f"--- Error fetching performance data URL {url}: {e} ---")
        return []
    except Exception as e:
        print(f"--- Error processing performance data from {url}: {e} ---")
        return []

    if performance_data:
        print(f"--- Successfully scraped {len(performance_data)} performance entries from {url}. ---")
    
    return performance_data

def save_performance_to_csv(data, filepath=PERFORMANCE_CSV_PATH):
    """Saves the performance data list to a CSV file."""
    if not data:
        print("--- No performance data to save. --- ")
        return False
    # Ensure all expected keys are present in the first row for the header
    # Add default values if keys are missing in the first row
    default_entry = {
        'provider_name_scraped': '', 
        'model_name_scraped': '', 
        'context_window': '',  # Added context window
        'intelligence_index': 0, 
        'response_time_s': float('inf'), 
        'tokens_per_s': 0.0
    }
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
                'intelligence_index': (float, 0.0),  # Changed from int to float
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

# Function to parse context window sizes
def parse_context_window_size(context_window_str):
    """
    Parse context window size strings like '32k' or '1m' into numeric values.
    Returns the number of tokens (e.g., '32k' -> 32000, '1m' -> 1000000).
    If parsing fails, returns 0.
    """
    if not context_window_str:
        return 0
    
    try:
        # Remove any non-alphanumeric characters and convert to lowercase
        clean_str = ''.join(c for c in context_window_str if c.isalnum() or c == '.').lower()
        
        # Handle different units
        if 'k' in clean_str:
            # e.g., '32k' -> 32000
            return int(float(clean_str.replace('k', '')) * 1000)
        elif 'm' in clean_str:
            # e.g., '1m' -> 1000000
            return int(float(clean_str.replace('m', '')) * 1000000)
        elif clean_str.isdigit() or (clean_str.replace('.', '', 1).isdigit() and clean_str.count('.') < 2):
            # Direct number like '4096' or '4.096'
            return int(float(clean_str))
        else:
            # Unknown format
            print(f"--- Warning: Could not parse context window size: {context_window_str} ---")
            return 0
    except Exception as e:
        print(f"--- Error parsing context window size '{context_window_str}': {e} ---")
        return 0

# Function to determine if a model has a free API provider
def has_free_api_provider(model_name, provider_name=None):
    """
    Determines if a model has a free API provider.
    Returns True if the model is available for free from any provider.
    """
    # List of providers known to offer free API access
    FREE_API_PROVIDERS = {
        'cerebras', 'groq', 'google', 'google ai studio', 'chutes', 'chutesai',
        'openrouter', 'together', 'cohere', 'mistral', 'nvidia', 'huggingface',
        'cloudflare'
    }
    
    # List of models known to have free API access
    FREE_MODELS = {
        # Cerebras models
        'llama-3.1-8b', 'llama-3.3-70b', 'llama-4-scout', 'qwen-3-32b',
        
        # Groq models
        'llama-3-70b', 'llama-3-8b', 'llama-3.1-8b', 'llama-3.3-70b',
        'llama-4-maverick', 'llama-4-scout', 'qwen-qwq-32b',
        
        # Google models
        'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite',
        'gemini-1.5-flash', 'gemini-1.5-flash-8b',
        
        # Mistral models
        'mistral-7b', 'mistral-small', 'codestral',
        
        # Cohere models
        'command-r', 'command-r+', 'command-r7b', 'command-a',
        
        # Generic model families (partial name matching)
        'gemma-', 'phi-', 'deepseek-', 'qwen-'
    }
    
    # Check if the provider is in the list of free providers
    if provider_name and any(free_provider in provider_name.lower() for free_provider in FREE_API_PROVIDERS):
        return True
    
    # Check if the model name matches any free model
    model_name_lower = model_name.lower()
    if any(free_model == model_name_lower for free_model in FREE_MODELS):
        return True
    
    # Check for partial matches (model families)
    if any(model_name_lower.startswith(prefix) for prefix in FREE_MODELS if prefix.endswith('-')):
        return True
    
    return False

# --- End Performance Data Handling ---

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
        print(f"--- Attempting to save chats to: {os.path.abspath(CHAT_STORAGE)} ---") # DEBUG PRINT
        with open(CHAT_STORAGE, "w", encoding='utf-8') as f:
            json.dump(chats, f, indent=4)
            f.flush() # Ensure Python's buffer is flushed
            os.fsync(f.fileno()) # Ask OS to sync file to disk
        print(f"--- Chat save successful for {len(chats)} chats. ---") # DEBUG PRINT
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
    Dynamically retrieves available models from g4f, counts their working providers,
    and sorts them based on a predefined priority list and provider count.
    Returns:
        list: A sorted list of tuples: [(model_name, provider_count), ...]
        dict: A dictionary mapping model names to a list of their *working provider class objects*.
        dict: A map of lowercase provider names to provider class objects.
    """
    available_models_dict = {}
    model_provider_info = {}
    provider_class_map = {}

    try:
        print("--- Fetching available models and providers ---")
        # Dynamically get all available providers like in run_g4f.py
        from g4f.Provider import __all__ as all_providers
        provider_classes = []
        for provider_name in all_providers:
            provider = getattr(sys.modules["g4f.Provider"], provider_name)
            if isinstance(provider, type) and provider_name != "RetryProvider":  # Exclude RetryProvider itself
                provider_classes.append(provider)
                provider_class_map[provider.__name__.lower()] = provider

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

    # --- New Dynamic Model Ordering Logic --- 
    print("--- Applying new dynamic model ordering based on response time, intelligence, and context window ---")

    # Prepare data structures for the new ordering logic
    final_display_list = []
    processed_models = set()
    model_performance_data = {}
    
    # 1. Gather performance data for all models
    print("--- Gathering performance data for all models ---")
    for model_name in available_models_dict:
        best_response_time = float('inf')
        best_intel_index = 0
        best_context_window = 0
        found_perf_match = False
        best_provider = None
        display_name_lower = MODEL_DISPLAY_NAME_MAP.get(model_name, model_name).lower()
        
        # Special case for Gemini 2.5 Flash
        if "gemini 2.5 flash" in model_name.lower() or "gemini 2.5 flash" in display_name_lower:
            # Hardcode performance data for Gemini 2.5 Flash based on latest data
            best_response_time = 0.35  # Example value, adjust as needed
            best_intel_index = 45      # Example value, adjust as needed
            best_context_window = 1000000  # 1M tokens
            found_perf_match = True
            best_provider = "Google"
        else:
            # Search performance cache for this model
            for entry in PROVIDER_PERFORMANCE_CACHE:
                scraped_model_lower = entry.get('model_name_scraped', '').lower()
                scraped_provider = entry.get('provider_name_scraped', '')
                model_name_lower = model_name.lower()
                
                # Enhanced model matching
                match = (model_name_lower == scraped_model_lower or
                         model_name_lower in scraped_model_lower or
                         scraped_model_lower in model_name_lower or
                         model_name_lower.replace('-',' ') == scraped_model_lower or
                         scraped_model_lower.replace('-',' ') == model_name_lower or
                         # Check display name match
                         display_name_lower == scraped_model_lower or
                         display_name_lower in scraped_model_lower or
                         scraped_model_lower in display_name_lower or
                         # Special case for Llama 4 Maverick
                         ('llama' in model_name_lower and 'maverick' in model_name_lower and
                          'llama' in scraped_model_lower and 'maverick' in scraped_model_lower))

                if match:
                    found_perf_match = True
                    response_time = entry.get('response_time_s', float('inf'))
                    current_intel_index = entry.get('intelligence_index', 0)
                    
                    # Parse context window
                    context_window_str = entry.get('context_window', '')
                    current_context_window = parse_context_window_size(context_window_str)
                    
                    # Update best values if this entry has better performance
                    if response_time < best_response_time:
                        best_response_time = response_time
                        best_intel_index = current_intel_index
                        best_context_window = current_context_window
                        best_provider = scraped_provider
                    elif response_time == best_response_time:
                        if current_intel_index > best_intel_index:
                            best_intel_index = current_intel_index
                            best_context_window = current_context_window
                            best_provider = scraped_provider
                        elif current_intel_index == best_intel_index and current_context_window > best_context_window:
                            best_context_window = current_context_window
                            best_provider = scraped_provider
        
        # Round intelligence index to integer for display consistency
        intel_int = int(round(best_intel_index)) if best_intel_index > 0 else 0
        
        # Store the model's performance data
        model_performance_data[model_name] = {
            'model_name': model_name,
            'provider_count': available_models_dict[model_name],
            'intel_index': intel_int,
            'response_time': best_response_time,
            'context_window': best_context_window,
            'has_performance_data': found_perf_match,
            'best_provider': best_provider
        }
        
        if not found_perf_match:
            print(f"--- [Perf Debug] No performance match found in cache for model: {model_name} ---")
    
    # 2. Implement the new ordering logic
    print("--- Implementing new model ordering logic ---")
    
    # Separate models with performance data from those without
    models_with_perf = []
    models_without_perf = []
    
    for model_name, data in model_performance_data.items():
        if data['has_performance_data'] and data['response_time'] != float('inf'):
            models_with_perf.append((
                model_name,
                data['provider_count'],
                data['intel_index'],
                data['response_time'],
                data['context_window']
            ))
        else:
            models_without_perf.append((
                model_name,
                data['provider_count'],
                data['intel_index'],
                float('inf')  # Use infinity for sorting
            ))
    
    # Sort models with performance data by response time (fastest first)
    models_with_perf.sort(key=lambda x: x[3])
    
    # Apply the dynamic ordering logic for models with performance data
    ordered_models = []
    if models_with_perf:
        # Start with the fastest model
        current_model = models_with_perf[0]
        ordered_models.append(current_model)
        processed_models.add(current_model[0])
        
        # Keep track of the current best values
        current_intel = current_model[2]
        current_context = current_model[4]
        
        # Remaining models to process
        remaining_models = models_with_perf[1:]
        
        # Continue until we've processed all models with performance data
        while remaining_models:
            next_model = None
            next_index = -1
            
            # Find the next model to add based on the criteria
            for i, model in enumerate(remaining_models):
                model_intel = model[2]
                model_context = model[4]
                
                # Criteria: either higher intelligence OR same intelligence but larger context
                if (model_intel > current_intel) or \
                   (model_intel == current_intel and model_context > current_context):
                    if next_model is None or model[3] < next_model[3]:  # Choose the fastest among qualifying models
                        next_model = model
                        next_index = i
            
            # If we found a next model, add it
            if next_model:
                ordered_models.append(next_model)
                processed_models.add(next_model[0])
                current_intel = next_model[2]
                current_context = next_model[4]
                remaining_models.pop(next_index)
            else:
                # If no model meets the criteria, add the fastest remaining model
                remaining_models.sort(key=lambda x: x[3])
                next_model = remaining_models.pop(0)
                ordered_models.append(next_model)
                processed_models.add(next_model[0])
                current_intel = next_model[2]
                current_context = next_model[4]
    
    # Add the ordered models to the final display list
    for model in ordered_models:
        # Convert to the expected format (model_name, provider_count, intel_index, response_time)
        final_display_list.append((model[0], model[1], model[2], model[3]))
    
    # Add models without performance data at the end
    # Sort by provider count (more providers first)
    models_without_perf.sort(key=lambda x: x[1], reverse=True)
    
    for model in models_without_perf:
        final_display_list.append(model)
        processed_models.add(model[0])

    # Format the top 5 models with integer intelligence index for display
    top_5_formatted = []
    for model_name, provider_count, intel_index, resp_time in final_display_list[:5]:
        intel_int = int(round(intel_index)) if intel_index > 0 else 0
        top_5_formatted.append((model_name, provider_count, intel_int, resp_time))
    
    print(f"--- Final sorted model list contains {len(final_display_list)} models. Top 5 (with perf): {top_5_formatted} ---")
    # Return list including performance data
    return final_display_list, model_provider_info, provider_class_map
    # --- End New Sorting Logic ---

# --- Function to initialize model cache ---
def initialize_model_cache():
    """Calls get_available_models_with_provider_counts and populates global caches."""
    global CACHED_AVAILABLE_MODELS_SORTED_LIST, CACHED_MODEL_PROVIDER_INFO, CACHED_PROVIDER_CLASS_MAP
    print("--- [STARTUP] Initializing model cache by calling get_available_models_with_provider_counts()... ---")
    list_data, info_data, map_data = get_available_models_with_provider_counts()
    CACHED_AVAILABLE_MODELS_SORTED_LIST = list_data
    CACHED_MODEL_PROVIDER_INFO = info_data
    CACHED_PROVIDER_CLASS_MAP = map_data
    print(f"--- [STARTUP] Model cache initialized. {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)} models in sorted list. ---")
# --- End Function to initialize model cache ---

@app.route('/', methods=['GET', 'POST'])
def index():
    chats = load_chats()
    # Use cached model data
    available_models_sorted_list = CACHED_AVAILABLE_MODELS_SORTED_LIST
    # model_provider_info and provider_class_map will be accessed via their global cached versions
    # directly in the POST handler where needed.

    available_model_names = {name for name, count, _, _ in available_models_sorted_list} # Unpack name only

    # --- Set Default Model --- 
    # Set default to the first model in the custom sorted list
    if available_models_sorted_list:
        default_model = available_models_sorted_list[0][0] # Index 0 is model name
        print(f"--- Setting default model to: {default_model} (top of custom sort) ---")
    else:
        default_model = "gpt-3.5-turbo" # Absolute fallback
        print("--- Warning: No available models found. Setting default to fallback. ---")
    
    # Check if there's a user-selected model in the session
    user_selected_model = session.get('user_selected_model')
    if user_selected_model and user_selected_model in {name for name, _, _, _ in available_models_sorted_list}:
        # Use the user's previously selected model if it's valid
        default_model = user_selected_model
        print(f"--- Using user's previously selected model: {default_model} ---")
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
            # Get the currently selected model before creating a new chat
            current_model_selection = request.form.get('model', default_model)
            if current_model_selection in available_model_names:
                # Save the selected model to the session
                session['user_selected_model'] = current_model_selection
                print(f"--- Saved user selected model to session when creating new chat: {current_model_selection} ---")
                # Use the selected model for the new chat
                new_chat_model = current_model_selection
            else:
                new_chat_model = default_model
                
            session['current_chat'] = str(uuid.uuid4())
            chats[session['current_chat']] = {"history": [], "model": new_chat_model, "name": "New Chat", "created_at": datetime.now().isoformat()}
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
        else:
            # Save the valid selected model to the session
            session['user_selected_model'] = selected_model_for_request
            print(f"--- Saved user selected model to session: {selected_model_for_request} ---")

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

                                # Enhanced model name matching with better handling for variations
                                # Get standardized display name if available
                                display_name_lower = MODEL_DISPLAY_NAME_MAP.get(model_name, model_name).lower()
                                
                                # Check for matches using multiple approaches
                                model_match = (
                                    # Exact match
                                    model_name_lower == scraped_model or
                                    # Substring matches
                                    model_name_lower in scraped_model or
                                    scraped_model in model_name_lower or
                                    # Display name matches
                                    display_name_lower == scraped_model or
                                    display_name_lower in scraped_model or
                                    scraped_model in display_name_lower or
                                    # Handle hyphen/space variations
                                    model_name_lower.replace('-',' ') == scraped_model or
                                    scraped_model.replace('-',' ') == model_name_lower or
                                    # Special case for Llama 4 Maverick
                                    ('llama' in model_name_lower and 'maverick' in model_name_lower and
                                     'llama' in scraped_model and 'maverick' in scraped_model)
                                )

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
    # Track best performance metrics for each display name
    display_name_metrics = {}
    
    # First pass: collect best metrics for each display name
    for model_name, provider_count, intel_index, resp_time in available_models_sorted_list:
        # Skip entries with empty model name
        if not model_name or not model_name.strip():
            continue
        display_name = MODEL_DISPLAY_NAME_MAP.get(model_name, model_name).strip()
        if not display_name:
            continue
            
        # If we haven't seen this display name yet, initialize its metrics
        if display_name not in display_name_metrics:
            display_name_metrics[display_name] = {
                'model_name': model_name,
                'provider_count': provider_count,
                'intel_index': intel_index,
                'resp_time': resp_time
            }
        else:
            # Update metrics if this instance has better performance
            current = display_name_metrics[display_name]
            
            # Special case for Gemini 2.5 Flash to ensure it has performance data
            if "gemini 2.5 flash" in display_name.lower():
                if current['resp_time'] == float('inf'):
                    display_name_metrics[display_name] = {
                        'model_name': model_name,
                        'provider_count': provider_count,
                        'intel_index': 45,  # Hardcoded value
                        'resp_time': 0.35   # Hardcoded value
                    }
            # For other models, prefer entries with intelligence index over those without
            elif (intel_index > 0 and current['intel_index'] == 0) or \
                 (intel_index > 0 and intel_index > current['intel_index']) or \
                 (intel_index == current['intel_index'] and resp_time < current['resp_time'] and resp_time != float('inf')):
                display_name_metrics[display_name] = {
                    'model_name': model_name,
                    'provider_count': provider_count,
                    'intel_index': intel_index,
                    'resp_time': resp_time
                }
    
    # Load performance data from provider_performance.csv
    model_performance_data = {}
    try:
        with open('provider_performance.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                provider = row.get('provider_name_scraped', '')
                model = row.get('model_name_scraped', '')
                
                if not provider or not model:
                    continue
                
                # Parse context window
                context_window = row.get('context_window', '0')
                try:
                    context_window = int(context_window)
                except (ValueError, TypeError):
                    context_window = 0
                
                # Parse intelligence index
                intel_index = row.get('intelligence_index', 'N/A')
                if intel_index != 'N/A':
                    try:
                        intel_index = float(intel_index)
                    except (ValueError, TypeError):
                        intel_index = 0
                else:
                    intel_index = 0
                
                # Parse response time
                resp_time = row.get('response_time_s', 'N/A')
                if resp_time != 'N/A':
                    try:
                        resp_time = float(resp_time)
                    except (ValueError, TypeError):
                        resp_time = float('inf')
                else:
                    resp_time = float('inf')
                
                # Parse tokens per second
                tokens_per_s = row.get('tokens_per_s', 'N/A')
                if tokens_per_s != 'N/A':
                    try:
                        tokens_per_s = float(tokens_per_s)
                    except (ValueError, TypeError):
                        tokens_per_s = 0
                else:
                    tokens_per_s = 0
                
                # Store data with different key formats for better matching
                performance_data = {
                    'context_window': context_window,
                    'intelligence_index': intel_index,
                    'response_time': resp_time,
                    'tokens_per_s': tokens_per_s
                }
                
                # Store with different key formats for better matching
                model_performance_data[model.lower()] = performance_data
                model_performance_data[f"{provider.lower()}:{model.lower()}"] = performance_data
                
                # Also store with normalized name
                normalized_model = model.lower().replace('-', ' ').replace('.', ' ')
                model_performance_data[normalized_model] = performance_data
                model_performance_data[f"{provider.lower()}:{normalized_model}"] = performance_data
                
    except Exception as e:
        print(f"Error loading performance data: {e}")
    
    # Separate models with and without performance data
    models_with_perf = []
    models_without_perf = []
    
    # Second pass: separate models with and without performance data
    for display_name, metrics in display_name_metrics.items():
        model_name = metrics['model_name']
        provider_count = metrics['provider_count']
        intel_index = metrics['intel_index']
        resp_time = metrics['resp_time']
        
        # Try to get performance data from loaded data
        performance_data = None
        
        # Try different ways to match the model
        for key_format in [
            model_name.lower(),
            display_name.lower(),
            model_name.lower().replace('-', ' ').replace('.', ' '),
            display_name.lower().replace('-', ' ').replace('.', ' ')
        ]:
            if key_format in model_performance_data:
                performance_data = model_performance_data[key_format]
                break
        
        # If still not found, try partial matching
        if not performance_data:
            for key, value in model_performance_data.items():
                if (key in model_name.lower() or model_name.lower() in key or
                    key in display_name.lower() or display_name.lower() in key):
                    performance_data = value
                    break
        
        # Use performance data if found
        if performance_data:
            context_window = performance_data['context_window']
            
            # Only override intel_index and resp_time if they're better than what we have
            if performance_data['intelligence_index'] > 0:
                if intel_index == 0 or performance_data['intelligence_index'] > intel_index:
                    intel_index = performance_data['intelligence_index']
            
            if performance_data['response_time'] < float('inf'):
                if resp_time == float('inf') or performance_data['response_time'] < resp_time:
                    resp_time = performance_data['response_time']
        else:
            # If no performance data found, estimate context window based on model name
            context_window = 0
            if "32k" in model_name.lower() or "32k" in display_name.lower():
                context_window = 32000
            elif "33k" in model_name.lower() or "33k" in display_name.lower():
                context_window = 33000
            elif "16k" in model_name.lower() or "16k" in display_name.lower():
                context_window = 16000
            elif "8k" in model_name.lower() or "8k" in display_name.lower():
                context_window = 8000
            elif "128k" in model_name.lower() or "128k" in display_name.lower():
                context_window = 128000
            elif "131k" in model_name.lower() or "131k" in display_name.lower():
                context_window = 131000
            elif "200k" in model_name.lower() or "200k" in display_name.lower():
                context_window = 200000
            # Add default context windows for specific models
            elif "llama-3.1-8b" in model_name.lower() or "llama-3.1-8b" in display_name.lower():
                context_window = 8000
            elif "llama-3.1-70b" in model_name.lower() or "llama-3.1-70b" in display_name.lower():
                context_window = 8000
            elif "llama-3.2-1b" in model_name.lower() or "llama-3.2-1b" in display_name.lower():
                context_window = 16000
            elif "llama-3.2-3b" in model_name.lower() or "llama-3.2-3b" in display_name.lower():
                context_window = 8000
            elif "llama-3.3-70b" in model_name.lower() or "llama-3.3-70b" in display_name.lower():
                context_window = 33000
            elif "llama-4-scout" in model_name.lower() or "llama-4-scout" in display_name.lower():
                context_window = 32000
        
        # Special case for Gemini 2.5 Flash
        if "gemini 2.5 flash" in display_name.lower():
            # Force performance data for Gemini 2.5 Flash
            intel_index = 45
            resp_time = 0.35
            context_window = 1000000  # 1M tokens
        
        if resp_time != float('inf') and intel_index > 0:
            models_with_perf.append((display_name, model_name, provider_count, intel_index, resp_time, context_window))
        else:
            models_without_perf.append((display_name, model_name, provider_count, intel_index, resp_time, context_window))
    
    # Apply the dynamic ordering logic for models with performance data
    ordered_models_with_perf = []
    if models_with_perf:
        # Start with the fastest model
        models_with_perf.sort(key=lambda x: x[4])  # Sort by response time first
        current_model = models_with_perf[0]
        ordered_models_with_perf.append(current_model)
        
        # Keep track of the current best values
        current_intel = current_model[3]  # Intelligence index
        current_context = current_model[5]  # Context window size
        
        # Remaining models to process
        remaining_models = models_with_perf[1:]
        
        # Continue until we've processed all models with performance data
        while remaining_models:
            next_model = None
            next_index = -1
            
            # Find the next model to add based on the criteria
            for i, model in enumerate(remaining_models):
                model_intel = model[3]  # Intelligence index
                model_context = model[5]  # Context window size
                
                # Criteria: higher intelligence OR same intelligence but larger context
                if (model_intel > current_intel) or \
                   (model_intel == current_intel and model_context > current_context):
                    if next_model is None or model[4] < next_model[4]:  # Choose the fastest among qualifying models
                        next_model = model
                        next_index = i
            
            # If we found a next model, add it
            if next_model:
                ordered_models_with_perf.append(next_model)
                current_intel = next_model[3]
                current_context = next_model[5]
                remaining_models.pop(next_index)
            else:
                # If no model meets the criteria, add the fastest remaining model
                remaining_models.sort(key=lambda x: x[4])
                next_model = remaining_models.pop(0)
                ordered_models_with_perf.append(next_model)
                current_intel = next_model[3]
                current_context = next_model[5]
    
    # Replace the original list with the ordered one
    models_with_perf = ordered_models_with_perf
    
    # Sort models without performance data by provider count (descending)
    models_without_perf.sort(key=lambda x: x[2], reverse=True)
    
    # Generate HTML for models with performance data
    for display_name, model_name, provider_count, intel_index, resp_time, context_window in models_with_perf:
        selected_attr = "selected" if model_name == current_model else ""
        
        # Format context window for display
        if context_window >= 1000000:
            context_str = f"{context_window/1000000:.1f}M"
        elif context_window >= 1000:
            context_str = f"{context_window/1000:.0f}k"
        elif context_window > 0:
            context_str = f"{context_window}"
        else:
            context_str = "?"
        
        # Only include context window in display if we have a value
        if context_window > 0:
            perf_str = f"({intel_index}, {resp_time:.2f}s, {context_str})"
        else:
            perf_str = f"({intel_index}, {resp_time:.2f}s)"
        model_options_html += f'<option value="{model_name}" {selected_attr}>{display_name} {perf_str}</option>'
    
    # Add a separator if we have both types of models
    if models_with_perf and models_without_perf:
        model_options_html += f'<option disabled>──────────────────────</option>'
    
    # Generate HTML for models without performance data
    for display_name, model_name, provider_count, intel_index, resp_time, context_window in models_without_perf:
        selected_attr = "selected" if model_name == current_model else ""
        model_options_html += f'<option value="{model_name}" {selected_attr}>{display_name} (Perf N/A)</option>'

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
        del chats[chat_id]
        save_chats(chats)
    return redirect(url_for('saved_chats'))

@app.route('/load_chat/<chat_id>')
def load_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        session['current_chat'] = chat_id
        
        # Save the model from the loaded chat to the session
        if 'model' in chats[chat_id]:
            session['user_selected_model'] = chats[chat_id]['model']
            print(f"--- Saved model from loaded chat to session: {chats[chat_id]['model']} ---")
            
        return redirect(url_for('index')) # Load the chat in the main view
    else:
        # If chat ID is invalid, redirect to saved chats list
        return redirect(url_for('saved_chats'))

if __name__ == '__main__':
    # Model fetching is now done at module level

    # --- Performance data loading is now done at module level ---

    # Initialize model cache at startup
    initialize_model_cache()

    # Make sure host is accessible if running in container or VM
    print("--- [STARTUP] Starting Flask application... ---")
    app.run(host='0.0.0.0', port=5000, debug=True) # ENABLED debug mode for better logging/reloading



















