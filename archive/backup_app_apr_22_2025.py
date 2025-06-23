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


# Load environment variables from .env file
load_dotenv()

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from flask import Flask, request, session, redirect, url_for

# --- Configuration ---
# Path to store chat histories
CHAT_STORAGE = "../chats.json" # Adjusted path relative to flask_app/app.py
# API Keys loaded from .env
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
GROQ_TARGET_MODELS = set(GROQ_MODEL_NAME_MAP.keys()) | {"gemma2-9b-it", "llama3-8b-8192", "llama3-70b-8192"}
CEREBRAS_TARGET_MODELS = {"llama-3.1-8b", "llama-3.3-70b"} # User-facing names

# Provider names as they might appear in the scraped data (case-insensitive check recommended)
SCRAPED_PROVIDER_NAME_CEREBRAS = "Cerebras"
SCRAPED_PROVIDER_NAME_GROQ = "Groq"

# Performance Data Config
PROVIDER_PERFORMANCE_URL = "https://artificialanalysis.ai/leaderboards/providers"
PERFORMANCE_CSV_PATH = "./provider_performance.csv" # Path relative to flask_app
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
        for row_index, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= 8:
                try:
                    provider_img = cols[0].find('img')
                    provider = provider_img['alt'].replace(' logo', '').strip() if provider_img and provider_img.has_attr('alt') else cols[0].get_text(strip=True)
                    model = cols[1].get_text(strip=True)
                    tokens_per_s_str = cols[5].get_text(strip=True)
                    response_time_str = cols[7].get_text(strip=True).lower().replace('s', '').strip()

                    try:
                        tokens_per_s = float(tokens_per_s_str) if tokens_per_s_str.lower() != 'n/a' else 0.0
                    except ValueError: tokens_per_s = 0.0
                    try:
                        response_time_s = float(response_time_str) if response_time_str.lower() != 'n/a' else float('inf')
                    except ValueError: response_time_s = float('inf')

                    if provider and model:
                        performance_data.append({
                            'provider_name_scraped': provider,
                            'model_name_scraped': model,
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
        print(f"--- Successfully scraped {len(performance_data)} performance entries. ---")
    return performance_data

def save_performance_to_csv(data, filepath=PERFORMANCE_CSV_PATH):
    """Saves the performance data list to a CSV file."""
    if not data:
        print("--- No performance data to save. ---")
        return False
    header = data[0].keys()
    print(f"--- Saving {len(data)} performance entries to {filepath} ---")
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)
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
            for row in reader:
                try:
                    # Ensure correct types are loaded
                    row['response_time_s'] = float(row.get('response_time_s', 'inf'))
                    row['tokens_per_s'] = float(row.get('tokens_per_s', '0.0'))
                    data.append(row)
                except (ValueError, TypeError) as e:
                    print(f"--- Warning: Skipping row due to conversion error in CSV: {row}. Error: {e} ---")
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

# --- Global variable to store scraped performance data ---
PROVIDER_PERFORMANCE_CACHE = [] # Initialize cache at module level

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
    provider_class_map = {} # Initialize map
    try: # Restore the main try block for the function
        print("--- Fetching available models and providers ---")
        # Get provider classes using ProviderUtils if available
        if ProviderUtils:
            provider_classes = ProviderUtils.convert.values()
            provider_class_map = {prov.__name__.lower(): prov for prov in provider_classes}
        else:
            print("--- ProviderUtils not available, cannot map provider names. ---")
            return [], {}, {}

        for model in ModelUtils.convert.values():
            working_providers_list = []
            if isinstance(model.best_provider, IterListProvider):
                # Filter providers that are actually working
                working_providers_list = [p for p in model.best_provider.providers if p.working]
            elif model.best_provider is not None and model.best_provider.working:
                working_providers_list = [model.best_provider]

            if working_providers_list: # Only add models with at least one working provider
                available_models_dict[model.name] = len(working_providers_list)
                model_provider_info[model.name] = working_providers_list # Store the list of working provider classes
        print(f"--- Finished fetching models. Found {len(available_models_dict)} models with working providers. ---")
    except Exception as e: # Correctly associate except with the try
        print(f"Error dynamically loading models: {e}")
        return [], {}, {} # Return empty dicts on error

    # Define priority models (Example: prioritize o3-mini if available)
    priority_order = ["o3-mini", "o1", "deepseek-r1", "deepseek-v3"] # Added o3-mini
    priority_models_sorted = []
    other_models = {}
    for name, count in available_models_dict.items():
        if name in priority_order:
            priority_models_sorted.append((priority_order.index(name), name, count))
        else: other_models[name] = count
    priority_models_sorted.sort()
    final_priority_list = [(name, count) for _, name, count in priority_models_sorted]
    # Sort other models primarily by provider count (descending), then alphabetically
    other_models_sorted = sorted(other_models.items(), key=lambda item: (-item[1], item[0]))
    return final_priority_list + other_models_sorted, model_provider_info, provider_class_map


@app.route('/', methods=['GET', 'POST'])
def index():
    chats = load_chats()
    # Get available models dynamically and sorted, plus provider info and class map
    available_models_sorted_list, model_provider_info, provider_class_map = get_available_models_with_provider_counts()
    available_model_names = {name for name, count in available_models_sorted_list}

    # --- Set Default Model ---
    # Prioritize o3-mini, then o1, then the first in the sorted list, then fallback
    if "o3-mini" in available_model_names:
        default_model = "o3-mini"
    elif "o1" in available_model_names:
        default_model = "o3-mini"
    elif available_models_sorted_list:
        default_model = available_models_sorted_list[0][0]
    else:
        default_model = "gpt-3.5-turbo" # Absolute fallback
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
                print(f"--- Web search explicitly enabled for: {prompt_to_use[:50]}... ---")
                search_results_str = perform_web_search(prompt_to_use)
            elif web_search_mode == 'smart':
                # Simple keyword check for smart search trigger
                smart_search_keywords = ["latest", "recent", "today", "current news", "what happened", "who is"]
                if any(keyword in prompt_to_use.lower() for keyword in smart_search_keywords):
                    print(f"--- Smart search triggered for: {prompt_to_use[:50]}... ---")
                    search_results_str = perform_web_search(prompt_to_use)
                else:
                    print(f"--- Smart search not triggered for prompt. ---")
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

                # Helper function to get performance score (lower is better, e.g., response time)
                def get_scraped_performance_metric(provider_class_name, model_name):
                    best_score = float('inf')
                    provider_name_lower = provider_class_name.lower()
                    model_name_lower = model_name.lower()
                    # Search cache for matching provider name (case-insensitive) and model name
                    for entry in PROVIDER_PERFORMANCE_CACHE:
                        scraped_provider = entry.get('provider_name_scraped', '').lower()
                        scraped_model = entry.get('model_name_scraped', '').lower()
                        # Check if the g4f provider name is part of the scraped name, or vice-versa,
                        # or if it's a known direct match (like Groq)
                        provider_match = (provider_name_lower in scraped_provider or
                                          scraped_provider in provider_name_lower or
                                          (provider_name_lower == 'groq' and scraped_provider == 'groq')) # Add more specific matches if needed

                        # Basic model name matching (can be improved)
                        model_match = (model_name_lower == scraped_model)

                        if provider_match and model_match:
                            score = entry.get('response_time_s', float('inf'))
                            if score < best_score:
                                best_score = score
                    return best_score

                # Step 1: Try Direct Cerebras
                if selected_model_for_request in CEREBRAS_TARGET_MODELS and CEREBRAS_API_KEY:
                    provider_name_for_attempt = "Cerebras (Direct)"
                    attempted_direct_providers.add("cerebras")
                    try:
                        print(f"--- Attempting provider: {provider_name_for_attempt} ---")
                        # Assuming g4f can handle direct provider specification or specific args
                        current_args = {"api_key": CEREBRAS_API_KEY, "model": selected_model_for_request, "messages": api_messages, "provider": "Cerebras"} # Hypothetical
                        response_content = ChatCompletion.create(**current_args)
                        # Check if response is non-empty and not an error message
                        if response_content and response_content.strip():
                            content_str = response_content.strip()
                            low = content_str.lower()
                            if not (low.startswith("error:") or low.startswith("you have reached") or "challenge error" in low):
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
                        response_content = None # Ensure reset on failure

                # Step 2: Try Direct Groq (if Cerebras not attempted or failed)
                if response_content is None and selected_model_for_request in GROQ_TARGET_MODELS and GROQ_API_KEY and GROQ_PROVIDER_CLASS:
                    provider_name_for_attempt = "Groq (Direct)"
                    attempted_direct_providers.add("groq")
                    groq_model_name = GROQ_MODEL_NAME_MAP.get(selected_model_for_request, selected_model_for_request)
                    current_args = {"api_key": GROQ_API_KEY, "provider": GROQ_PROVIDER_CLASS, "model": groq_model_name, "messages": api_messages}
                    try:
                        print(f"--- Attempting provider: {provider_name_for_attempt} ---")
                        response_content = ChatCompletion.create(**current_args)
                        # Check if response is non-empty and not an error message
                        if response_content and response_content.strip():
                            content_str = response_content.strip()
                            low = content_str.lower()
                            if not (low.startswith("error:") or low.startswith("you have reached") or "challenge error" in low):
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

                # Step 3: Try g4f Providers (Sorted by Scraped Performance)
                if response_content is None:
                    default_providers = model_provider_info.get(selected_model_for_request, [])
                    if not default_providers:
                        print(f"--- No default g4f providers found for model {selected_model_for_request}. Falling back to generic g4f provider. ---")
                        try:
                            response_content = ChatCompletion.create(model=selected_model_for_request, messages=api_messages)
                            provider_used_str = "Generic g4f"
                            print(f"--- Generic g4f fallback succeeded for model {selected_model_for_request} ---")
                        except Exception as e:
                            error_msg = f"Generic g4f fallback failed: {e}"
                            print(f"--- {error_msg} ---")
                            provider_errors['Generic g4f fallback'] = str(e)
                            response_content = None
                    else:
                        # Sort providers based on scraped performance (lower response time is better)
                        default_providers_sorted = sorted(
                            default_providers,
                            key=lambda p_class: get_scraped_performance_metric(p_class.__name__, selected_model_for_request)
                        )
                        print(f"--- g4f providers sorted by performance: {[(p.__name__, get_scraped_performance_metric(p.__name__, selected_model_for_request)) for p in default_providers_sorted]} ---")

                        for provider_class in default_providers_sorted:
                            provider_name_lower = provider_class.__name__.lower()
                            is_groq = provider_class == GROQ_PROVIDER_CLASS
                            is_cerebras = SCRAPED_PROVIDER_NAME_CEREBRAS.lower() in provider_name_lower

                            # Prepare base arguments
                            current_args = {"provider": provider_class, "model": selected_model_for_request, "messages": api_messages}
                            provider_name_for_attempt = f"{provider_class.__name__} (g4f List)"

                            # Internal Priority Check: Add API key if it's Groq/Cerebras and not attempted directly
                            if is_groq and GROQ_API_KEY and "groq" not in attempted_direct_providers:
                                current_args["api_key"] = GROQ_API_KEY
                                provider_name_for_attempt = f"{provider_class.__name__} (g4f Internal Priority)"
                                print(f"--- Applying internal priority for Groq (g4f list) ---")
                            elif is_cerebras and CEREBRAS_API_KEY and "cerebras" not in attempted_direct_providers:
                                current_args["api_key"] = CEREBRAS_API_KEY
                                provider_name_for_attempt = f"{provider_class.__name__} (g4f Internal Priority)"
                                print(f"--- Applying internal priority for Cerebras (g4f list) ---")

                            # Attempt the provider
                            try:
                                print(f"--- Attempting provider: {provider_name_for_attempt} ---")
                                response_content = ChatCompletion.create(**current_args)
                                # Check if response is non-empty and not an error message
                                if response_content and response_content.strip():
                                    content_str = response_content.strip()
                                    low = content_str.lower()
                                    if not (low.startswith("error:") or low.startswith("you have reached") or "challenge error" in low):
                                        provider_used_str = provider_name_for_attempt
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

            save_chats(chats)
            return redirect(url_for('index'))
        # Handle case where regeneration was requested but no prompt could be determined
        elif is_regeneration:
            print("--- Regeneration requested but no previous user prompt found. ---")
            return redirect(url_for('index'))
        # Handle empty prompt submission (do nothing, just reload)
        else:
            print("--- Empty prompt submitted. ---")
            return redirect(url_for('index'))


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

    model_options_html = ''.join([f'<option value="{model_name}" {"selected" if model_name == current_model else ""}>{model_name} ({provider_count} providers)</option>' for model_name, provider_count in available_models_sorted_list])

    # Determine checked state for web search (default to smart)
    # This assumes no session persistence for the mode yet.
    web_search_html = f'''
        <div style="margin-bottom: 10px;">
            <label style="margin-right: 10px; font-weight: bold;">Web Search:</label>
            <input type="radio" id="ws_off" name="web_search_mode" value="off"> <label for="ws_off" style="margin-right: 5px;">Off</label>
            <input type="radio" id="ws_on" name="web_search_mode" value="on"> <label for="ws_on" style="margin-right: 5px;">On</label>
            <input type="radio" id="ws_smart" name="web_search_mode" value="smart" checked> <label for="ws_smart">Smart</label>
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
    # --- Load or Scrape performance data on startup ---
    print("--- Initializing: Loading/Scraping provider performance data ---")
    PROVIDER_PERFORMANCE_CACHE = load_performance_from_csv(PERFORMANCE_CSV_PATH)
    if not PROVIDER_PERFORMANCE_CACHE:
        print("--- No cache found or loading failed, attempting to scrape... ---")
        scraped_data = scrape_provider_performance()
        if scraped_data:
            PROVIDER_PERFORMANCE_CACHE = scraped_data
            save_performance_to_csv(PROVIDER_PERFORMANCE_CACHE, PERFORMANCE_CSV_PATH) # Save if scrape succeeds
        else:
            print("--- Warning: Scraping also failed. Performance prioritization disabled. ---")
            PROVIDER_PERFORMANCE_CACHE = [] # Ensure it's an empty list if both fail
    else:
         print(f"--- Initialization complete using cached data. Found {len(PROVIDER_PERFORMANCE_CACHE)} performance entries. ---")
    # --- End data loading ---

    # Make sure host is accessible if running in container or VM
    app.run(host='0.0.0.0', port=5000, debug=False) # Disabled debug mode for lower idle resource usage