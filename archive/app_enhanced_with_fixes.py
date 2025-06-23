from g4f import ChatCompletion
from g4f.models import ModelUtils, IterListProvider

# Import specific providers
try:
    from g4f.Provider import Groq
    GROQ_PROVIDER_CLASS = Groq
except ImportError:
    print("Warning: Could not import g4f.Provider.Groq")
    GROQ_PROVIDER_CLASS = None

try:
    from g4f.Provider import Cerebras
    CEREBRAS_PROVIDER_CLASS = Cerebras
except ImportError:
    print("Warning: Could not import g4f.Provider.Cerebras")
    CEREBRAS_PROVIDER_CLASS = None

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
import urllib.parse
import time

# Try to import serpapi
try:
    import serpapi
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    print("Warning: serpapi not installed, SerpAPI functionality will be disabled")

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from flask import Flask, request, session, redirect, url_for
from flask_session import Session

# --- Configuration ---
CHAT_STORAGE = os.path.join(os.path.dirname(__file__), "chats.json")

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

print(f"--- API Keys Status ---")
print(f"GOOGLE_API_KEY: {'Loaded' if GOOGLE_API_KEY else 'Missing'}")
print(f"CEREBRAS_API_KEY: {'Loaded' if CEREBRAS_API_KEY else 'Missing'}")
print(f"GROQ_API_KEY: {'Loaded' if GROQ_API_KEY else 'Missing'}")
print(f"CHUTES_API_KEY: {'Loaded' if CHUTES_API_KEY else 'Missing'}")
print(f"SERPAPI_API_KEY: {'Loaded' if SERPAPI_API_KEY else 'Missing'}")
print(f"BRAVE_API_KEY: {'Loaded' if BRAVE_API_KEY else 'Missing'}")

# Model name mappings to handle duplicates and standardize display names
MODEL_DISPLAY_NAME_MAP = {
    # Llama 4 Maverick variants - all map to same display name
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "llama-4-maverick": "Llama 4 Maverick",
    "Llama 4 Maverick": "Llama 4 Maverick",
    "llama4-maverick": "Llama 4 Maverick",
    "llama-4-maverick-17b": "Llama 4 Maverick",
    "llama-4-maverick-17b-16e-instruct": "Llama 4 Maverick",
    "meta-llama/llama-4-maverick-17b-16e-instruct": "Llama 4 Maverick",
    "chutesai/Llama-4-Maverick-17B-128E-Instruct": "Llama 4 Maverick",
    "chutesai/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "chutesai/Llama-4-Maverick": "Llama 4 Maverick",
    "Llama 4 Maverick 17B 128E Instruct": "Llama 4 Maverick",
    "Llama 4 Maverick 17B 128E Instruct FP8": "Llama 4 Maverick",
    "Llama 4 Maverick (Turbo, FP8)": "Llama 4 Maverick",
    
    # Llama 4 Scout variants
    "llama-4-scout": "Llama 4 Scout",
    "Llama 4 Scout": "Llama 4 Scout",
    "llama4-scout": "Llama 4 Scout",
    "Llama 4 Scout 17B 16E Instruct": "Llama 4 Scout",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "llama-4-scout-17b": "Llama 4 Scout",
    
    # Llama 3.3 70B variants
    "llama-3.3-70b": "Llama 3.3 70B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "Llama 3.3 70B": "Llama 3.3 70B",
    "Llama 3.3 70B Instruct": "Llama 3.3 70B",
    
    # Llama 3.1 8B variants
    "llama-3.1-8b": "Llama 3.1 8B",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "Llama 3.1 8B": "Llama 3.1 8B",
    "Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    
    # QwQ variants
    "qwen-qwq-32b": "QwQ-32B",
    "Qwen QwQ 32B": "QwQ-32B",
    "QwQ 32B ArliAI RpR v1": "QwQ-32B",
    
    # Gemini variants
    "Gemini 2.5 Flash": "Gemini 2.5 Flash",
    "Gemini 2.0 Flash": "Gemini 2.0 Flash",
    "Gemini 2.0 Flash Lite": "Gemini 2.0 Flash Lite",
    
    # Other common models
    "claude-3-haiku": "Claude 3 Haiku",
    "Claude 3 Haiku": "Claude 3 Haiku",
    "deepseek-r1": "DeepSeek R1",
    "DeepSeek R1": "DeepSeek R1",
    "DeepSeek-R1": "DeepSeek R1",
    
    # Keep different parameter sizes separate
    "qwen/qwen3-32b": "Qwen 3 32B",
    "qwen/qwen3-235b-a22b": "Qwen 3 235B A22B",
    "Qwen 3 32B": "Qwen 3 32B", 
    "Qwen 3 235B A22B": "Qwen 3 235B A22B"
}

# API Configuration
CHUTES_API_URL = "https://llm.chutes.ai/v1"
GROQ_API_URL = "https://api.groq.com/openai/v1"
PROVIDER_PERFORMANCE_URL = "https://artificialanalysis.ai/leaderboards/providers"

# Performance data paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PERFORMANCE_CSV_PATH = os.path.join(APP_DIR, "provider_performance.csv")
LLM_LEADERBOARD_PATH = os.path.join("Coding Projects", "LLM-Performance-Leaderboard", "llm_leaderboard_20250521_013630.csv")

print(f"--- Performance CSV path: {PERFORMANCE_CSV_PATH} ---")

# Provider-specific model ID mapping
# This maps from display name to provider-specific model IDs
PROVIDER_MODEL_MAP = {
    "Llama 4 Maverick": {
        "cerebras": "cerebras/llama-4-maverick-17b-16e-instruct",
        "groq": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "chutes": "meta-llama/llama-4-7b-maverick"
    },
    "Llama 4 Scout": {
        "cerebras": "cerebras/llama-4-scout-17b-16e-instruct",
        "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
        "chutes": "meta-llama/llama-4-7b-scout"
    },
    "Llama-4-Maverick": {
        "cerebras": "cerebras/llama-4-maverick-17b-16e-instruct",
        "groq": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "chutes": "meta-llama/llama-4-7b-maverick"
    },
    "Llama-4-Scout": {
        "cerebras": "cerebras/llama-4-scout-17b-16e-instruct",
        "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
        "chutes": "meta-llama/llama-4-7b-scout"
    },
    "Llama 4 7B Maverick": {
        "cerebras": "cerebras/llama-4-maverick-17b-16e-instruct",
        "groq": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "chutes": "meta-llama/llama-4-7b-maverick"
    },
    "Llama 4 7B Scout": {
        "cerebras": "cerebras/llama-4-scout-17b-16e-instruct",
        "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
        "chutes": "meta-llama/llama-4-7b-scout"
    },
    "Llama 3.3 70B": {
        "cerebras": "cerebras/llama-3.3-70b-instruct",
        "groq": "llama3-70b-8192",
        "chutes": "meta-llama/llama-3.3-70b-instruct"
    },
    "Llama 3.3 8B": {
        "cerebras": "cerebras/llama-3.3-8b-instruct",
        "groq": "llama3-8b-8192",
        "chutes": "meta-llama/llama-3.3-8b-instruct"
    },
    "Llama 3.1 70B": {
        "groq": "llama3-70b-8192",
        "chutes": "meta-llama/llama-3.1-70b-instruct"
    },
    "Llama 3 8B": {
        "groq": "llama3-8b-8192",
        "chutes": "meta-llama/llama-3-8b-instruct"
    },
    "Llama 2 70B": {
        "groq": "llama2-70b-4096",
        "chutes": "meta-llama/llama-2-70b-chat"
    },
    "Gemini 2.5 Flash": {
        "google": "gemini-2.5-flash"
    },
    "Gemini 2.0 Flash": {
        "google": "gemini-2.0-flash"
    },
    "Claude 3.5 Sonnet": {
        "chutes": "anthropic/claude-3-5-sonnet-20240620"
    },
    "Claude 3 Opus": {
        "chutes": "anthropic/claude-3-opus-20240229"
    },
    "Claude 3 Sonnet": {
        "chutes": "anthropic/claude-3-sonnet-20240229"
    },
    "Claude 3 Haiku": {
        "chutes": "anthropic/claude-3-haiku-20240307"
    },
    "Qwen 3 32B": {
        "cerebras": "qwen-3-32b",
        "groq": "qwen/qwen3-32b",
        "google": "SKIP_PROVIDER",
        "chutes": "SKIP_PROVIDER"
    },
    "Qwen 3 235B A22B": {
        "cerebras": "SKIP_PROVIDER",
        "groq": "SKIP_PROVIDER",
        "google": "SKIP_PROVIDER",
        "chutes": "SKIP_PROVIDER"
    },
    "Qwen 3 72B": {
        "cerebras": "SKIP_PROVIDER",
        "groq": "SKIP_PROVIDER",
        "google": "SKIP_PROVIDER",
        "chutes": "SKIP_PROVIDER"
    }
}

# Global caches
CACHED_AVAILABLE_MODELS_SORTED_LIST = []
CACHED_MODEL_PROVIDER_INFO = {}
CACHED_PROVIDER_CLASS_MAP = {}
PROVIDER_PERFORMANCE_CACHE = []
CHUTES_MODELS_CACHE = []
GROQ_MODELS_CACHE = []

DEFAULT_MAX_OUTPUT_TOKENS = 4000  # Default for most models
REASONING_MAX_OUTPUT_TOKENS = 16384  # Higher limit for reasoning models (QwQ, Qwen3, etc.)

def get_max_tokens_for_model(model_name):
    """Determine appropriate max tokens based on model type"""
    model_lower = model_name.lower()
    
    # Reasoning models that need more tokens for thinking
    reasoning_keywords = ['qwq', 'qwen3', 'qwen 3', 'reasoning', 'thinking', 'claude-3', 'claude 3', 'o1']
    
    if any(keyword in model_lower for keyword in reasoning_keywords):
        return REASONING_MAX_OUTPUT_TOKENS
    
    return DEFAULT_MAX_OUTPUT_TOKENS

def is_continuation_request(prompt):
    """Check if the user is asking to continue a previous response"""
    prompt_lower = prompt.lower().strip()
    
    continuation_patterns = [
        r'\bcontinue\b',
        r'\bfinish\b',
        r'\bcomplete\b',
        r'\bmore\b',
        r'\bkeep going\b',
        r'\bgo on\b',
        r'\bplease continue\b',
        r'\bcontinue (?:your )?(?:response|answer|explanation|thinking)\b',
        r'\bfinish (?:your )?(?:response|answer|explanation|thinking)\b',
        r'\bcomplete (?:your )?(?:response|answer|explanation|thinking)\b',
        r'\bwhat (?:about|comes) next\b',
        r'\band then\?\s*$',
        r'\bwhat else\b'
    ]
    
    return any(re.search(pattern, prompt_lower) for pattern in continuation_patterns)

def build_continuation_context(chat_history, max_context_messages=5):
    """Build context for continuation requests using recent chat history"""
    if not chat_history or len(chat_history) < 2:
        return None
    
    # Get the last few messages for context
    recent_messages = chat_history[-max_context_messages:]
    
    # Find the last assistant message that might be incomplete
    last_assistant_msg = None
    last_user_msg = None
    
    for msg in reversed(chat_history):
        if msg["role"] == "assistant" and not last_assistant_msg:
            last_assistant_msg = msg
        elif msg["role"] == "user" and not last_user_msg:
            last_user_msg = msg
        
        if last_assistant_msg and last_user_msg:
            break
    
    if last_assistant_msg and last_user_msg:
        # Build context for continuation
        context = f"""Previous conversation context:

User's original question: {last_user_msg['content']}

Your previous response (which may have been cut off): {last_assistant_msg['content']}

User is now asking you to continue or complete your previous response. Please continue where you left off and provide the complete answer."""
        
        return context
    
    return None

def is_response_complete(response_text, model_name):
    """
    Check if a response appears to be complete or cut off.
    This is a heuristic and may not be 100% accurate.
    """
    if not response_text:
        return False
    
    response_text = response_text.strip()
    
    # Check for common indicators of incomplete responses
    incomplete_indicators = [
        # Mid-sentence cuts
        r'[a-z,]\s*$',  # Ends with lowercase letter or comma
        r'\.\.\.\s*$',  # Ends with ellipsis
        r':\s*$',       # Ends with colon
        r'-\s*$',       # Ends with dash
        r'and\s*$',     # Ends with "and"
        r'or\s*$',      # Ends with "or"
        r'but\s*$',     # Ends with "but"
        r'however\s*$', # Ends with "however"
        r'therefore\s*$', # Ends with "therefore"
        r'because\s*$', # Ends with "because"
        r'since\s*$',   # Ends with "since"
        r'while\s*$',   # Ends with "while"
        r'although\s*$', # Ends with "although"
        r'for example\s*$', # Ends with "for example"
        r'such as\s*$', # Ends with "such as"
        r'including\s*$', # Ends with "including"
        # Incomplete lists or enumerations
        r'\d+\.\s*$',   # Ends with number and period (list item)
        r'[•\-\*]\s*$', # Ends with bullet point
        # Incomplete code blocks (but not complete ones)
        r'```(?!.*```\s*$)[^`]*$',   # Unclosed code block (doesn't end with ```)
        r'`[^`]*[^`]\s*$',           # Unclosed inline code (not ending with `)
    ]
    
    for pattern in incomplete_indicators:
        if re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE):
            return False
    
    # For reasoning models, check for incomplete thinking blocks
    if any(keyword in model_name.lower() for keyword in ['qwq', 'qwen3', 'reasoning', 'thinking']):
        # Check for unclosed thinking tags or incomplete reasoning
        if '<thinking>' in response_text and '</thinking>' not in response_text:
            return False
        if response_text.count('<thinking>') != response_text.count('</thinking>'):
            return False
    
    # Response appears complete if it ends with proper punctuation or formatting
    complete_endings = ['.', '!', '?', '"', "'", ')', ']', '}', '```', '</thinking>']
    
    # Special check for code blocks - if it contains ``` and ends with ```, it's complete
    if '```' in response_text and response_text.strip().endswith('```'):
        return True
    
    return any(response_text.endswith(ending) for ending in complete_endings)

def get_complete_response_with_retries(prompt, model_name, max_tokens, api_call_function, max_retries=3):
    """
    Attempt to get a complete response, with retries if response appears incomplete.
    This implements the user's request to ensure complete responses before sending to user.
    """
    print(f"--- Attempting to get complete response for model: {model_name} ---")
    
    accumulated_response = ""
    continuation_prompt = prompt
    
    for attempt in range(max_retries):
        print(f"--- Response attempt {attempt + 1}/{max_retries} ---")
        
        try:
            # Get response from the API
            response = api_call_function(continuation_prompt, max_tokens)
            
            if not response or not response.strip():
                print(f"--- Attempt {attempt + 1} returned empty response ---")
                continue
                
            response = response.strip()
            print(f"--- Attempt {attempt + 1} response length: {len(response)} characters ---")
            
            # If this is our first response, use it as-is
            if attempt == 0:
                accumulated_response = response
            else:
                # For continuation attempts, append the new content
                accumulated_response += " " + response
            
            # Check if the response appears complete
            is_complete = is_response_complete(accumulated_response, model_name)
            print(f"--- Response appears complete: {is_complete} ---")
            
            if is_complete:
                print(f"--- Complete response obtained after {attempt + 1} attempt(s) ---")
                return accumulated_response.strip()
            
            # If incomplete and we have more retries, prepare continuation prompt
            if attempt < max_retries - 1:
                print(f"--- Response appears incomplete, preparing continuation for attempt {attempt + 2} ---")
                
                # Create a continuation prompt that includes the context
                continuation_prompt = f"""Continue your previous response. Here's what you said so far:

{accumulated_response}

Please continue from where you left off and complete your answer to the original question: {prompt}

Continue your response:"""
                
        except Exception as e:
            print(f"--- Error in attempt {attempt + 1}: {e} ---")
            continue
    
    # If we've exhausted all retries, return what we have
    print(f"--- Returning response after {max_retries} attempts (may be incomplete) ---")
    return accumulated_response.strip() if accumulated_response else None

# --- Utility Functions ---

def parse_model_name(model_name):
    """
    Parse a model name into its components: family, version, variant name, and parameter size.
    
    Args:
        model_name (str): The model name to parse
        
    Returns:
        dict: A dictionary with keys 'family', 'version', 'variant', 'size_b'
    """
    import re
    
    # Handle special cases first for exact matches
    special_cases = {
        "Llama 4 Maverick": {"family": "llama", "version": "4", "variant": "maverick", "size_b": 17.0},
        "Llama-4-Maverick": {"family": "llama", "version": "4", "variant": "maverick", "size_b": 17.0},
        "Llama 4 7B Maverick": {"family": "llama", "version": "4", "variant": "maverick", "size_b": 7.0},
        "Llama 4 Scout": {"family": "llama", "version": "4", "variant": "scout", "size_b": 17.0},
        "Llama-4-Scout": {"family": "llama", "version": "4", "variant": "scout", "size_b": 17.0},
        "Llama 4 7B Scout": {"family": "llama", "version": "4", "variant": "scout", "size_b": 7.0},
        "Qwen 3 32B": {"family": "qwen", "version": "3", "variant": "instruct", "size_b": 32.0},
        "Llama 3.3 70B": {"family": "llama", "version": "3.3", "variant": "instruct", "size_b": 70.0},
        "Llama 3.3 8B": {"family": "llama", "version": "3.3", "variant": "instruct", "size_b": 8.0}
    }
    
    # Check for exact match in special cases (case-insensitive)
    for case, values in special_cases.items():
        if model_name.lower() == case.lower():
            return values
    
    # Normalize the model name
    normalized = model_name.lower().strip()
    
    # Remove common prefixes and paths
    normalized = re.sub(r'^(meta-llama/|chutesai/|qwen/|meta-|deepseek-|claude-|gemini-|cerebras/)', '', normalized)
    
    # Extract parameter size (in billions)
    size_match = re.search(r'(\d+\.?\d*)[-\s]?[bB]', normalized)
    size_b = float(size_match.group(1)) if size_match else 0.0
    
    # Extract version (like 3.1, 4, 2.5)
    version_match = re.search(r'(\d+(?:\.\d+)?)', normalized)
    version = version_match.group(1) if version_match else ''
    
    # Identify model family (llama, gemini, claude, qwen, etc.)
    family_patterns = {
        'llama': r'llama',
        'gemini': r'gemini',
        'claude': r'claude',
        'qwen': r'qwen',
        'qwq': r'qwq',
        'deepseek': r'deepseek',
        'mistral': r'mistral',
        'gpt': r'gpt',
        'phi': r'phi',
        'yi': r'yi',
        'falcon': r'falcon',
        'mixtral': r'mixtral'
    }
    
    family = None
    for fam, pattern in family_patterns.items():
        if re.search(pattern, normalized):
            family = fam
            break
    
    if not family:
        family = normalized.split('-')[0].split(' ')[0]
    
    # Extract variant name (scout, maverick, haiku, etc.)
    variant_patterns = [
        r'(scout)',
        r'(maverick)',
        r'(haiku)',
        r'(sonnet)',
        r'(opus)',
        r'(flash)',
        r'(pro)',
        r'(lite)',
        r'(instruct)',
        r'(instant)',
        r'(r\d+)'  # For models like deepseek-r1
    ]
    
    variant = None
    for pattern in variant_patterns:
        match = re.search(pattern, normalized)
        if match:
            variant = match.group(1)
            break
            
    # Special handling for model names that contain specific keywords
    if 'llama-4-maverick' in normalized or 'llama 4 maverick' in normalized:
        family = 'llama'
        version = '4'
        variant = 'maverick'
        if not size_b:
            size_b = 17.0
            
    if 'llama-4-scout' in normalized or 'llama 4 scout' in normalized:
        family = 'llama'
        version = '4'
        variant = 'scout'
        if not size_b:
            size_b = 17.0
            
    if 'qwen 3' in normalized or 'qwen-3' in normalized:
        family = 'qwen'
        version = '3'
        if '32' in normalized:
            size_b = 32.0
    
    return {
        'family': family,
        'version': version,
        'variant': variant,
        'size_b': size_b
    }

def are_same_model(model_name1, model_name2):
    """
    Determine if two model names refer to the same underlying model.
    
    Args:
        model_name1 (str): First model name
        model_name2 (str): Second model name
        
    Returns:
        bool: True if they appear to be the same model, False otherwise
    """
    # Parse both model names
    model1 = parse_model_name(model_name1)
    model2 = parse_model_name(model_name2)
    
    # If both have families and they don't match, they're different models
    if model1['family'] and model2['family'] and model1['family'] != model2['family']:
        return False
    
    # If both have versions and they don't match, they're different models
    if model1['version'] and model2['version'] and model1['version'] != model2['version']:
        return False
    
    # If both have variants and they don't match, they're different models
    if model1['variant'] and model2['variant'] and model1['variant'] != model2['variant']:
        return False
    
    # If both have sizes and they don't match, they're different models
    if model1['size_b'] > 0 and model2['size_b'] > 0 and abs(model1['size_b'] - model2['size_b']) > 0.1:
        return False
    
    # If we have at least one matching component (besides size) and no conflicts, consider them the same
    has_matching_component = (
        (model1['family'] and model2['family'] and model1['family'] == model2['family']) or
        (model1['version'] and model2['version'] and model1['version'] == model2['version']) or
        (model1['variant'] and model2['variant'] and model1['variant'] == model2['variant'])
    )
    
    return has_matching_component

def get_canonical_model_name(model_name):
    """
    Generate a standardized canonical name for a model based on its components.
    
    Args:
        model_name (str): The model name to standardize
        
    Returns:
        str: A canonical model name in the format "Family Version Variant SizeB"
    """
    components = parse_model_name(model_name)
    
    # Capitalize family name
    family = components['family'].capitalize() if components['family'] else "Unknown"
    
    # Format the canonical name
    parts = [family]
    
    if components['version']:
        parts.append(components['version'])
    
    if components['variant']:
        parts.append(components['variant'].capitalize())
    
    if components['size_b'] > 0:
        # Check if size_b is a float before calling is_integer()
        if isinstance(components['size_b'], float):
            parts.append(f"{int(components['size_b']) if components['size_b'].is_integer() else components['size_b']}B")
        else:
            parts.append(f"{components['size_b']}B")
    
    return " ".join(parts)

def parse_context_window(context_str):
    """Parse context window size from string format (e.g., '32k', '1m') to numeric value."""
    if not context_str or str(context_str).lower() == 'n/a':
        return 0
    
    context_str = str(context_str).lower().strip()
    
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
        try:
            return int(float(context_str))
        except ValueError:
            return 0

def has_free_api_provider(model_name, provider_name):
    """Determine if a model has a free API provider."""
    if not model_name or not provider_name:
        return False
    
    # Known free providers (case-insensitive)
    free_providers = {
        'cerebras', 'groq', 'google ai studio', 'chutes', 'openrouter', 
        'sambanova', 'deepinfra', 'together', 'cohere', 'github models',
        'cloudflare workers ai', 'nvidia nim', 'mistral', 'huggingface'
    }
    
    provider_lower = provider_name.lower().strip()
    
    # Check if provider is in our free list
    for free_provider in free_providers:
        if free_provider in provider_lower:
            return True
    
    # G4F is always considered free but should be lowest priority
    if 'g4f' in provider_lower:
        return True
    
    return False

def get_provider_priority(provider_name):
    """Get priority score for provider ordering (lower = higher priority)."""
    if not provider_name:
        return 999
    
    provider_lower = provider_name.lower()
    
    # High priority providers (direct APIs)
    if 'cerebras' in provider_lower:
        return 1
    elif 'groq' in provider_lower:
        return 2
    elif 'google' in provider_lower:
        return 3
    elif 'chutes' in provider_lower:
        return 4
    elif 'sambanova' in provider_lower:
        return 5
    elif 'deepinfra' in provider_lower:
        return 6
    
    # Medium priority providers
    elif any(p in provider_lower for p in ['openrouter', 'together', 'cohere']):
        return 10
    
    # Low priority providers
    elif 'g4f' in provider_lower:
        return 50  # G4F should be last
    
    # Unknown providers
    return 25

# --- Performance Data Functions ---

def scrape_provider_performance(url=PROVIDER_PERFORMANCE_URL):
    """Fetch and parse provider performance data from multiple sources."""
    print(f"--- Scraping performance data from: {url} ---")
    performance_data = []
    
    # Scrape from main URL
    scraped_data = scrape_from_url(url)
    if scraped_data:
        performance_data.extend(scraped_data)
    
    # Load additional data from LLM-Performance-Leaderboard if available
    if os.path.exists(LLM_LEADERBOARD_PATH):
        try:
            print(f"--- Loading additional data from {LLM_LEADERBOARD_PATH} ---")
            with open(LLM_LEADERBOARD_PATH, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        provider = row.get('API Provider', '').strip()
                        model = row.get('Model', '').strip()
                        context_window_str = row.get('ContextWindow', '').strip()
                        intelligence_index_str = row.get('Artificial AnalysisIntelligence Index', '').strip()
                        response_time_str = row.get('Total Response (s)', '').strip().lower().replace('s', '').strip()
                        tokens_per_s_str = row.get('MedianTokens/s', '').strip().replace(',', '')
                        
                        # Parse numeric values with improved N/A handling
                        # Handle tokens_per_s
                        if tokens_per_s_str.lower() in ['n/a', 'na', '']:
                            tokens_per_s = 0.0
                        else:
                            try:
                                tokens_per_s = float(tokens_per_s_str)
                            except ValueError:
                                tokens_per_s = 0.0
                        
                        # Handle response_time_s
                        if response_time_str.lower() in ['n/a', 'na', 'inf', '']:
                            response_time_s = float('inf')
                        else:
                            try:
                                response_time_s = float(response_time_str)
                            except ValueError:
                                response_time_s = float('inf')
                        
                        # Handle intelligence_index
                        if intelligence_index_str.lower() in ['n/a', 'na', ''] or not intelligence_index_str:
                            intelligence_index = 0.0
                        else:
                            try:
                                intelligence_index = float(intelligence_index_str)
                            except ValueError:
                                intelligence_index = 0.0
                        
                        if provider and model:
                            # Check if this model+provider already exists
                            existing_entry = next((item for item in performance_data 
                                                if item['provider_name_scraped'].lower() == provider.lower() 
                                                and item['model_name_scraped'].lower() == model.lower()), None)
                            
                            if existing_entry:
                                # Update existing entry with better data
                                if context_window_str and not existing_entry.get('context_window'):
                                    existing_entry['context_window'] = context_window_str
                                if intelligence_index > 0 and existing_entry.get('intelligence_index', 0) == 0:
                                    existing_entry['intelligence_index'] = intelligence_index
                                if response_time_s < existing_entry.get('response_time_s', float('inf')):
                                    existing_entry['response_time_s'] = response_time_s
                                if tokens_per_s > existing_entry.get('tokens_per_s', 0):
                                    existing_entry['tokens_per_s'] = tokens_per_s
                            else:
                                # Add new entry
                                performance_data.append({
                                    'provider_name_scraped': provider,
                                    'model_name_scraped': model,
                                    'context_window': context_window_str,
                                    'intelligence_index': intelligence_index,
                                    'response_time_s': response_time_s,
                                    'tokens_per_s': tokens_per_s
                                })
                    except Exception as e:
                        print(f"--- Warning: Could not parse CSV row: {e} ---")
            
            print(f"--- Successfully loaded additional data from LLM-Performance-Leaderboard ---")
        except Exception as e:
            print(f"--- Error loading LLM-Performance-Leaderboard data: {e} ---")
    
    print(f"--- Total performance entries collected: {len(performance_data)} ---")
    return performance_data

def scrape_from_url(url):
    """Scrape performance data from a specific URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table')
        if not table:
            print(f"--- Warning: No table found on {url} ---")
            return []
        
        data = []
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                row_data = [cell.get_text(strip=True) for cell in cells]
                
                provider = row_data[0] if len(row_data) > 0 else ''
                model = row_data[1] if len(row_data) > 1 else ''
                
                # Parse performance data from remaining columns
                intelligence_index = 0.0
                response_time_s = float('inf')
                context_window = ''
                tokens_per_s = 0.0
                
                for i, cell_text in enumerate(row_data[2:], 2):
                    try:
                        if cell_text.replace('.', '', 1).isdigit():
                            val = float(cell_text)
                            if 0 <= val <= 100 and intelligence_index == 0:
                                intelligence_index = val
                            elif val > 100 and tokens_per_s == 0:
                                tokens_per_s = val
                        elif 's' in cell_text.lower():
                            time_str = cell_text.lower().replace('s', '').strip()
                            if time_str.replace('.', '', 1).isdigit():
                                response_time_s = float(time_str)
                        elif any(suffix in cell_text.lower() for suffix in ['k', 'm']) or cell_text.isdigit():
                            context_window = cell_text
                    except ValueError:
                        continue
                
                if provider and model:
                    data.append({
                        'provider_name_scraped': provider,
                        'model_name_scraped': model,
                        'context_window': context_window,
                        'intelligence_index': intelligence_index,
                        'response_time_s': response_time_s,
                        'tokens_per_s': tokens_per_s
                    })
        
        print(f"--- Scraped {len(data)} entries from {url} ---")
        return data
        
    except Exception as e:
        print(f"--- Error scraping from {url}: {e} ---")
        return []

def save_performance_to_csv(performance_data, csv_path=PERFORMANCE_CSV_PATH):
    """Save performance data to CSV file."""
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['provider_name_scraped', 'model_name_scraped', 'context_window', 
                         'intelligence_index', 'response_time_s', 'tokens_per_s', 
                         'source_url', 'last_updated_utc', 'is_free_source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            
            for entry in performance_data:
                is_free = has_free_api_provider(
                    entry.get('model_name_scraped', ''), 
                    entry.get('provider_name_scraped', '')
                )
                
                writer.writerow({
                    'provider_name_scraped': entry.get('provider_name_scraped', ''),
                    'model_name_scraped': entry.get('model_name_scraped', ''),
                    'context_window': entry.get('context_window', ''),
                    'intelligence_index': entry.get('intelligence_index', 0),
                    'response_time_s': entry.get('response_time_s', float('inf')),
                    'tokens_per_s': entry.get('tokens_per_s', 0),
                    'source_url': PROVIDER_PERFORMANCE_URL,
                    'last_updated_utc': current_time,
                    'is_free_source': is_free
                })
        
        print(f"--- Successfully saved {len(performance_data)} entries to {csv_path} ---")
        
    except Exception as e:
        print(f"--- Error saving performance data to CSV: {e} ---")

def load_performance_from_csv(csv_path=PERFORMANCE_CSV_PATH):
    """Load performance data from CSV file."""
    performance_data = []
    
    if not os.path.exists(csv_path):
        print(f"--- Performance CSV not found at {csv_path}. Will scrape fresh data. ---")
        return performance_data
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    # Handle intelligence_index
                    intel_str = row.get('intelligence_index', '0').strip()
                    if intel_str.lower() in ['n/a', 'na', '']:
                        intelligence_index = 0.0
                    else:
                        intelligence_index = float(intel_str)
                    
                    # Handle response_time_s
                    time_str = row.get('response_time_s', 'inf').strip()
                    if time_str.lower() in ['n/a', 'na', 'inf', '']:
                        response_time_s = float('inf')
                    else:
                        response_time_s = float(time_str)
                    
                    # Handle tokens_per_s
                    tokens_str = row.get('tokens_per_s', '0').strip()
                    if tokens_str.lower() in ['n/a', 'na', '']:
                        tokens_per_s = 0.0
                    else:
                        tokens_per_s = float(tokens_str)
                    
                    performance_data.append({
                        'provider_name_scraped': row.get('provider_name_scraped', ''),
                        'model_name_scraped': row.get('model_name_scraped', ''),
                        'context_window': row.get('context_window', ''),
                        'intelligence_index': intelligence_index,
                        'response_time_s': response_time_s,
                        'tokens_per_s': tokens_per_s
                    })
                except ValueError as e:
                    print(f"--- Warning: Could not parse row in CSV: {e} ---")
                    
        print(f"--- Successfully loaded {len(performance_data)} entries from {csv_path} ---")
        
    except Exception as e:
        print(f"--- Error loading performance data from CSV: {e} ---")
    
    return performance_data

# --- Model Sorting and Aggregation ---

def get_available_models_with_provider_counts():
    """
    Returns sorted models with dynamic ordering logic:
    1. Priority models: fastest with increasing intelligence or same intelligence with larger context
    2. Remaining free models by response time  
    3. Models with no intelligence score at bottom
    """
    global PROVIDER_PERFORMANCE_CACHE
    import re
    
    # Build performance entries from cache
    perf_entries = []
    for entry in PROVIDER_PERFORMANCE_CACHE:
        model_name = entry.get('model_name_scraped', '').strip()
        provider_name = entry.get('provider_name_scraped', '').strip()
        if not model_name or not provider_name:
            continue
            
        intelligence_index = entry.get('intelligence_index', 0)
        response_time_s = entry.get('response_time_s', float('inf'))
        context_window_str = entry.get('context_window', '')
        
        try:
            context_window_size = parse_context_window(context_window_str)
        except:
            context_window_size = 0
            
        is_free = has_free_api_provider(model_name, provider_name)
        provider_priority = get_provider_priority(provider_name)
        
        perf_entries.append({
            'model': model_name,
            'provider': provider_name,
            'intelligence_index': intelligence_index,
            'response_time_s': response_time_s,
            'context_window_size': context_window_size,
            'is_free': is_free,
            'provider_priority': provider_priority
        })
    
    # Helper to extract parameter count from model name
    def extract_param_count(name):
        match = re.search(r'(\d+\.?\d*)[bB]', name)
        return float(match.group(1)) if match else 0.0
    
    # Aggregate models using our intelligent model name parser
    model_aggregation = {}
    for entry in perf_entries:
        if not entry['is_free']:
            continue
            
        model_name = entry['model']
        provider_name = entry['provider']
        
        # Generate a canonical model name
        canonical_name = get_canonical_model_name(model_name)
        
        # Extract model components for grouping
        model_components = parse_model_name(model_name)
        
        # Create a unique key based on family, version, variant, and size
        model_key = (
            model_components['family'],
            model_components['version'],
            model_components['variant'],
            model_components['size_b']
        )
        
        if model_key not in model_aggregation:
            model_aggregation[model_key] = {
                'display_name': canonical_name,
                'providers': {provider_name},
                'intelligence_index': entry['intelligence_index'],
                'response_time_s': entry['response_time_s'],
                'context_window_size': entry['context_window_size'],
                'best_model_name': model_name,
                'best_provider_priority': entry['provider_priority'],
                'original_model_names': {model_name}  # Track all original names
            }
        else:
            # Update with better metrics, prioritizing better providers
            existing = model_aggregation[model_key]
            existing['providers'].add(provider_name)
            existing['original_model_names'].add(model_name)
            
            # Prefer entries with better provider priority, then performance
            is_better = (
                entry['provider_priority'] < existing['best_provider_priority'] or
                (entry['provider_priority'] == existing['best_provider_priority'] and 
                 (entry['response_time_s'] < existing['response_time_s'] or 
                  (entry['response_time_s'] == existing['response_time_s'] and 
                   entry['intelligence_index'] > existing['intelligence_index'])))
            )
            
            if is_better:
                existing.update({
                    'intelligence_index': entry['intelligence_index'],
                    'response_time_s': entry['response_time_s'],
                    'context_window_size': entry['context_window_size'],
                    'best_model_name': model_name,
                    'best_provider_priority': entry['provider_priority']
                })
    
    # Convert to list and separate by intelligence score
    free_models = list(model_aggregation.values())
    models_with_intelligence = [m for m in free_models if m['intelligence_index'] > 0]
    models_without_intelligence = [m for m in free_models if m['intelligence_index'] == 0]
    
    # Sort models with intelligence by response time first
    models_with_intelligence.sort(key=lambda x: x['response_time_s'])
    
    # Apply dynamic ordering logic for priority models
    priority_models = []
    if models_with_intelligence:
        # Start with the fastest model
        current_model = models_with_intelligence[0]
        priority_models.append(current_model)
        
        current_intelligence = current_model['intelligence_index']
        current_context = current_model['context_window_size']
        
        # Find subsequent models that meet the criteria
        for model in models_with_intelligence[1:]:
            if (model['intelligence_index'] > current_intelligence or 
                (model['intelligence_index'] == current_intelligence and 
                 model['context_window_size'] > current_context)):
                priority_models.append(model)
                current_intelligence = model['intelligence_index']
                current_context = model['context_window_size']
    
    # Remaining models with intelligence (not in priority list)
    remaining_models = [m for m in models_with_intelligence if m not in priority_models]
    remaining_models.sort(key=lambda x: x['response_time_s'])
    
    # Models without intelligence, sorted by response time
    models_without_intelligence.sort(key=lambda x: x['response_time_s'])
    
    # Combine all models: priority first, then remaining, then no intelligence
    final_models = priority_models + remaining_models + models_without_intelligence
    
    # Prepare output format
    sorted_models = []
    model_provider_info = {}
    for model in final_models:
        display_name = model['display_name']
        provider_count = len(model['providers'])
        intelligence_index = model['intelligence_index']
        response_time_s = model['response_time_s']
        
        sorted_models.append((display_name, provider_count, intelligence_index, response_time_s))
        model_provider_info[display_name] = list(model['providers'])
    
    return sorted_models, model_provider_info, {}

# --- External API Functions ---

async def fetch_chutes_models():
    """Fetch available models from Chutes AI API."""
    global CHUTES_MODELS_CACHE
    
    if not CHUTES_API_KEY:
        print("--- Warning: CHUTES_API_KEY not found. Skipping Chutes model fetch. ---")
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {CHUTES_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{CHUTES_API_URL}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])
                    CHUTES_MODELS_CACHE = [model.get('id', '') for model in models if model.get('id')]
                    print(f"--- Successfully fetched {len(CHUTES_MODELS_CACHE)} models from Chutes AI ---")
                    return CHUTES_MODELS_CACHE
                else:
                    print(f"--- Chutes API error: {response.status} ---")
                    return []
    except Exception as e:
        print(f"--- Error fetching Chutes models: {e} ---")
        return []

async def fetch_groq_models():
    """Fetch available models from Groq API."""
    global GROQ_MODELS_CACHE
    
    if not GROQ_API_KEY:
        print("--- Warning: GROQ_API_KEY not found. Skipping Groq model fetch. ---")
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{GROQ_API_URL}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])
                    GROQ_MODELS_CACHE = [model.get('id', '') for model in models if model.get('id')]
                    print(f"--- Successfully fetched {len(GROQ_MODELS_CACHE)} models from Groq ---")
                    return GROQ_MODELS_CACHE
                else:
                    print(f"--- Groq API error: {response.status} ---")
                    return []
    except Exception as e:
        print(f"--- Error fetching Groq models: {e} ---")
        return []

# --- Web Search Function ---

def perform_bing_search_fallback(query, max_results=5):
    """
    Fallback search using Bing's public search without API key.
    This scrapes Bing search results as a last resort.
    """
    results = []
    try:
        print(f"--- Attempting Bing fallback search for: {query} ---")
        
        # Encode the query for URL
        encoded_query = urllib.parse.quote_plus(query)
        bing_url = f"https://www.bing.com/search?q={encoded_query}&count={max_results}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(bing_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result items
            search_results = soup.find_all('li', class_='b_algo')
            
            for result in search_results[:max_results]:
                title_elem = result.find('h2')
                snippet_elem = result.find('div', class_='b_caption')
                
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    
                    # Try to get the link
                    link_elem = title_elem.find('a')
                    link = link_elem.get('href', '') if link_elem else ''
                    
                    if title and snippet:
                        results.append({
                            'title': title,
                            'link': link,
                            'snippet': snippet
                        })
        
        print(f"--- Bing fallback search completed. Found {len(results)} results. ---")
        return results
    except Exception as e:
        print(f"--- Bing fallback search failed: {e} ---")
        return []

def perform_brave_search(query, max_results=5):
    """
    Perform search using Brave Search API.
    Brave provides 2000 free searches per month.
    """
    results = []
    try:
        print(f"--- Attempting Brave Search API for: {query} ---")
        
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': BRAVE_API_KEY
        }
        
        params = {
            'q': query,
            'count': max_results,
            'search_lang': 'en',
            'country': 'US',
            'safesearch': 'moderate',
            'result_filter': 'web'
        }
        
        response = requests.get(
            'https://api.search.brave.com/res/v1/web/search',
            headers=headers,
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if 'web' in data and 'results' in data['web']:
                for result in data['web']['results'][:max_results]:
                    results.append({
                        'title': result.get('title', ''),
                        'link': result.get('url', ''),
                        'snippet': result.get('description', '')
                    })
        elif response.status_code == 429:
            print("--- Brave Search API rate limited ---")
        else:
            print(f"--- Brave Search API error: {response.status_code} ---")
        
        print(f"--- Brave Search API completed. Found {len(results)} results. ---")
        return results
    except Exception as e:
        print(f"--- Brave Search API failed: {e} ---")
        return []

def perform_google_fallback_search(query, max_results=5):
    """
    Fallback search using Google's public search without API key.
    This scrapes Google search results as a last resort.
    """
    results = []
    try:
        print(f"--- Attempting Google fallback search for: {query} ---")
        
        # Encode the query for URL
        encoded_query = urllib.parse.quote_plus(query)
        google_url = f"https://www.google.com/search?q={encoded_query}&num={max_results}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(google_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result items (Google's structure)
            search_results = soup.find_all('div', class_=['g', 'Gx5Zad'])
            
            for result in search_results[:max_results]:
                # Try to find title
                title_elem = result.find('h3')
                if not title_elem:
                    continue
                    
                title = title_elem.get_text(strip=True)
                
                # Try to find snippet
                snippet_elem = result.find(['span', 'div'], class_=['st', 'VwiC3b'])
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                # Try to find link
                link_elem = result.find('a')
                link = link_elem.get('href', '') if link_elem else ''
                
                if title:
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
        
        print(f"--- Google fallback search completed. Found {len(results)} results. ---")
        return results
    except Exception as e:
        print(f"--- Google fallback search failed: {e} ---")
        return []

def perform_web_search(query, max_results=5):
    """
    Perform a web search using multiple fallback methods.
    Tries DuckDuckGo first, then SerpAPI (if available and has credits),
    then Brave Search API (if available), then free fallback methods (Bing, Google scraping).
    """
    results = []
    
    # Method 1: Try DuckDuckGo first (most reliable free option)
    try:
        print(f"--- Performing DuckDuckGo search for: {query} ---")
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query, max_results=max_results))
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
                for result in search_results['organic_results']:
                    results.append({
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'snippet': result.get('snippet', '')
                    })
                print(f"--- SerpApi search completed. Found {len(results)} results. ---")
                if results:
                    return results
        except Exception as e:
            error_str = str(e).lower()
            if 'quota' in error_str or 'limit' in error_str or 'credit' in error_str:
                print(f"--- SerpAPI quota/credits exhausted: {e} ---")
            else:
                print(f"--- SerpApi search failed: {e} ---")
    
    # Method 3: Try Brave Search API if available
    if BRAVE_API_KEY:
        results = perform_brave_search(query, max_results)
        if results:
            return results
    
    # Method 4: Try Bing fallback search (scraping)
    results = perform_bing_search_fallback(query, max_results)
    if results:
        return results
    
    # Method 5: Try Google fallback search (scraping) as last resort
    results = perform_google_fallback_search(query, max_results)
    if results:
        return results
    
    print("--- All web search methods failed. Returning empty results. ---")
    return results

# --- Chat Management Functions ---

def load_chats():
    """Load chat histories from JSON file."""
    if os.path.exists(CHAT_STORAGE):
        try:
            with open(CHAT_STORAGE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("--- Warning: Chat storage file is corrupted. Starting fresh. ---")
            return {}
    return {}

def save_chats(chats):
    """Save chat histories to JSON file."""
    try:
        with open(CHAT_STORAGE, 'w', encoding='utf-8') as f:
            json.dump(chats, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"--- Error saving chats: {e} ---")

# --- Initialize Model Cache ---

def initialize_model_cache():
    """Initialize the global model cache by calling get_available_models_with_provider_counts."""
    global CACHED_AVAILABLE_MODELS_SORTED_LIST, CACHED_MODEL_PROVIDER_INFO, CACHED_PROVIDER_CLASS_MAP
    print("--- [STARTUP] Initializing model cache... ---")
    
    try:
        sorted_models, provider_info, provider_class_map = get_available_models_with_provider_counts()
        CACHED_AVAILABLE_MODELS_SORTED_LIST = sorted_models
        CACHED_MODEL_PROVIDER_INFO = provider_info
        CACHED_PROVIDER_CLASS_MAP = provider_class_map
        
        print(f"--- [STARTUP] Model cache initialized successfully. {len(sorted_models)} models available. ---")
        
    except Exception as e:
        print(f"--- [STARTUP] Error initializing model cache: {e} ---")
        CACHED_AVAILABLE_MODELS_SORTED_LIST = []
        CACHED_MODEL_PROVIDER_INFO = {}
        CACHED_PROVIDER_CLASS_MAP = {}

# --- Initialize Performance Data ---
print("--- [MODULE LOAD] Loading performance data... ---")
PROVIDER_PERFORMANCE_CACHE = load_performance_from_csv(PERFORMANCE_CSV_PATH)

if not PROVIDER_PERFORMANCE_CACHE:
    print("--- [MODULE LOAD] No cached data found. Scraping fresh data... ---")
    scraped_data = scrape_provider_performance()
    if scraped_data:
        save_performance_to_csv(scraped_data, PERFORMANCE_CSV_PATH)
        PROVIDER_PERFORMANCE_CACHE = scraped_data
        print(f"--- [MODULE LOAD] Successfully scraped and cached {len(scraped_data)} performance entries ---")
    else:
        print("--- [MODULE LOAD] Warning: No performance data could be scraped ---")
else:
    print(f"--- [MODULE LOAD] Loaded {len(PROVIDER_PERFORMANCE_CACHE)} performance entries from cache ---")

# --- Fetch External Models ---
print("--- [MODULE LOAD] Fetching external models... ---")
try:
    asyncio.run(fetch_chutes_models())
    asyncio.run(fetch_groq_models())
except Exception as e:
    print(f"--- [MODULE LOAD] Error fetching external models: {e} ---")

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = "your-secret-key-change-this-in-production"
app.config["SESSION_TYPE"] = "filesystem"

# Ensure the session directory exists
SESSION_DIR = './flask_session'
if not os.path.exists(SESSION_DIR):
    os.makedirs(SESSION_DIR)
app.config["SESSION_FILE_DIR"] = SESSION_DIR
Session(app)

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    chats = load_chats()
    
    # Use cached model data
    available_models_sorted_list = CACHED_AVAILABLE_MODELS_SORTED_LIST
    available_model_names = {name for name, _, _, _ in available_models_sorted_list}
    
    # Set default model
    if available_models_sorted_list:
        default_model = available_models_sorted_list[0][0]
        print(f"--- Setting default model to: {default_model} ---")
    else:
        default_model = "gpt-3.5-turbo"
        print("--- Warning: No available models found. Using fallback. ---")
    
    # Check user's previously selected model
    user_selected_model = session.get('user_selected_model')
    if user_selected_model and user_selected_model in available_model_names:
        default_model = user_selected_model
        print(f"--- Using user's previously selected model: {default_model} ---")
    
    # Handle model selection from form
    if request.method == 'POST' and 'model' in request.form:
        selected_model = request.form.get('model')
        if selected_model in available_model_names:
            session['user_selected_model'] = selected_model
            default_model = selected_model
            print(f"--- Using model from form: {default_model} ---")
    
    # Initialize or load current chat
    if 'current_chat' not in session or session['current_chat'] not in chats:
        current_time = datetime.now().isoformat()
        session['current_chat'] = str(uuid.uuid4())
        chats[session['current_chat']] = {
            "history": [],
            "model": default_model,
            "name": "New Chat",
            "created_at": current_time,
            "last_modified": current_time
        }
        save_chats(chats)
    
    # Handle case where session refers to a deleted chat
    if session['current_chat'] not in chats:
        if chats:
            latest_chat_id = sorted(chats.keys(), 
                                  key=lambda k: chats[k].get('last_modified', 
                                                            chats[k].get('created_at', '')), 
                                  reverse=True)[0]
            session['current_chat'] = latest_chat_id
        else:
            current_time = datetime.now().isoformat()
            session['current_chat'] = str(uuid.uuid4())
            chats[session['current_chat']] = {
                "history": [],
                "model": default_model,
                "name": "New Chat",
                "created_at": current_time,
                "last_modified": current_time
            }
            save_chats(chats)
    
    current_chat = chats[session['current_chat']]
    
    # Update chat model if changed
    if request.method == 'POST' and 'model' in request.form:
        selected_model = request.form.get('model')
        if selected_model in available_model_names:
            current_chat["model"] = selected_model
            session['user_selected_model'] = selected_model
            current_chat["last_modified"] = datetime.now().isoformat()
            save_chats(chats)
            print(f"--- Updated chat model to: {selected_model} ---")
    
    current_model = current_chat.get("model", default_model)
    
    # Handle POST requests (new chat, delete chat, message submission)
    if request.method == 'POST':
        # Handle New Chat
        if 'new_chat' in request.form:
            current_model_selection = request.form.get('model', default_model)
            if current_model_selection in available_model_names:
                session['user_selected_model'] = current_model_selection
                new_chat_model = current_model_selection
            else:
                new_chat_model = default_model
                
            session['current_chat'] = str(uuid.uuid4())
            chats[session['current_chat']] = {
                "history": [],
                "model": new_chat_model,
                "name": "New Chat",
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat()
            }
            save_chats(chats)
            return redirect(url_for('index'))
        
        # Handle Delete Chat
        if 'delete_chat' in request.form:
            chat_to_delete = session.get('current_chat')
            if chat_to_delete and chat_to_delete in chats:
                print(f"--- Deleting chat: {chat_to_delete} ---")
                del chats[chat_to_delete]
                save_chats(chats)
                session.pop('current_chat', None)
                return redirect(url_for('index'))
        
        # Handle message submission
        prompt_from_input = request.form.get('prompt', '').strip()
        web_search_mode = request.form.get('web_search_mode', 'smart')
        is_regeneration = 'regenerate' in request.form
        
        if prompt_from_input or is_regeneration:
            try:
                # Handle regeneration - remove last assistant message
                if is_regeneration and current_chat["history"] and current_chat["history"][-1]["role"] == "assistant":
                    print("--- Removing previous assistant message for regeneration ---")
                    current_chat["history"].pop()
                    # Use the last user message for regeneration
                    last_user_msg = next((msg["content"] for msg in reversed(current_chat["history"]) if msg["role"] == "user"), None)
                    prompt_to_use = last_user_msg if last_user_msg else "Hello"
                else:
                    prompt_to_use = prompt_from_input
                
                # Add user message to history (only for new messages, not regeneration)
                if prompt_from_input and not is_regeneration:
                    current_time = datetime.now().isoformat()
                    current_chat["history"].append({
                        "role": "user",
                        "content": prompt_from_input,
                        "timestamp": current_time
                    })
                
                # Web search logic with error handling
                search_results_str = ""
                try:
                    if web_search_mode == 'on':
                        print(f"--- Web search explicitly enabled for: {prompt_to_use[:50]}... ---")
                        search_results = perform_web_search(prompt_to_use)
                        if search_results:
                            search_results_str = "Web search results:\n" + "\n".join([
                                f"• {result['title']}: {result['snippet']}" for result in search_results[:3]
                            ])
                            print(f"--- Web search (ON) found {len(search_results)} results ---")
                        else:
                            print("--- Web search (ON) found no results ---")
                    elif web_search_mode == 'smart':
                        # Enhanced smart search logic
                        should_search = False
                        prompt_lower = prompt_to_use.lower()
                        
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
                        
                        if not is_excluded:
                            # Check all keyword categories
                            all_keywords = time_keywords + question_patterns + tech_keywords + sports_keywords + finance_keywords
                            
                            # Basic keyword matching
                            if any(keyword in prompt_lower for keyword in all_keywords):
                                should_search = True
                        
                        # Additional pattern-based checks
                        if not should_search:
                            # Check for year references (2020-2030)
                            if re.search(r'\b(202[0-9])\b', prompt_lower):
                                should_search = True
                            
                            # Check for "when did" or "when was" questions
                            if re.search(r'\b(when (did|was|is|will)|how (long|much|many))\b', prompt_lower):
                                should_search = True
                            
                            # Check for specific date patterns
                            if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', prompt_lower):
                                should_search = True
                            
                            # Check for company/person names that might need current info
                            if re.search(r'\b(google|apple|microsoft|amazon|tesla|meta|facebook|twitter|x\.com|openai|anthropic|nvidia)\b', prompt_lower):
                                should_search = True
                        
                        if should_search:
                            print(f"--- Smart search triggered for: {prompt_to_use[:50]}... ---")
                            search_results = perform_web_search(prompt_to_use)
                            if search_results:
                                search_results_str = "Current web search results:\n" + "\n".join([
                                    f"• {result['title']}: {result['snippet']}" for result in search_results[:3]
                                ])
                                print(f"--- Smart search found {len(search_results)} results ---")
                            else:
                                print("--- Smart search found no results ---")
                        else:
                            print(f"--- Smart search: No current info needed for: {prompt_to_use[:50]}... ---")
                except Exception as e:
                    print(f"--- Web search error: {e} ---")
                    search_results_str = ""  # Ensure we continue without search results
                
                # Get the display name and canonical name for the current model
                display_name = MODEL_DISPLAY_NAME_MAP.get(current_model, current_model)
                canonical_name = get_canonical_model_name(current_model)
                
                # Get appropriate max tokens for this model
                max_tokens_for_model = get_max_tokens_for_model(display_name)
                
                # Try to call actual LLM APIs
                response_content = None
                provider_used = "Unknown"
                
                # Check if this is a continuation request
                is_continuation = is_continuation_request(prompt_to_use)
                continuation_context = None
                
                if is_continuation:
                    print(f"--- Detected continuation request: {prompt_to_use[:50]}... ---")
                    continuation_context = build_continuation_context(current_chat["history"])
                    if continuation_context:
                        print("--- Built continuation context from chat history ---")
                
                # Prepare the message for the API with search results if available
                final_prompt = prompt_to_use
                
                if continuation_context:
                    # Use continuation context instead of normal prompt
                    final_prompt = continuation_context
                elif search_results_str:
                    # Format the prompt with search results for better LLM understanding
                    final_prompt = f"""{search_results_str}

Based on the above search results and your knowledge, please answer the following question:
{prompt_to_use}

Please provide a comprehensive answer that incorporates both the current information from the search results and relevant background knowledge."""
                
                # Define all available API providers to try, regardless of model
                # Prioritize direct API providers over G4F
                direct_api_providers = []
                g4f_providers = []
                
                # First, check if we have specific providers for this model
                model_providers = CACHED_MODEL_PROVIDER_INFO.get(display_name, [])
                
                # Separate direct API providers from G4F providers
                for provider in model_providers:
                    if any(api_name in provider.lower() for api_name in ['cerebras', 'groq', 'google', 'chutes']):
                        direct_api_providers.append(provider)
                    else:
                        g4f_providers.append(provider)
                
                # Sort direct API providers by priority
                direct_api_providers = sorted(direct_api_providers, key=get_provider_priority)
                
                # Try all direct API providers first, even if they're not listed for this specific model
                # This allows us to try matching similar models across providers
                
                # Track attempted providers and errors
                attempted_direct_providers = set()
                provider_errors = {}
                
                # Try Cerebras API
                if CEREBRAS_API_KEY and not response_content:
                    try:
                        # Get the Cerebras-specific model name
                        cerebras_model = find_provider_specific_model(current_model, "cerebras")
                        provider_name_for_attempt = "Cerebras (Direct API)"
                        attempted_direct_providers.add("cerebras")
                        
                        # Skip this provider if the model is not supported
                        if cerebras_model == "SKIP_PROVIDER":
                            print(f"--- Skipping Cerebras API for {display_name} (not supported) ---")
                            provider_errors[provider_name_for_attempt] = "Model not supported"
                        else:
                            # Use the mapped ID if it exists, otherwise use the original selection
                            model_to_use_cerebras = cerebras_model if cerebras_model else current_model
                            
                            # Ensure the model name is in the correct format for Cerebras
                            if not model_to_use_cerebras.startswith("cerebras/") and not "/" in model_to_use_cerebras:
                                model_to_use_cerebras = f"cerebras/{model_to_use_cerebras}"
                                
                            print(f"--- Attempting Cerebras API for {display_name} using model: {model_to_use_cerebras} ---")
                            response_content = call_cerebras_api(model_to_use_cerebras, final_prompt)
                            
                            # Validate response content
                            if response_content and response_content.strip():
                                content_str = response_content.strip()
                                low = content_str.lower()
                                if not (low.startswith("error:") or low.startswith("you have reached") or 
                                        "challenge error" in low or "rate limit" in low or 
                                        "no provider found" in low or "no providers found" in low or 
                                        "context_length_exceeded" in low or "request entity too large" in low or 
                                        "model_not_found" in low or "token" in low):
                                    provider_used = provider_name_for_attempt
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
                            error_msg = f"Provider Cerebras (Direct API) failed due to token rate limit: {e}"
                            provider_errors["Cerebras (Direct API)"] = f"Token rate limit: {str(e)}"
                        else:
                            error_msg = f"Provider Cerebras (Direct API) failed: {e}"
                            provider_errors["Cerebras (Direct API)"] = str(e)
                        print(f"--- {error_msg} ---")
                        response_content = None
                
                # Try Groq API
                if GROQ_API_KEY and not response_content:
                    try:
                        # Get the Groq-specific model name
                        groq_model = find_provider_specific_model(current_model, "groq")
                        provider_name_for_attempt = "Groq (Direct API)"
                        attempted_direct_providers.add("groq")
                        
                        # Skip this provider if the model is not supported
                        if groq_model == "SKIP_PROVIDER":
                            print(f"--- Skipping Groq API for {display_name} (not supported) ---")
                            provider_errors[provider_name_for_attempt] = "Model not supported"
                        else:
                            # Check if it's a Llama model and choose appropriate fallback if needed
                            if not groq_model or groq_model == current_model:
                                model_components = parse_model_name(current_model)
                                if 'llama' in model_components['family'].lower():
                                    if '4' in str(model_components['version']):
                                        if 'maverick' in str(model_components['variant']).lower():
                                            groq_model = "meta-llama/llama-4-maverick-17b-128e-instruct"
                                        elif 'scout' in str(model_components['variant']).lower():
                                            groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
                                        else:
                                            groq_model = "meta-llama/llama-4-maverick-17b-128e-instruct"  # Default to Maverick
                                    elif '3.3' in str(model_components['version']):
                                        if '70' in str(model_components['size_b']):
                                            groq_model = "llama3-70b-8192"
                                        else:
                                            groq_model = "llama3-8b-8192"
                                    elif '3.1' in str(model_components['version']) or '3' in str(model_components['version']):
                                        if '70' in str(model_components['size_b']):
                                            groq_model = "llama3-70b-8192"
                                        else:
                                            groq_model = "llama3-8b-8192"
                                    else:
                                        groq_model = "llama3-70b-8192"  # Default to Llama 3
                                elif 'qwen' in model_components['family'].lower():
                                    if '3' in display_name or 'qwen3' in canonical_name.lower():
                                        groq_model = "qwen/qwen3-32b"
                                    else:
                                        groq_model = "qwen-qwq-32b"  # For QwQ models
                                
                            # Use the mapped ID if it exists, otherwise use the original selection
                            model_to_use_groq = groq_model if groq_model else current_model
                            
                            print(f"--- Attempting Groq API for {display_name} using model: {model_to_use_groq} ---")
                            response_content = call_groq_api(model_to_use_groq, final_prompt)
                            
                            # Validate response content
                            if response_content and response_content.strip():
                                content_str = response_content.strip()
                                low = content_str.lower()
                                if not (low.startswith("error:") or low.startswith("you have reached") or 
                                        "challenge error" in low or "rate limit" in low or 
                                        "no provider found" in low or "no providers found" in low or 
                                        "context_length_exceeded" in low or "request entity too large" in low or 
                                        "model_not_found" in low or "token" in low):
                                    provider_used = provider_name_for_attempt
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
                        if "rate_limit_exceeded" in error_msg_str or ("token" in error_msg_str and ("limit" in error_msg_str or "quota" in error_msg_str or "rate" in error_msg_str)):
                            error_msg = f"Provider Groq (Direct API) failed due to token rate limit: {e}"
                            provider_errors["Groq (Direct API)"] = f"Token rate limit: {str(e)}"
                        else:
                            error_msg = f"Provider Groq (Direct API) failed: {e}"
                            provider_errors["Groq (Direct API)"] = str(e)
                        print(f"--- {error_msg} ---")
                        response_content = None
                
                # Try Google API
                if GOOGLE_API_KEY and not response_content:
                    try:
                        # Get the Google-specific model name
                        google_model = find_provider_specific_model(current_model, "google")
                        provider_name_for_attempt = "Google AI Studio"
                        attempted_direct_providers.add("google")
                        
                        # Skip this provider if the model is not supported
                        if google_model == "SKIP_PROVIDER" or 'qwen' in display_name.lower() or 'qwen' in current_model.lower():
                            print(f"--- Skipping Google AI for {display_name} (not supported) ---")
                            provider_errors[provider_name_for_attempt] = "Model not supported"
                        else:
                            # Check if it's a Gemini model
                            model_components = parse_model_name(current_model)
                            is_gemini = 'gemini' in model_components['family'].lower()
                            
                            # Map the display name to the actual model ID
                            model_id_mapping = {
                                "Gemini 2.5 Flash": "gemini-2.5-flash",
                                "Gemini 2.0 Flash": "gemini-2.0-flash",
                                "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
                                "Gemini 1.5 Flash": "gemini-1.5-flash",
                                "Gemini 1.0 Pro": "gemini-1.0-pro"
                            }
                            
                            # Use the mapped ID if it exists, otherwise use the original selection
                            if is_gemini:
                                actual_model_id = model_id_mapping.get(display_name)
                                if not actual_model_id:
                                    # Try to determine from model components
                                    if '2.5' in str(model_components['version']):
                                        actual_model_id = "gemini-2.5-flash"
                                    elif '2.0' in str(model_components['version']) or '2' in str(model_components['version']):
                                        if 'lite' in str(model_components['variant']).lower():
                                            actual_model_id = "gemini-2.0-flash-lite"
                                        else:
                                            actual_model_id = "gemini-2.0-flash"
                                    elif '1.5' in str(model_components['version']):
                                        actual_model_id = "gemini-1.5-flash"
                                    else:
                                        actual_model_id = "gemini-2.0-flash"  # Default
                            else:
                                actual_model_id = google_model
                            
                            print(f"--- Attempting Google AI for {display_name} using model: {actual_model_id} ---")
                            response_content = call_google_api(actual_model_id, final_prompt)
                            
                            # Validate response content
                            if response_content and response_content.strip():
                                content_str = response_content.strip()
                                low = content_str.lower()
                                if not (low.startswith("error:") or low.startswith("you have reached") or 
                                        "challenge error" in low or "rate limit" in low or 
                                        "no provider found" in low or "no providers found" in low or 
                                        "context_length_exceeded" in low or "request entity too large" in low or 
                                        "model_not_found" in low or "token" in low):
                                    provider_used = provider_name_for_attempt
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
                        if "resource_exhausted" in error_msg_str or ("token" in error_msg_str and ("limit" in error_msg_str or "quota" in error_msg_str or "rate" in error_msg_str)):
                            error_msg = f"Provider Google AI Studio failed due to token rate limit: {e}"
                            provider_errors["Google AI Studio"] = f"Token rate limit: {str(e)}"
                        else:
                            error_msg = f"Provider Google AI Studio failed: {e}"
                            provider_errors["Google AI Studio"] = str(e)
                        print(f"--- {error_msg} ---")
                        response_content = None
                
                # Try Chutes API
                if CHUTES_API_KEY and not response_content:
                    try:
                        # Get the Chutes-specific model name
                        chutes_model = find_provider_specific_model(current_model, "chutes")
                        provider_name_for_attempt = "Chutes AI"
                        attempted_direct_providers.add("chutes")
                        
                        # Skip this provider if the model is not supported
                        if chutes_model == "SKIP_PROVIDER":
                            print(f"--- Skipping Chutes AI for {display_name} (not supported) ---")
                            provider_errors[provider_name_for_attempt] = "Model not supported"
                        else:
                            # Check if the model is in the Chutes models cache
                            chutes_model_to_try = None
                            if chutes_model in CHUTES_MODELS_CACHE:
                                chutes_model_to_try = chutes_model
                            elif current_model in CHUTES_MODELS_CACHE:
                                chutes_model_to_try = current_model
                            
                            # If model not in cache, try to map it
                            if not chutes_model_to_try:
                                # Check if it's a Llama model and map accordingly
                                model_components = parse_model_name(current_model)
                                if 'llama' in model_components['family'].lower():
                                    if '4' in str(model_components['version']):
                                        if 'scout' in str(model_components['variant']).lower():
                                            chutes_model_to_try = "meta-llama/llama-4-7b-scout"
                                        elif 'maverick' in str(model_components['variant']).lower():
                                            chutes_model_to_try = "meta-llama/llama-4-7b-maverick"
                                        else:
                                            chutes_model_to_try = "meta-llama/llama-4-7b-instruct"
                                    elif '3.3' in str(model_components['version']):
                                        if '70' in str(model_components['size_b']):
                                            chutes_model_to_try = "meta-llama/llama-3.3-70b-instruct"
                                        else:
                                            chutes_model_to_try = "meta-llama/llama-3.3-8b-instruct"
                                    elif '3.1' in str(model_components['version']) or '3' in str(model_components['version']):
                                        if '70' in str(model_components['size_b']):
                                            chutes_model_to_try = "meta-llama/llama-3.1-70b-instruct"
                                        else:
                                            chutes_model_to_try = "meta-llama/llama-3.1-8b-instruct"
                                elif 'claude' in model_components['family'].lower():
                                    if '3.5' in str(model_components['version']):
                                        chutes_model_to_try = "anthropic/claude-3-5-sonnet-20240620"
                                    elif '3' in str(model_components['version']):
                                        if 'opus' in str(model_components['variant']).lower():
                                            chutes_model_to_try = "anthropic/claude-3-opus-20240229"
                                        elif 'sonnet' in str(model_components['variant']).lower():
                                            chutes_model_to_try = "anthropic/claude-3-sonnet-20240229"
                                        elif 'haiku' in str(model_components['variant']).lower():
                                            chutes_model_to_try = "anthropic/claude-3-haiku-20240307"
                                
                            # If we found a model to try, use it
                            if chutes_model_to_try:
                                print(f"--- Attempting Chutes AI for {display_name} using model: {chutes_model_to_try} ---")
                                
                                try:
                                    response_content = asyncio.run(call_chutes_api(chutes_model_to_try, final_prompt))
                                except RuntimeError as e:
                                    if "cannot run nested event loops" in str(e):
                                        # This happens if Flask/Gunicorn/Uvicorn is already running an event loop.
                                        # Try running in the existing loop.
                                        print("--- Detected existing event loop. Running Chutes call within it. ---")
                                        loop = asyncio.get_event_loop()
                                        response_content = loop.run_until_complete(call_chutes_api(chutes_model_to_try, final_prompt))
                                    else:
                                        raise  # Re-raise other runtime errors
                                
                                # Validate response content
                                if response_content and response_content.strip():
                                    content_str = response_content.strip()
                                    low = content_str.lower()
                                    if not (low.startswith("error:") or low.startswith("you have reached") or 
                                            "challenge error" in low or "rate limit" in low or 
                                            "no provider found" in low or "no providers found" in low or 
                                            "context_length_exceeded" in low or "request entity too large" in low or 
                                            "model_not_found" in low or "token" in low):
                                        provider_used = provider_name_for_attempt
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
                            else:
                                print(f"--- Skipping Chutes AI for {display_name} (no matching model found) ---")
                                provider_errors[provider_name_for_attempt] = "No matching model found"
                    except Exception as e:
                        error_msg_str = str(e).lower()
                        if ("rate_limit_exceeded" in error_msg_str and ("tokens" in error_msg_str or "tpm" in error_msg_str)) or \
                           ("request entity too large" in error_msg_str) or \
                           ("context_length_exceeded" in error_msg_str) or \
                           ("token limit" in error_msg_str) or \
                           (("quota" in error_msg_str or "limit" in error_msg_str) and "token" in error_msg_str):
                            error_msg = f"Provider Chutes AI failed due to request size/token limit: {e}"
                            provider_errors["Chutes AI"] = f"Request size/token limit exceeded: {str(e)}"
                        else:
                            error_msg = f"Provider Chutes AI failed: {e}"
                            provider_errors["Chutes AI"] = str(e)
                        print(f"--- {error_msg} ---")
                        response_content = None
                
                # Fallback to G4F only if no direct API worked
                if not response_content:
                    try:
                        print(f"--- Falling back to G4F for {display_name} ---")
                        
                        # Get the list of potential providers (classes or strings)
                        potential_providers = g4f_providers
                        
                        # Skip providers already tried directly
                        filtered_providers = []
                        for p_id in potential_providers:
                            provider_key = p_id.lower() if isinstance(p_id, str) else ""
                            if "groq" in provider_key and "groq" in attempted_direct_providers: 
                                continue
                            if "cerebras" in provider_key and "cerebras" in attempted_direct_providers: 
                                continue
                            if "google" in provider_key and "google" in attempted_direct_providers: 
                                continue
                            if "chutes" in provider_key and "chutes" in attempted_direct_providers: 
                                continue
                            filtered_providers.append(p_id)
                        
                        # Try with specific providers if available
                        if filtered_providers:
                            print(f"--- Trying G4F with specific providers: {filtered_providers} ---")
                            for provider_name in filtered_providers:
                                try:
                                    # Import the provider dynamically
                                    provider_module = __import__('g4f.Provider', fromlist=[provider_name])
                                    provider_class = getattr(provider_module, provider_name, None)
                                    
                                    if provider_class:
                                        print(f"--- Attempting G4F with provider: {provider_name} ---")
                                        response = ChatCompletion.create(
                                            model=current_model,
                                            messages=[{"role": "user", "content": final_prompt}],
                                            provider=provider_class,
                                            max_tokens=max_tokens_for_model
                                        )
                                        if response and response.strip():
                                            content_str = response.strip()
                                            low = content_str.lower()
                                            if not (low.startswith("error:") or low.startswith("you have reached") or 
                                                    "challenge error" in low or "rate limit" in low or 
                                                    "no provider found" in low or "no providers found" in low or 
                                                    "context_length_exceeded" in low or "request entity too large" in low or 
                                                    "model_not_found" in low or "token" in low):
                                                response_content = content_str
                                                provider_used = f"G4F ({provider_name})"
                                                print(f"--- G4F provider {provider_name} succeeded! ---")
                                                break
                                            else:
                                                print(f"--- G4F provider {provider_name} returned error string: {content_str} ---")
                                                provider_errors[f"G4F ({provider_name})"] = content_str
                                except Exception as provider_error:
                                    print(f"--- G4F provider {provider_name} error: {provider_error} ---")
                                    provider_errors[f"G4F ({provider_name})"] = str(provider_error)
                        
                        # If still no response, try with automatic provider selection
                        if not response_content:
                            print(f"--- Trying G4F with automatic provider selection ---")
                            try:
                                response = ChatCompletion.create(
                                    model=current_model,
                                    messages=[{"role": "user", "content": final_prompt}],
                                    max_tokens=max_tokens_for_model
                                )
                                if response and response.strip():
                                    content_str = response.strip()
                                    low = content_str.lower()
                                    if not (low.startswith("error:") or low.startswith("you have reached") or 
                                            "challenge error" in low or "rate limit" in low or 
                                            "no provider found" in low or "no providers found" in low or 
                                            "context_length_exceeded" in low or "request entity too large" in low or 
                                            "model_not_found" in low or "token" in low):
                                        response_content = content_str
                                        provider_used = "G4F (Auto)"
                                        print(f"--- G4F automatic provider selection succeeded! ---")
                                    else:
                                        print(f"--- G4F automatic provider selection returned error string: {content_str} ---")
                                        provider_errors["G4F (Auto)"] = content_str
                                else:
                                    print(f"--- G4F returned empty response ---")
                                    provider_errors["G4F (Auto)"] = "Returned empty response"
                                    response_content = None
                            except Exception as auto_error:
                                print(f"--- G4F automatic provider selection error: {auto_error} ---")
                                provider_errors["G4F (Auto)"] = str(auto_error)
                    except Exception as e:
                        print(f"--- G4F error: {e} ---")
                        provider_errors["G4F"] = str(e)
                        response_content = None
                
                # Final fallback to demo response if all APIs fail
                if not response_content:
                    # Try one last time with a known working model as a last resort
                    try:
                        print("--- Attempting emergency fallback with Mixtral-8x7b via Groq ---")
                        provider_name_for_attempt = "Groq (Emergency Fallback)"
                        
                        if GROQ_API_KEY:
                            # Try direct API call first
                            try:
                                import requests
                                import json
                                
                                print("--- Attempting direct emergency API call to Groq ---")
                                headers = {
                                    "Authorization": f"Bearer {GROQ_API_KEY}",
                                    "Content-Type": "application/json"
                                }
                                
                                data = {
                                    "model": "llama3-70b-8192",  # Use a model that's definitely available on Groq
                                    "messages": [{"role": "user", "content": final_prompt}],
                                    "max_tokens": min(max_tokens_for_model, 8192)  # Respect Groq's limits
                                }
                                
                                response = requests.post(
                                    "https://api.groq.com/openai/v1/chat/completions",
                                    headers=headers,
                                    json=data,
                                    timeout=60
                                )
                                
                                print(f"--- Emergency Groq API response status: {response.status_code} ---")
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    if result and "choices" in result and len(result["choices"]) > 0:
                                        content = result["choices"][0].get("message", {}).get("content", "")
                                        if content:
                                            content_str = content.strip()
                                            low = content_str.lower()
                                            if not (low.startswith("error:") or low.startswith("you have reached") or 
                                                    "challenge error" in low or "rate limit" in low or 
                                                    "no provider found" in low or "no providers found" in low or 
                                                    "context_length_exceeded" in low or "request entity too large" in low or 
                                                    "model_not_found" in low or "token" in low):
                                                print("--- Direct emergency Groq API call successful ---")
                                                response_content = content_str
                                                provider_used = provider_name_for_attempt
                                                print("--- Emergency fallback successful ---")
                                            else:
                                                print(f"--- Emergency Groq API returned error string: {content_str} ---")
                                                provider_errors[provider_name_for_attempt] = content_str
                                        else:
                                            print("--- Emergency Groq API returned empty content ---")
                                            provider_errors[provider_name_for_attempt] = "Returned empty content"
                                    else:
                                        print(f"--- Emergency Groq API returned unexpected JSON structure: {result} ---")
                                        provider_errors[provider_name_for_attempt] = f"Unexpected JSON structure: {str(result)[:100]}"
                                else:
                                    error_text = response.text
                                    print(f"--- Emergency Groq API failed with status {response.status_code}: {error_text[:200]} ---")
                                    provider_errors[provider_name_for_attempt] = f"Status {response.status_code}: {error_text[:100]}"
                            
                            except Exception as direct_emergency_error:
                                print(f"--- Direct emergency Groq API error: {direct_emergency_error}, falling back to g4f ---")
                                provider_errors[provider_name_for_attempt] = str(direct_emergency_error)
                            
                            # If direct API call failed, try g4f
                            if not response_content and GROQ_PROVIDER_CLASS:
                                provider_name_for_attempt = "Groq (G4F Emergency Fallback)"
                                print("--- Falling back to g4f for emergency Groq API call ---")
                                try:
                                    emergency_response = ChatCompletion.create(
                                        model="llama3-70b-8192",  # Use a model that's definitely available on Groq
                                        messages=[{"role": "user", "content": final_prompt}],
                                        provider=GROQ_PROVIDER_CLASS,
                                        api_key=GROQ_API_KEY,
                                        max_tokens=min(max_tokens_for_model, 8192)  # Respect Groq's limits
                                    )
                                    if emergency_response and emergency_response.strip():
                                        content_str = emergency_response.strip()
                                        low = content_str.lower()
                                        if not (low.startswith("error:") or low.startswith("you have reached") or 
                                                "challenge error" in low or "rate limit" in low or 
                                                "no provider found" in low or "no providers found" in low or 
                                                "context_length_exceeded" in low or "request entity too large" in low or 
                                                "model_not_found" in low or "token" in low):
                                            response_content = content_str
                                            provider_used = provider_name_for_attempt
                                            print("--- G4F emergency fallback successful ---")
                                        else:
                                            print(f"--- G4F emergency fallback returned error string: {content_str} ---")
                                            provider_errors[provider_name_for_attempt] = content_str
                                    else:
                                        print("--- G4F emergency fallback returned empty response ---")
                                        provider_errors[provider_name_for_attempt] = "Returned empty response"
                                except Exception as g4f_emergency_error:
                                    print(f"--- G4F emergency fallback error: {g4f_emergency_error} ---")
                                    provider_errors[provider_name_for_attempt] = str(g4f_emergency_error)
                    except Exception as e:
                        print(f"--- Emergency fallback failed: {e} ---")
                        provider_errors["Emergency Fallback"] = str(e)
                
                # If still no response, show error message
                if not response_content:
                    # Create a more informative error message with specific errors
                    error_details = []
                    for provider, error in provider_errors.items():
                        error_details.append(f"{provider}: {error}")
                    
                    error_details_str = "\n".join(error_details)
                    
                    response_content = f"""I apologize, but I'm unable to process your request at the moment.

I tried to use various API providers for {display_name}, but none were successful.

This could be due to:
1. The selected model ({display_name}) may not be available through any configured provider
2. Temporary API service disruptions
3. Rate limiting or quota issues with the API keys

Please try:
- Selecting a different model from the dropdown
- Waiting a few minutes and trying again
- Checking your API keys if you're the administrator

Technical details:
{error_details_str}"""
                    provider_used = "Error - No Provider Available"

                # For reasoning models, try to ensure complete response
                if response_content and any(keyword in display_name.lower() for keyword in ['qwq', 'qwen3', 'reasoning', 'thinking']):
                    print(f"--- Checking response completeness for reasoning model: {display_name} ---")
                    
                    if not is_response_complete(response_content, display_name):
                        print("--- Response appears incomplete, attempting completion ---")
                        
                        # Try to get a continuation if the response is incomplete
                        if is_continuation:
                            # If this was already a continuation request, use the response as-is
                            print("--- This was already a continuation request, using response as-is ---")
                        else:
                            # Try to get completion by making a continuation request
                            try:
                                completion_prompt = f"""Please continue and complete your previous response. Here's what you said:

{response_content}

Please continue from where you left off and provide the complete answer."""
                                
                                # Try to get completion (simplified, just one retry)
                                completion_response = None
                                
                                # Use the same provider that worked for the original response
                                if provider_used.startswith("Cerebras") and CEREBRAS_API_KEY:
                                    completion_response = call_cerebras_api(current_model, completion_prompt)
                                elif provider_used.startswith("Groq") and GROQ_API_KEY:
                                    completion_response = call_groq_api(current_model, completion_prompt)
                                elif provider_used.startswith("Chutes") and CHUTES_API_KEY:
                                    completion_response = call_chutes_api(current_model, completion_prompt)
                                
                                if completion_response and completion_response.strip():
                                    print("--- Successfully obtained response completion ---")
                                    response_content = response_content + "\n\n" + completion_response.strip()
                                else:
                                    print("--- Could not obtain completion, using original response ---")
                                    
                            except Exception as completion_error:
                                print(f"--- Error getting completion: {completion_error} ---")

                # Add assistant response
                current_time = datetime.now().isoformat()
                current_chat["history"].append({
                    "role": "assistant",
                    "content": response_content,
                    "model": current_model,
                    "provider": provider_used,
                    "timestamp": current_time
                })
                
                current_chat["last_modified"] = current_time
                
                # Auto-name chat if it's new
                if current_chat.get("name") == "New Chat" and len(current_chat["history"]) >= 2:
                    first_user_msg = next((msg["content"] for msg in current_chat["history"] if msg["role"] == "user"), None)
                    if first_user_msg:
                        clean_prompt = ''.join(c for c in ' '.join(first_user_msg.split()[:6]) if c.isalnum() or c.isspace()).strip()
                        timestamp_str = datetime.fromisoformat(current_time).strftime("%b %d, %I:%M%p")
                        chat_name = f"{clean_prompt[:30]}... ({timestamp_str})" if clean_prompt else f"Chat ({timestamp_str})"
                        current_chat["name"] = chat_name
                
                save_chats(chats)
                
            except Exception as e:
                print(f"--- Error processing message: {e} ---")
                # Add error response
                current_time = datetime.now().isoformat()
                current_chat["history"].append({
                    "role": "assistant",
                    "content": f"Error processing your request: {str(e)}",
                    "model": current_model,
                    "provider": "Error Handler",
                    "timestamp": current_time
                })
                save_chats(chats)
            
            return redirect(url_for('index'))
    
    # Prepare chat history HTML
    history_html = ""
    for msg in current_chat.get("history", []):
        role_display = html.escape(msg["role"].title())
        timestamp_str = msg.get("timestamp", "")
        try:
            timestamp_display = datetime.fromisoformat(timestamp_str).strftime("%I:%M:%S %p") if timestamp_str else "No Time"
        except ValueError:
            timestamp_display = "Invalid Time"
        
        content_display = html.escape(msg["content"])
        
        # Create message metadata
        metadata = []
        if msg["role"] == "assistant":
            if msg.get('model', 'N/A') != 'N/A':
                metadata.append(f"Model: {html.escape(msg.get('model', 'N/A'))}")
            if msg.get('provider', 'N/A') != 'N/A':
                metadata.append(f"Provider: {html.escape(msg.get('provider', 'N/A'))}")
        
        metadata_display = f"<small>{' | '.join(metadata)}</small>" if metadata else ""
        
        # Determine message class based on role
        message_class = "user-message" if msg['role'] == 'user' else "assistant-message"
        
        history_html += f'''<div class="message {message_class}">
                             <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <b>{role_display}</b>
                                <small>{timestamp_display}</small>
                             </div>
                             {f'<div style="margin-bottom: 4px; font-size: 0.85em; color: #666;">{metadata_display}</div>' if metadata_display else ''}
                             <div style="white-space: pre-wrap; word-wrap: break-word;">{content_display}</div>
                           </div>'''
    
    # Prepare model dropdown options with performance scores
    model_options_html = ''
    seen_display_names = set()
    
    for model_name, provider_count, intelligence_index, response_time_s in available_models_sorted_list:
        if not model_name or not model_name.strip():
            continue
        
        # Apply display name mapping to remove duplicates
        display_name = MODEL_DISPLAY_NAME_MAP.get(model_name, model_name).strip()
        if not display_name or display_name in seen_display_names:
            continue
        
        seen_display_names.add(display_name)
        
        # Check if this model is selected
        current_display = MODEL_DISPLAY_NAME_MAP.get(current_model, current_model)
        is_selected = (model_name == current_model or display_name == current_display)
        selected_attr = "selected" if is_selected else ""
        
        # Format the display text with performance info
        if intelligence_index > 0 and response_time_s != float('inf'):
            perf_str = f" ({intelligence_index:.0f}, {response_time_s:.2f}s)"
        elif intelligence_index > 0:
            perf_str = f" ({intelligence_index:.0f}, N/A)"
        elif response_time_s != float('inf'):
            perf_str = f" (N/A, {response_time_s:.2f}s)"
        else:
            perf_str = " (N/A)"
        
        model_options_html += f'<option value="{model_name}" {selected_attr}>{display_name}{perf_str}</option>'
    
    # Navigation and controls
    nav_links_html = f'''
        <a href="/saved_chats" style="border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333;">Saved Chats</a>
        <form method="post" style="display: inline;">
            <button type="submit" name="new_chat" value="1" style="border: 1px solid #ccc; border-radius: 4px; background-color: #e7e7e7; color: #333; cursor: pointer;">New Chat</button>
        </form>
    '''
    
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
    <title>Enhanced LLM Chat Interface</title>
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

@app.route('/saved_chats')
def saved_chats():
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
        chats[chat_id]['last_modified'] = datetime.now().isoformat()
        save_chats(chats)
        
        if 'model' in chats[chat_id]:
            session['user_selected_model'] = chats[chat_id]['model']
            print(f"--- Loaded model from chat: {chats[chat_id]['model']} ---")
            
        return redirect(url_for('index'))
    else:
        return redirect(url_for('saved_chats'))

# --- API Call Functions ---

def find_provider_specific_model(model_name, provider_name):
    """
    Find a provider-specific model ID for a given model name and provider.
    
    Args:
        model_name (str): The model name to find a provider-specific ID for
        provider_name (str): The provider name to find a model ID for
        
    Returns:
        str: The provider-specific model ID, or the original model name if none is found
    """
    # First check the static mapping using the exact model name
    if model_name in PROVIDER_MODEL_MAP and provider_name.lower() in PROVIDER_MODEL_MAP[model_name]:
        print(f"--- Found exact model mapping for {model_name} on {provider_name}: {PROVIDER_MODEL_MAP[model_name][provider_name.lower()]} ---")
        return PROVIDER_MODEL_MAP[model_name][provider_name.lower()]
    
    # Then check using the display name mapping
    display_name = MODEL_DISPLAY_NAME_MAP.get(model_name, model_name)
    if display_name in PROVIDER_MODEL_MAP and provider_name.lower() in PROVIDER_MODEL_MAP[display_name]:
        print(f"--- Found display name mapping for {model_name} ({display_name}) on {provider_name}: {PROVIDER_MODEL_MAP[display_name][provider_name.lower()]} ---")
        return PROVIDER_MODEL_MAP[display_name][provider_name.lower()]
        
    # Also check if any key in PROVIDER_MODEL_MAP contains the model name as a substring
    for key in PROVIDER_MODEL_MAP:
        if model_name.lower() in key.lower() or key.lower() in model_name.lower():
            if provider_name.lower() in PROVIDER_MODEL_MAP[key]:
                print(f"--- Found partial match mapping for {model_name} using {key} on {provider_name}: {PROVIDER_MODEL_MAP[key][provider_name.lower()]} ---")
                return PROVIDER_MODEL_MAP[key][provider_name.lower()]
    
    # If not found in static mapping, try to find a matching model using our parser
    provider_models = []
    
    # Get all models for this provider from the performance cache
    for entry in PROVIDER_PERFORMANCE_CACHE:
        entry_provider = entry.get('provider_name_scraped', '').strip().lower()
        if provider_name.lower() in entry_provider:
            provider_models.append(entry.get('model_name_scraped', '').strip())
    
    # Add provider-specific model caches
    if provider_name.lower() == 'chutes':
        provider_models.extend(CHUTES_MODELS_CACHE)
    elif provider_name.lower() == 'groq':
        provider_models.extend(GROQ_MODELS_CACHE)
    
    # Parse the input model name to get its components
    model_components = parse_model_name(model_name)
    
    # Find the best matching model
    best_match = None
    for provider_model in provider_models:
        if are_same_model(model_name, provider_model):
            best_match = provider_model
            break
    
    if best_match:
        print(f"--- Found matching model for {model_name} on {provider_name}: {best_match} ---")
        return best_match
    
    # If no exact match found, try to find a similar model based on family and size
    if not best_match:
        # Special handling for specific model families
        if 'llama' in model_components['family'].lower():
            # For Llama models
            if provider_name.lower() == 'groq':
                # Groq has specific mappings for Llama models
                if '3.3' in str(model_components['version']) and '70' in str(model_components['size_b']):
                    return "llama3-70b-8192"
                elif '3.3' in str(model_components['version']) and '8' in str(model_components['size_b']):
                    return "llama3-8b-8192"
                elif '3.1' in str(model_components['version']):
                    return "llama3-70b-8192"  # Default to 70B for Llama 3.1
                elif '3' in str(model_components['version']):
                    return "llama3-8b-8192"  # Default to 8B for Llama 3
                elif '4' in str(model_components['version']) and 'maverick' in str(model_components['variant']).lower():
                    return "llama4-7b-8192"  # Llama 4 Maverick
                elif '4' in str(model_components['version']) and 'scout' in str(model_components['variant']).lower():
                    return "llama4-7b-8192"  # Llama 4 Scout
                elif '4' in str(model_components['version']):
                    return "llama4-7b-8192"  # Default for Llama 4
                elif '2' in str(model_components['version']):
                    return "llama2-70b-4096"  # Default for Llama 2
            
            elif provider_name.lower() == 'chutes':
                # Chutes has specific mappings for Llama models
                if '3.3' in str(model_components['version']) and '70' in str(model_components['size_b']):
                    return "meta-llama/llama-3.3-70b-instruct"
                elif '3.3' in str(model_components['version']) and '8' in str(model_components['size_b']):
                    return "meta-llama/llama-3.3-8b-instruct"
                elif '3.1' in str(model_components['version']):
                    return "meta-llama/llama-3.1-70b-instruct"  # Default to 70B for Llama 3.1
                elif '3' in str(model_components['version']):
                    return "meta-llama/llama-3-8b-instruct"  # Default to 8B for Llama 3
                elif '4' in str(model_components['version']) and 'maverick' in str(model_components['variant']).lower():
                    return "meta-llama/llama-4-7b-maverick"  # Llama 4 Maverick
                elif '4' in str(model_components['version']) and 'scout' in str(model_components['variant']).lower():
                    return "meta-llama/llama-4-7b-scout"  # Llama 4 Scout
                elif '4' in str(model_components['version']):
                    return "meta-llama/llama-4-7b-instruct"  # Default for Llama 4
                elif '2' in str(model_components['version']):
                    return "meta-llama/llama-2-70b-chat"  # Default for Llama 2
            
            elif provider_name.lower() == 'cerebras':
                # Cerebras doesn't have Llama models, so return a default
                return "cerebras/Cerebras-GPT-13B-v1.0"
            
            elif provider_name.lower() == 'google':
                # Google doesn't have Llama models, but we'll return SKIP_PROVIDER for specific Llama models
                # to prevent them from using Google Vertex API
                if ('4' in str(model_components['version']) and 
                    ('maverick' in str(model_components['variant']).lower() or 
                     'scout' in str(model_components['variant']).lower())):
                    return "SKIP_PROVIDER"  # Skip Google for Llama 4 Maverick and Scout
                elif '3.3' in str(model_components['version']) and '70' in str(model_components['size_b']):
                    return "SKIP_PROVIDER"  # Skip Google for Llama 3.3 70B
                else:
                    return "gemini-2.0-flash"
        
        # Special handling for Gemini models
        elif 'gemini' in model_components['family'].lower():
            if provider_name.lower() == 'google':
                if '2.5' in str(model_components['version']):
                    return "gemini-2.5-flash"
                elif '2.0' in str(model_components['version']) or '2' in str(model_components['version']):
                    return "gemini-2.0-flash"
                elif '1.5' in str(model_components['version']) or '1' in str(model_components['version']):
                    return "gemini-1.5-flash"
                else:
                    return "gemini-2.0-flash"  # Default to latest version
            else:
                # Other providers don't have Gemini models
                return model_name
        
        # Special handling for Claude models
        elif 'claude' in model_components['family'].lower():
            if provider_name.lower() == 'chutes':
                if '3.5' in str(model_components['version']) or '3' in str(model_components['version']):
                    return "anthropic/claude-3-5-sonnet-20240620"
                elif '2.1' in str(model_components['version']) or '2' in str(model_components['version']):
                    return "anthropic/claude-2.1"
                else:
                    return "anthropic/claude-3-5-sonnet-20240620"  # Default to latest version
            else:
                # Other providers don't have Claude models
                return model_name
        
        # Special handling for Qwen models
        elif 'qwen' in model_components['family'].lower():
            # Don't try to map Qwen models to other providers
            if provider_name.lower() != 'g4f':
                # For now, we don't have direct API access to Qwen models
                # Return a special marker to indicate we should skip this provider
                return "SKIP_PROVIDER"
            return model_name
    
    # If no match found, return the original model name
    return model_name

def call_cerebras_api(model, prompt):
    """Call Cerebras API directly."""
    try:
        print(f"--- DEBUG: Starting Cerebras API call for model: {model} ---")
        print(f"--- DEBUG: CEREBRAS_API_KEY exists: {bool(CEREBRAS_API_KEY)} ---")
        print(f"--- DEBUG: CEREBRAS_PROVIDER_CLASS exists: {bool(CEREBRAS_PROVIDER_CLASS)} ---")
        
        if CEREBRAS_PROVIDER_CLASS:
            # Find the provider-specific model ID
            cerebras_model = find_provider_specific_model(model, "cerebras")
            print(f"--- DEBUG: After find_provider_specific_model, cerebras_model: {cerebras_model} ---")
            
            if cerebras_model != model:
                print(f"--- Using Cerebras-specific model ID: {cerebras_model} (for {model}) ---")
            
            # Default to a known working model if needed
            if cerebras_model == "SKIP_PROVIDER" or not cerebras_model:
                print(f"--- Model {model} not supported by Cerebras, using default model ---")
                cerebras_model = "cerebras/Cerebras-GPT-13B-v1.0"
            
            # Ensure the model name is in the correct format for Cerebras
            if not cerebras_model.startswith("cerebras/") and not "/" in cerebras_model:
                cerebras_model = f"cerebras/{cerebras_model}"
                print(f"--- Adjusted Cerebras model name to: {cerebras_model} ---")
            
            print(f"--- Calling Cerebras API with model: {cerebras_model} ---")
            
            # Try direct API call without g4f
            try:
                import requests
                import json
                
                print("--- DEBUG: Attempting direct API call to Cerebras ---")
                headers = {
                    "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                # Get max tokens for this specific model
                model_max_tokens = get_max_tokens_for_model(cerebras_model)
                
                data = {
                    "model": cerebras_model.replace("cerebras/", ""),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": model_max_tokens
                }
                
                print(f"--- DEBUG: Cerebras API request data: {json.dumps(data)} ---")
                
                response = requests.post(
                    "https://api.cerebras.ai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                print(f"--- DEBUG: Cerebras API response status: {response.status_code} ---")
                
                if response.status_code == 200:
                    result = response.json()
                    if result and "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0].get("message", {}).get("content", "")
                        if content:
                            print("--- DEBUG: Direct Cerebras API call successful ---")
                            return content.strip()
                else:
                    print(f"--- DEBUG: Cerebras API error response: {response.text} ---")
            
            except Exception as direct_api_error:
                print(f"--- DEBUG: Direct Cerebras API error: {direct_api_error}, falling back to g4f ---")
            
            # Fallback to g4f
            print("--- DEBUG: Falling back to g4f for Cerebras API call ---")
            response = ChatCompletion.create(
                model=cerebras_model,
                messages=[{"role": "user", "content": prompt}],
                provider=CEREBRAS_PROVIDER_CLASS,
                api_key=CEREBRAS_API_KEY,
                max_tokens=get_max_tokens_for_model(cerebras_model)
            )
            print(f"--- DEBUG: g4f Cerebras response: {bool(response)} ---")
            return response.strip() if response and response.strip() else None
        else:
            print("--- DEBUG: CEREBRAS_PROVIDER_CLASS is None, skipping Cerebras API call ---")
            return None
    except Exception as e:
        print(f"--- Cerebras API error: {e} ---")
        print(f"--- DEBUG: Exception type: {type(e).__name__} ---")
        import traceback
        print(f"--- DEBUG: Traceback: {traceback.format_exc()} ---")
        return None

def call_groq_api(model, prompt):
    """Call Groq API directly."""
    try:
        print(f"--- DEBUG: Starting Groq API call for model: {model} ---")
        print(f"--- DEBUG: GROQ_API_KEY exists: {bool(GROQ_API_KEY)} ---")
        print(f"--- DEBUG: GROQ_PROVIDER_CLASS exists: {bool(GROQ_PROVIDER_CLASS)} ---")
        
        if GROQ_PROVIDER_CLASS:
            # Find the provider-specific model ID
            groq_model = find_provider_specific_model(model, "groq")
            print(f"--- DEBUG: After find_provider_specific_model, groq_model: {groq_model} ---")
            
            if groq_model != model:
                print(f"--- Using Groq-specific model ID: {groq_model} (for {model}) ---")
            
            # Default to a known working model if needed
            if groq_model == "SKIP_PROVIDER" or not groq_model:
                print(f"--- Model {model} not supported by Groq, using default model ---")
                
                # Check if it's a Llama model and choose appropriate fallback
                model_components = parse_model_name(model)
                if 'llama' in model_components['family'].lower():
                    if '4' in str(model_components['version']):
                        groq_model = "llama4-7b-8192"
                    elif '3.3' in str(model_components['version']) or '3' in str(model_components['version']):
                        groq_model = "llama3-70b-8192"
                    else:
                        groq_model = "llama2-70b-4096"
                else:
                    # Default to a generally available model
                    groq_model = "mixtral-8x7b-32768"
            
            print(f"--- Calling Groq API with model: {groq_model} ---")
            
            # Try direct API call without g4f
            try:
                import requests
                import json
                
                print("--- DEBUG: Attempting direct API call to Groq ---")
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                # Get max tokens for this specific model, respecting Groq's limits
                model_max_tokens = min(get_max_tokens_for_model(groq_model), 8192)
                
                data = {
                    "model": groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": model_max_tokens
                }
                
                print(f"--- DEBUG: Groq API request data: {json.dumps(data)} ---")
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                print(f"--- DEBUG: Groq API response status: {response.status_code} ---")
                
                if response.status_code == 200:
                    result = response.json()
                    if result and "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0].get("message", {}).get("content", "")
                        if content:
                            print("--- DEBUG: Direct Groq API call successful ---")
                            return content.strip()
                else:
                    print(f"--- DEBUG: Groq API error response: {response.text} ---")
            
            except Exception as direct_api_error:
                print(f"--- DEBUG: Direct Groq API error: {direct_api_error}, falling back to g4f ---")
            
            # Fallback to g4f
            print("--- DEBUG: Falling back to g4f for Groq API call ---")
            response = ChatCompletion.create(
                model=groq_model,
                messages=[{"role": "user", "content": prompt}],
                provider=GROQ_PROVIDER_CLASS,
                api_key=GROQ_API_KEY,
                max_tokens=min(get_max_tokens_for_model(groq_model), 8192)
            )
            print(f"--- DEBUG: g4f Groq response: {bool(response)} ---")
            return response.strip() if response and response.strip() else None
        else:
            print("--- DEBUG: GROQ_PROVIDER_CLASS is None, skipping Groq API call ---")
            return None
    except Exception as e:
        print(f"--- Groq API error: {e} ---")
        print(f"--- DEBUG: Exception type: {type(e).__name__} ---")
        import traceback
        print(f"--- DEBUG: Traceback: {traceback.format_exc()} ---")
        return None

def call_google_api(model, prompt):
    """Call Google AI Studio API."""
    try:
        # Check if this is a Llama model that should skip Google
        model_components = parse_model_name(model)
        if 'llama' in model_components['family'].lower():
            if ('4' in str(model_components['version']) and 
                ('maverick' in str(model_components['variant']).lower() or 
                 'scout' in str(model_components['variant']).lower())):
                print(f"--- Skipping Google API for {model} (Llama 4 Scout/Maverick not supported) ---")
                return None
            elif '3.3' in str(model_components['version']) and '70' in str(model_components['size_b']):
                print(f"--- Skipping Google API for {model} (Llama 3.3 70B not supported) ---")
                return None
        
        # If it's a Gemini model, use it directly
        if 'gemini' in model_components['family'].lower():
            # Try direct API call first
            try:
                import google.generativeai as genai
                genai.configure(api_key=GOOGLE_API_KEY)
                
                # Determine the correct Gemini model ID
                gemini_model = None
                if '2.5' in str(model_components['version']):
                    gemini_model = "gemini-2.5-flash"
                elif '2.0' in str(model_components['version']) or '2' in str(model_components['version']):
                    if 'lite' in str(model_components['variant']).lower():
                        gemini_model = "gemini-2.0-flash-lite"
                    else:
                        gemini_model = "gemini-2.0-flash"
                elif '1.5' in str(model_components['version']) or '1' in str(model_components['version']):
                    gemini_model = "gemini-1.5-flash"
                else:
                    gemini_model = "gemini-2.0-flash"  # Default
                
                print(f"--- Using Gemini model directly: {gemini_model} ---")
                model_instance = genai.GenerativeModel(gemini_model)
                response = model_instance.generate_content(prompt)
                
                if response and hasattr(response, 'text') and response.text:
                    return response.text.strip()
            except Exception as gemini_error:
                print(f"--- Direct Gemini API error: {gemini_error}, falling back to standard method ---")
        
        # Standard method for non-Gemini models or if direct call failed
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Find the provider-specific model ID
        google_model = find_provider_specific_model(model, "google")
        
        # Skip if the model is explicitly marked to skip
        if google_model == "SKIP_PROVIDER":
            print(f"--- Skipping Google API for {model} (marked as SKIP_PROVIDER) ---")
            return None
        
        # Fallback mapping for Google models
        model_mapping = {
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.0 Flash": "gemini-2.0-flash",
            "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 1.0 Pro": "gemini-1.0-pro"
        }
        
        # If we didn't find a match using our parser, try the static mapping
        if google_model == model:
            canonical_name = get_canonical_model_name(model)
            google_model = model_mapping.get(canonical_name, "gemini-2.0-flash")
        
        # Final check to ensure we're not using Google for Llama models
        if 'llama' in model.lower() and 'gemini' in google_model.lower():
            print(f"--- Skipping Google API for {model} (Llama models should not use Gemini) ---")
            return None
            
        print(f"--- Using Google model: {google_model} (for {model}) ---")
        
        model_instance = genai.GenerativeModel(google_model)
        response = model_instance.generate_content(prompt)
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        return None
    except Exception as e:
        print(f"--- Google API error: {e} ---")
        return None

async def call_chutes_api(model, prompt):
    """Call Chutes AI API."""
    try:
        # Find the provider-specific model ID
        chutes_model = find_provider_specific_model(model, "chutes")
        
        if chutes_model != model:
            print(f"--- Using Chutes-specific model ID: {chutes_model} (for {model}) ---")
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {CHUTES_API_KEY}",
                "Content-Type": "application/json"
            }
            
            body = {
                "model": chutes_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": get_max_tokens_for_model(chutes_model),
                "stream": False
            }
            
            async with session.post(f"{CHUTES_API_URL}/chat/completions", 
                                   headers=headers, json=body, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'choices' in data and data['choices']:
                        content = data['choices'][0]['message']['content']
                        return content.strip() if content else None
        return None
    except Exception as e:
        print(f"--- Chutes API error: {e} ---")
        return None

if __name__ == '__main__':
    # Initialize model cache
    initialize_model_cache()
    
    print("--- [STARTUP] Flask application ready ---")
    print(f"--- Available models: {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)} ---")
    print("--- Starting Flask development server... ---")
    app.run(host='0.0.0.0', port=5000, debug=True)