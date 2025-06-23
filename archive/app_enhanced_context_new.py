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
    
    # Qwen 3 32B variants (reasoning and non-reasoning)
    "qwen-3-32b": "Qwen 3 32B",
    "qwen-3-32b-reasoning": "Qwen 3 32B (Reasoning)",
    "qwen-3-32b-instruct": "Qwen 3 32B",
    "Qwen 3 32B (Reasoning)": "Qwen 3 32B (Reasoning)",
    "Qwen3 32B (Reasoning)": "Qwen 3 32B (Reasoning)",
    "qwen3-32b-reasoning": "Qwen 3 32B (Reasoning)",
    
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
        "cerebras": "SKIP_PROVIDER",  # Cerebras only has reasoning version
        "groq": "qwen/qwen3-32b-instruct",
        "google": "SKIP_PROVIDER",
        "chutes": "Qwen/Qwen3-32B"  # Chutes has non-reasoning version
    },
    "Qwen 3 32B (Reasoning)": {
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

def build_smart_context_messages(chat_history, current_message, model_name):
    # Smart context management for chat history
    context_window = 8192  # Default
    if 'gemini' in model_name.lower():
        context_window = 1000000
    elif 'llama-4' in model_name.lower():
        context_window = 128000
    elif 'qwen' in model_name.lower():
        context_window = 32768  # Qwen models have larger context
    
    max_tokens = int(context_window * 0.6)
    
    # If no current message, just use chat history
    if not current_message:
        if not chat_history:
            return []
        
        messages = []
        used_tokens = 0
        
        for msg in reversed(chat_history):
            msg_content = msg.get('content', '')
            msg_tokens = len(msg_content) // 4
            
            if used_tokens + msg_tokens > max_tokens:
                break
            
            messages.insert(0, {
                'role': msg.get('role', 'user'),
                'content': msg_content
            })
            used_tokens += msg_tokens
        
        print(f'Smart context: Using {len(messages)} messages from history, ~{used_tokens} tokens for {model_name}')
        return messages
    
    # Original logic for when we have a current message
    current_tokens = len(current_message) // 4
    
    if not chat_history:
        return [{'role': 'user', 'content': current_message}]
    
    messages = [{'role': 'user', 'content': current_message}]
    used_tokens = current_tokens
    
    for msg in reversed(chat_history):
        msg_content = msg.get('content', '')
        msg_tokens = len(msg_content) // 4
        
        if used_tokens + msg_tokens > max_tokens:
            break
        
        messages.insert(0, {
            'role': msg.get('role', 'user'),
            'content': msg_content
        })
        used_tokens += msg_tokens
    
    print(f'Smart context: Using {len(messages)} messages, ~{used_tokens} tokens for {model_name}')
    return messages



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
        "Qwen 3 32B (Reasoning)": {"family": "qwen", "version": "3", "variant": "reasoning", "size_b": 32.0},
        "Qwen3 32B (Reasoning)": {"family": "qwen", "version": "3", "variant": "reasoning", "size_b": 32.0},
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
        r'(reasoning)',  # Add reasoning pattern first for priority
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
        variant = components['variant'].capitalize()
        # Special formatting for reasoning variant
        if variant.lower() == 'reasoning':
            # We'll add this at the end in parentheses
            pass
        else:
            parts.append(variant)
    
    if components['size_b'] > 0:
        # Check if size_b is a float before calling is_integer()
        if isinstance(components['size_b'], float):
            parts.append(f"{int(components['size_b']) if components['size_b'].is_integer() else components['size_b']}B")
        else:
            parts.append(f"{components['size_b']}B")
    
    # Add reasoning variant at the end in parentheses
    base_name = " ".join(parts)
    if components['variant'] and components['variant'].lower() == 'reasoning':
        return f"{base_name} (Reasoning)"
    
    return base_name

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
    
    # Tier 1: Best direct APIs (Groq/Cerebras)
    if 'groq' in provider_lower:
        return 1
    elif 'cerebras' in provider_lower:
        return 2
    
    # Tier 2: Good direct APIs (Chutes/Google)
    elif 'chutes' in provider_lower:
        return 3
    elif 'google' in provider_lower:
        return 4
    
    # Tier 3: Other providers (accessed through G4F or with limitations)
    elif 'sambanova' in provider_lower:
        return 20  # Limited free credits
    elif 'deepinfra' in provider_lower:
        return 21  # Limited free access
    elif any(p in provider_lower for p in ['openrouter', 'together', 'cohere']):
        return 25
    
    # Tier 4: G4F providers (lowest priority)
    elif 'g4f' in provider_lower:
        return 50
    
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

