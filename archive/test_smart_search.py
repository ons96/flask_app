#!/usr/bin/env python3
"""
Test script for smart search logic
"""

import re

def test_smart_search_logic():
    """Test the smart search keyword detection logic"""
    
    # Time-sensitive keywords
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
    
    all_keywords = time_keywords + question_patterns + tech_keywords + sports_keywords + finance_keywords
    
    # Test queries - should trigger smart search
    should_search_queries = [
        "what is the latest news about AI",
        "current weather in New York",
        "who is the current president of the US",
        "latest version of Python",
        "what happened today in the stock market",
        "Tesla stock price now",
        "breaking news about OpenAI",
        "current inflation rate",
        "latest iPhone release",
        "what's the score of the game today",
        "current population of Tokyo",
        "recent updates on ChatGPT"
    ]
    
    # Test queries - should NOT trigger smart search
    should_not_search_queries = [
        "what is 2+2",
        "explain machine learning",
        "how to write a Python function",
        "what is the capital of France",
        "explain quantum physics",
        "how to cook pasta",
        "what is the meaning of life",
        "tell me a joke"
    ]
    
    print("Testing Smart Search Logic\n")
    
    # Common exclusion patterns for both test cases
    exclusion_patterns = [
        r'\bwhat is \d+[\+\-\*\/]\d+',  # Basic math like "what is 2+2"
        r'\bwhat is the capital of',     # Geography basics
        r'\bwhat is the meaning of life', # Philosophical questions
        r'\bexplain (how to|what is)',   # General explanations
        r'\bhow to (write|code|program|make|cook|do)', # How-to questions
        r'\bwhat does .* mean',          # Definition questions
        r'\btell me (a joke|about)',     # Entertainment requests
    ]
    
    def should_trigger_search(query_text):
        prompt_lower = query_text.lower()
        should_search = False
        
        # Check if query matches exclusion patterns first
        is_excluded = any(re.search(pattern, prompt_lower) for pattern in exclusion_patterns)
        
        if not is_excluded:
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
        
        return should_search
    
    print("=== Queries that SHOULD trigger search ===")
    for query in should_search_queries:
        should_search = should_trigger_search(query)
        status = "✓ SEARCH" if should_search else "✗ NO SEARCH"
        print(f"{status}: {query}")
    
    print("\n=== Queries that should NOT trigger search ===")
    for query in should_not_search_queries:
        should_search = should_trigger_search(query)
        status = "✗ NO SEARCH" if not should_search else "⚠ SEARCH (unexpected)"
        print(f"{status}: {query}")

if __name__ == "__main__":
    test_smart_search_logic()