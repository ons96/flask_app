#!/usr/bin/env python3
"""
Test script for web search functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the web search functions from the main app
from app_fixed_comprehensive_final_new import perform_web_search

def test_web_search():
    """Test the web search functionality"""
    print("Testing web search functionality...")
    
    # Test queries
    test_queries = [
        "current weather",
        "latest news about AI",
        "python programming tutorial",
        "what is 2+2"  # This shouldn't trigger smart search
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        try:
            results = perform_web_search(query, max_results=3)
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   Snippet: {result['snippet'][:100]}...")
                    print(f"   Link: {result['link']}")
            else:
                print("No results found")
        except Exception as e:
            print(f"Error during search: {e}")
    
    print("\n--- Web search test completed ---")

if __name__ == "__main__":
    test_web_search()