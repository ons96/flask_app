#!/usr/bin/env python3
"""
Test script for Brave Search API functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the web search functions from the main app
from app_fixed_comprehensive_final_new import perform_brave_search, BRAVE_API_KEY

def test_brave_search():
    """Test the Brave Search API functionality"""
    print("Testing Brave Search API functionality...")
    
    if not BRAVE_API_KEY:
        print("✗ BRAVE_API_KEY not found in environment variables")
        print("Please make sure you have BRAVE_API_KEY set in your .env file")
        return False
    
    print(f"✓ BRAVE_API_KEY is configured")
    
    # Test queries
    test_queries = [
        "latest AI news",
        "current weather New York",
        "Python programming tutorial"
    ]
    
    all_tests_passed = True
    
    for query in test_queries:
        print(f"\n--- Testing Brave Search with query: '{query}' ---")
        try:
            results = perform_brave_search(query, max_results=3)
            if results:
                print(f"✓ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['title']}")
                    print(f"     Snippet: {result['snippet'][:100]}...")
                    print(f"     Link: {result['link']}")
            else:
                print("⚠ No results found (this might indicate rate limiting or API issues)")
                all_tests_passed = False
        except Exception as e:
            print(f"✗ Error during Brave search: {e}")
            all_tests_passed = False
    
    return all_tests_passed

def test_complete_web_search_with_brave():
    """Test the complete web search function with Brave as fallback"""
    print("\n--- Testing complete web search function with Brave fallback ---")
    
    # Import the main web search function
    from app_fixed_comprehensive_final_new import perform_web_search
    
    try:
        # Test a query that might trigger fallbacks
        results = perform_web_search("test search query", max_results=2)
        
        if results:
            print(f"✓ Complete web search found {len(results)} results")
            print("✓ Web search fallback chain is working")
            return True
        else:
            print("⚠ Complete web search returned no results")
            return False
            
    except Exception as e:
        print(f"✗ Error in complete web search: {e}")
        return False

def main():
    """Run all Brave Search tests"""
    print("=== Brave Search API Tests ===\n")
    
    test_results = []
    
    # Test 1: Brave Search API directly
    test_results.append(test_brave_search())
    
    # Test 2: Complete web search integration
    test_results.append(test_complete_web_search_with_brave())
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All Brave Search tests passed!")
    else:
        print("✗ Some Brave Search tests failed.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)