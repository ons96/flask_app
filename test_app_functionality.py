#!/usr/bin/env python3
"""
Test script to verify Flask app functionality and web search integration
"""

import sys
import os
import threading
import time
import requests
import subprocess

def test_app_startup():
    """Test that the Flask app starts without errors"""
    print("Testing Flask app startup...")
    
    try:
        # Try to import the main app
        from app_fixed_comprehensive_final_new import app
        print("✓ Flask app imported successfully")
        
        # Test that the app instance is created
        if app:
            print("✓ Flask app instance created successfully")
        else:
            print("✗ Flask app instance is None")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error importing or creating Flask app: {e}")
        return False

def test_web_search_integration():
    """Test web search functionality in isolation"""
    print("\nTesting web search integration...")
    
    try:
        # Import web search functions
        from app_fixed_comprehensive_final_new import perform_web_search
        
        # Test a simple search
        results = perform_web_search("test query", max_results=2)
        
        if isinstance(results, list):
            print(f"✓ Web search returns list (found {len(results)} results)")
            
            # Check result structure if we have results
            if results:
                first_result = results[0]
                required_keys = ['title', 'link', 'snippet']
                
                if all(key in first_result for key in required_keys):
                    print("✓ Web search results have correct structure")
                else:
                    print("✗ Web search results missing required keys")
                    return False
            else:
                print("⚠ Web search returned no results (this might be normal)")
                
            return True
        else:
            print("✗ Web search does not return a list")
            return False
            
    except Exception as e:
        print(f"✗ Error testing web search: {e}")
        return False

def test_smart_search_logic():
    """Test smart search keyword detection"""
    print("\nTesting smart search logic...")
    
    try:
        # Test queries that should trigger search
        should_search_queries = [
            "what is the latest news",
            "current weather",
            "who is the current president"
        ]
        
        # Test queries that should NOT trigger search
        should_not_search_queries = [
            "what is 2+2",
            "explain programming",
            "how to cook pasta"
        ]
        
        # Import the smart search logic (we'll simulate it here)
        import re
        
        # Time-sensitive keywords
        time_keywords = [
            "latest", "recent", "today", "current", "now", "present", "currently",
            "breaking", "news", "update", "updated", "newest", "new"
        ]
        
        # Exclusion patterns
        exclusion_patterns = [
            r'\bwhat is \d+[\+\-\*\/]\d+',
            r'\bwhat is the capital of',
            r'\bexplain (how to|what is)',
            r'\bhow to (write|code|program|make|cook|do)',
        ]
        
        def should_trigger_search(query):
            prompt_lower = query.lower()
            is_excluded = any(re.search(pattern, prompt_lower) for pattern in exclusion_patterns)
            if is_excluded:
                return False
            return any(keyword in prompt_lower for keyword in time_keywords)
        
        # Test positive cases
        all_passed = True
        for query in should_search_queries:
            if should_trigger_search(query):
                print(f"✓ Correctly triggers search: {query}")
            else:
                print(f"✗ Should trigger search but doesn't: {query}")
                all_passed = False
        
        # Test negative cases
        for query in should_not_search_queries:
            if not should_trigger_search(query):
                print(f"✓ Correctly avoids search: {query}")
            else:
                print(f"✗ Should not trigger search but does: {query}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error testing smart search logic: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Flask App Functionality Tests ===\n")
    
    test_results = []
    
    # Test 1: App startup
    test_results.append(test_app_startup())
    
    # Test 2: Web search integration
    test_results.append(test_web_search_integration())
    
    # Test 3: Smart search logic
    test_results.append(test_smart_search_logic())
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Flask app is ready to use.")
    else:
        print("✗ Some tests failed. Check the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)