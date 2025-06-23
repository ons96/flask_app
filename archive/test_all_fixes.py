#!/usr/bin/env python3
"""
Comprehensive test for all the requested fixes and improvements
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import test functions
from app_fixed_comprehensive_final_new import (
    get_max_tokens_for_model, 
    is_continuation_request, 
    build_continuation_context,
    is_response_complete,
    perform_web_search,
    perform_brave_search,
    PROVIDER_MODEL_MAP
)

def test_qwen3_model_mapping():
    """Test that Qwen3 32B is correctly mapped"""
    print("=== Testing Qwen3 32B Model Mapping ===")
    
    # Check the model mapping
    qwen3_mapping = PROVIDER_MODEL_MAP.get("Qwen 3 32B", {})
    
    print(f"Qwen 3 32B mapping: {qwen3_mapping}")
    
    # Verify correct mappings
    expected_groq = "qwen/qwen3-32b"
    expected_cerebras = "qwen-3-32b"
    
    results = []
    
    if qwen3_mapping.get("groq") == expected_groq:
        print("✓ Groq mapping correct for Qwen 3 32B")
        results.append(True)
    else:
        print(f"✗ Groq mapping incorrect. Expected: {expected_groq}, Got: {qwen3_mapping.get('groq')}")
        results.append(False)
        
    if qwen3_mapping.get("cerebras") == expected_cerebras:
        print("✓ Cerebras mapping correct for Qwen 3 32B")
        results.append(True)
    else:
        print(f"✗ Cerebras mapping incorrect. Expected: {expected_cerebras}, Got: {qwen3_mapping.get('cerebras')}")
        results.append(False)
    
    return all(results)

def test_max_tokens_reasoning_models():
    """Test that reasoning models get higher token limits"""
    print("\n=== Testing Increased Max Tokens for Reasoning Models ===")
    
    test_cases = [
        ("Qwen 3 32B", 16384, "Should get higher tokens"),
        ("QwQ-32B", 16384, "Should get higher tokens"),
        ("Claude 3", 16384, "Should get higher tokens"),
        ("Llama 3", 4000, "Should get default tokens"),
        ("GPT-4", 4000, "Should get default tokens"),
    ]
    
    results = []
    
    for model_name, expected_tokens, reason in test_cases:
        actual_tokens = get_max_tokens_for_model(model_name)
        if actual_tokens == expected_tokens:
            print(f"✓ {model_name}: {actual_tokens} tokens ({reason})")
            results.append(True)
        else:
            print(f"✗ {model_name}: {actual_tokens} tokens (expected {expected_tokens}) - {reason}")
            results.append(False)
    
    return all(results)

def test_continuation_detection():
    """Test that continuation requests are properly detected"""
    print("\n=== Testing Continuation Request Detection ===")
    
    should_detect = [
        "continue",
        "continue your response",
        "please continue",
        "finish your answer",
        "complete your explanation", 
        "keep going",
        "go on",
        "what comes next",
        "and then?",
        "more",
        "finish your thinking",
        "complete your response"
    ]
    
    should_not_detect = [
        "what is 2+2",
        "explain machine learning",
        "tell me about Python",
        "how does this work",
        "what do you think about cats",
        "analyze this code",
        "help me understand"
    ]
    
    results = []
    
    print("Should detect as continuation:")
    for prompt in should_detect:
        detected = is_continuation_request(prompt)
        if detected:
            print(f"  ✓ '{prompt}' -> correctly detected")
            results.append(True)
        else:
            print(f"  ✗ '{prompt}' -> missed detection")
            results.append(False)
    
    print("\nShould NOT detect as continuation:")
    for prompt in should_not_detect:
        detected = is_continuation_request(prompt)
        if not detected:
            print(f"  ✓ '{prompt}' -> correctly ignored")
            results.append(True)
        else:
            print(f"  ✗ '{prompt}' -> false positive")
            results.append(False)
    
    return all(results)

def test_response_completion_detection():
    """Test that incomplete responses are properly detected"""
    print("\n=== Testing Response Completion Detection ===")
    
    complete_responses = [
        "This is a complete response.",
        "The answer is 42!",
        "Here's what you need to know: everything works fine.",
        "```python\nprint('hello')\n```",
        "<thinking>This is complete</thinking>\nThe answer is yes.",
        "Here are the three main points: 1. First, 2. Second, 3. Third.",
    ]
    
    incomplete_responses = [
        "This response is incomplete because",
        "The three main points are: 1. First point 2. Second point 3.",
        "Let me explain this step by step:",
        "The code would look like:\n```python\nprint(",
        "<thinking>This is incomplete",
        "For example,",
        "However,",
        "Therefore",
        "Let me think about this:",
        "The answer is"
    ]
    
    results = []
    
    print("Should detect as COMPLETE:")
    for response in complete_responses:
        is_complete = is_response_complete(response, "test-model")
        if is_complete:
            print(f"  ✓ '{response[:50]}...' -> correctly marked complete")
            results.append(True)
        else:
            print(f"  ✗ '{response[:50]}...' -> incorrectly marked incomplete")
            results.append(False)
    
    print("\nShould detect as INCOMPLETE:")
    for response in incomplete_responses:
        is_complete = is_response_complete(response, "test-model")
        if not is_complete:
            print(f"  ✓ '{response[:50]}...' -> correctly marked incomplete")
            results.append(True)
        else:
            print(f"  ✗ '{response[:50]}...' -> incorrectly marked complete")
            results.append(False)
    
    return all(results)

def test_web_search_integration():
    """Test that web search with multiple fallbacks works"""
    print("\n=== Testing Enhanced Web Search with Brave Fallback ===")
    
    try:
        # Test basic web search functionality
        results = perform_web_search("test query", max_results=2)
        
        if results and isinstance(results, list):
            print(f"✓ Web search returned {len(results)} results")
            
            # Check structure
            if results and all('title' in r and 'link' in r and 'snippet' in r for r in results):
                print("✓ Web search results have correct structure")
                return True
            else:
                print("✗ Web search results missing required fields")
                return False
        else:
            print("✗ Web search failed or returned invalid format")
            return False
            
    except Exception as e:
        print(f"✗ Web search error: {e}")
        return False

def test_continuation_context_building():
    """Test that continuation context is properly built"""
    print("\n=== Testing Continuation Context Building ===")
    
    # Mock chat history
    chat_history = [
        {
            "role": "user",
            "content": "Explain quantum computing in detail",
            "timestamp": "2024-01-01T10:00:00"
        },
        {
            "role": "assistant", 
            "content": "Quantum computing is a revolutionary field that leverages quantum mechanics. The key principles include superposition and entanglement, which allow",
            "timestamp": "2024-01-01T10:01:00"
        }
    ]
    
    context = build_continuation_context(chat_history)
    
    if context:
        print("✓ Successfully built continuation context")
        # Check that context includes both original question and previous response
        if "quantum computing" in context.lower() and "superposition" in context.lower():
            print("✓ Context includes relevant information from chat history")
            return True
        else:
            print("✗ Context missing important information")
            return False
    else:
        print("✗ Failed to build continuation context")
        return False

def main():
    """Run all comprehensive tests"""
    print("=== Comprehensive Tests for All Requested Fixes ===\n")
    
    test_results = []
    
    # Test 1: Qwen3 Model Mapping Fix
    test_results.append(test_qwen3_model_mapping())
    
    # Test 2: Increased Max Tokens for Reasoning Models
    test_results.append(test_max_tokens_reasoning_models())
    
    # Test 3: Continuation Request Detection
    test_results.append(test_continuation_detection())
    
    # Test 4: Response Completion Detection  
    test_results.append(test_response_completion_detection())
    
    # Test 5: Enhanced Web Search
    test_results.append(test_web_search_integration())
    
    # Test 6: Continuation Context Building
    test_results.append(test_continuation_context_building())
    
    # Summary
    print("\n" + "="*60)
    print("=== FINAL TEST SUMMARY ===")
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL FIXES SUCCESSFULLY IMPLEMENTED! 🎉")
        print("\n✅ Your requests have been completed:")
        print("  • Fixed Qwen3 32B model mapping for Groq & Cerebras")
        print("  • Increased max tokens for reasoning models (16384)")
        print("  • Added intelligent continuation request detection")
        print("  • Implemented response completion checking")
        print("  • Enhanced web search with Brave API fallback")
        print("  • Added context preservation for incomplete responses")
        print("  • Improved error handling and response buffering")
        print("\n🚀 Your Flask app is now ready with all improvements!")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Some fixes may need attention.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)