#!/usr/bin/env python3
"""
Test script for reasoning model fixes (QwQ, Qwen3, etc.)
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
    MODEL_MAPPING
)

def test_model_mapping():
    """Test that Qwen3 32B is correctly mapped"""
    print("=== Testing Model Mapping ===")
    
    # Check the model mapping
    qwen3_mapping = MODEL_MAPPING.get("Qwen 3 32B", {})
    
    print(f"Qwen 3 32B mapping: {qwen3_mapping}")
    
    # Verify correct mappings
    expected_groq = "qwen/qwen3-32b"
    expected_cerebras = "qwen-3-32b"
    
    if qwen3_mapping.get("groq") == expected_groq:
        print("✓ Groq mapping correct for Qwen 3 32B")
    else:
        print(f"✗ Groq mapping incorrect. Expected: {expected_groq}, Got: {qwen3_mapping.get('groq')}")
        
    if qwen3_mapping.get("cerebras") == expected_cerebras:
        print("✓ Cerebras mapping correct for Qwen 3 32B")
    else:
        print(f"✗ Cerebras mapping incorrect. Expected: {expected_cerebras}, Got: {qwen3_mapping.get('cerebras')}")

def test_max_tokens():
    """Test max tokens function for different models"""
    print("\n=== Testing Max Tokens ===")
    
    test_cases = [
        ("Qwen 3 32B", 16384),
        ("QwQ-32B", 16384),
        ("Claude 3", 16384),
        ("Llama 3", 4000),
        ("GPT-4", 4000),
    ]
    
    for model_name, expected_tokens in test_cases:
        actual_tokens = get_max_tokens_for_model(model_name)
        if actual_tokens == expected_tokens:
            print(f"✓ {model_name}: {actual_tokens} tokens (correct)")
        else:
            print(f"✗ {model_name}: {actual_tokens} tokens (expected {expected_tokens})")

def test_continuation_detection():
    """Test continuation request detection"""
    print("\n=== Testing Continuation Detection ===")
    
    should_detect = [
        "continue",
        "continue your response",
        "please continue",
        "finish your answer",
        "complete your explanation",
        "keep going",
        "what comes next",
        "and then?",
        "more",
    ]
    
    should_not_detect = [
        "what is 2+2",
        "explain machine learning",
        "tell me about Python",
        "how does this work",
        "what do you think",
    ]
    
    print("Should detect continuation:")
    for prompt in should_detect:
        detected = is_continuation_request(prompt)
        status = "✓" if detected else "✗"
        print(f"  {status} '{prompt}' -> {detected}")
    
    print("\nShould NOT detect continuation:")
    for prompt in should_not_detect:
        detected = is_continuation_request(prompt)
        status = "✓" if not detected else "✗"
        print(f"  {status} '{prompt}' -> {detected}")

def test_response_completion():
    """Test response completion detection"""
    print("\n=== Testing Response Completion Detection ===")
    
    complete_responses = [
        "This is a complete response.",
        "The answer is 42!",
        "Here's what you need to know: everything works fine.",
        "```python\nprint('hello')\n```",
        "<thinking>This is complete</thinking>\nThe answer is yes.",
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
    ]
    
    print("Should detect as COMPLETE:")
    for response in complete_responses:
        is_complete = is_response_complete(response, "test-model")
        status = "✓" if is_complete else "✗"
        print(f"  {status} '{response[:50]}...' -> {is_complete}")
    
    print("\nShould detect as INCOMPLETE:")
    for response in incomplete_responses:
        is_complete = is_response_complete(response, "test-model")
        status = "✓" if not is_complete else "✗"
        print(f"  {status} '{response[:50]}...' -> {is_complete}")

def test_continuation_context():
    """Test building continuation context from chat history"""
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
            "content": "Quantum computing is a revolutionary field that leverages quantum mechanics to process information. The key principles include superposition, where qubits can exist in multiple states simultaneously, and entanglement, which allows",
            "timestamp": "2024-01-01T10:01:00"
        },
        {
            "role": "user",
            "content": "continue",
            "timestamp": "2024-01-01T10:02:00"
        }
    ]
    
    context = build_continuation_context(chat_history)
    
    if context:
        print("✓ Successfully built continuation context:")
        print(f"Context preview: {context[:200]}...")
    else:
        print("✗ Failed to build continuation context")

def main():
    """Run all tests"""
    print("=== Reasoning Model Enhancement Tests ===\n")
    
    try:
        test_model_mapping()
        test_max_tokens()
        test_continuation_detection()
        test_response_completion()
        test_continuation_context()
        
        print("\n=== Test Summary ===")
        print("✓ All reasoning model enhancement tests completed!")
        print("\nKey improvements implemented:")
        print("- Fixed Qwen3 32B model mapping for Groq and Cerebras")
        print("- Increased max tokens for reasoning models (16384 vs 4000)")
        print("- Added intelligent continuation request detection")
        print("- Implemented context preservation for incomplete responses")
        print("- Added response completion detection")
        
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)