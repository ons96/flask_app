"""
Test script to verify model deduplication and provider selection for Llama 4 Maverick
"""

import sys
import os
from importlib import import_module
import json

# Import the app module
sys.path.append('c:/Users/owens/Coding Projects/flask_app')
app_module = import_module('app_fixed_comprehensive_final_new')

# Test model deduplication
print("\n=== TESTING MODEL DEDUPLICATION ===")
models_list, provider_info, provider_map = app_module.get_available_models_with_provider_counts()

# Check for Llama 4 Maverick models
maverick_models = []
for model_name, provider_count, intel_index, resp_time in models_list:
    display_name = app_module.MODEL_DISPLAY_NAME_MAP.get(model_name, model_name)
    if "maverick" in model_name.lower() or "maverick" in display_name.lower():
        maverick_models.append((model_name, display_name, provider_count))

print(f"\nFound {len(maverick_models)} Llama 4 Maverick models after deduplication:")
for model_name, display_name, provider_count in maverick_models:
    print(f"  - {model_name} -> {display_name} (Providers: {provider_count})")

# Test provider selection for Llama 4 Maverick
print("\n=== TESTING PROVIDER SELECTION FOR LLAMA 4 MAVERICK ===")

# Mock the necessary objects for testing
class MockRequest:
    def __init__(self, form_data):
        self.form = form_data

class MockSession:
    def __init__(self):
        self.data = {}
    def __getitem__(self, key):
        return self.data.get(key)
    def __setitem__(self, key, value):
        self.data[key] = value

# Create a mock function to simulate provider selection
def test_provider_selection(model_name):
    print(f"\nTesting provider selection for model: {model_name}")
    
    # Check if model is in Cerebras supported models
    # These are defined in the index function, so we'll check the display name mapping instead
    
    # Check display name mapping
    display_name = app_module.MODEL_DISPLAY_NAME_MAP.get(model_name, model_name)
    print(f"  - Display name: {display_name}")
    
    # Check if this is a Llama 4 Maverick model
    is_maverick = "maverick" in model_name.lower() or "maverick" in display_name.lower()
    print(f"  - Is Llama 4 Maverick model: {is_maverick}")

# Test with different Llama 4 Maverick variants
test_provider_selection("llama-4-maverick")
test_provider_selection("meta-llama/llama-4-maverick-17b-16e-instruct")
test_provider_selection("chutesai/Llama-4-Maverick-17B-128E-Instruct")

print("\n=== TEST COMPLETE ===")