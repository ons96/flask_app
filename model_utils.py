# Enhanced normalize_model_name function
def normalize_model_name(name):
    """
    Normalizes model names to help identify duplicates.
    Handles cases like "Llama 4 Maverick" vs "llama-4-maverick"
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Replace special characters with spaces
    name = re.sub(r'[_\-/]', ' ', name)
    
    # Handle specific model families
    if "llama" in name:
        # Extract llama version and variant
        llama_match = re.search(r'llama[\s-]*(\d+)[\s-]*(maverick|scout|reasoning|instruct)?', name)
        if llama_match:
            version = llama_match.group(1)
            variant = llama_match.group(2) or ""
            return f"llama{version}{variant}"
    
    if "qwen" in name:
        # Keep parameter size for Qwen models
        qwen_match = re.search(r'qwen[\s-]*(\d+)[\s-]*(\d+[bB])', name)
        if qwen_match:
            version = qwen_match.group(1)
            size = qwen_match.group(2)
            return f"qwen{version}{size}"
    
    # For reasoning models with different levels
    reasoning_match = re.search(r'(.*?)[\s-]*(mini|high|low|reasoning)', name)
    if reasoning_match:
        base = reasoning_match.group(1)
        level = reasoning_match.group(2)
        return f"{base}{level}"
    
    # General normalization for other models
    # Remove spaces and special characters
    normalized = re.sub(r'\s+', '', name)
    # Remove version numbers and parameter sizes for general comparison
    if not any(x in name for x in ["qwen", "llama"]):
        normalized = re.sub(r'\d+[bBmMkK]?', '', normalized)
    
    return normalized

# Function to check if two models are duplicates
def are_duplicate_models(model1, model2):
    """
    Determines if two models are duplicates based on their normalized names.
    Returns True if they are the same model, False otherwise.
    """
    # Don't consider models with different parameter sizes as duplicates
    # For example, Qwen 3 32B and Qwen 3 235B should not be merged
    
    # Extract parameter size if present
    size1_match = re.search(r'(\d+)[bB]', model1)
    size2_match = re.search(r'(\d+)[bB]', model2)
    
    # If both have parameter sizes and they're different, not duplicates
    if size1_match and size2_match:
        size1 = int(size1_match.group(1))
        size2 = int(size2_match.group(1))
        if size1 != size2:
            return False
    
    # Check for different reasoning levels
    level1_match = re.search(r'(mini|high|low|reasoning)', model1.lower())
    level2_match = re.search(r'(mini|high|low|reasoning)', model2.lower())
    
    if level1_match and level2_match:
        level1 = level1_match.group(1)
        level2 = level2_match.group(1)
        if level1 != level2:
            return False
    
    # Normalize both names and compare
    norm1 = normalize_model_name(model1)
    norm2 = normalize_model_name(model2)
    
    return norm1 == norm2 and norm1 != ""

# Function to get a display name for a model
def get_model_display_name(model_name, intel_index, resp_time):
    """
    Formats the model name for display in the dropdown with performance metrics.
    Format: "Model Name (58, 1.28s)" where 58 is intelligence index and 1.28s is response time.
    """
    # Use the display name mapping if available
    display_name = MODEL_DISPLAY_NAME_MAP.get(model_name.lower(), model_name)
    
    # Format with performance metrics
    if intel_index is not None and resp_time is not None:
        if intel_index == 0 or intel_index == "N/A":
            intel_str = "Intel N/A"
        else:
            intel_str = str(intel_index)
        
        if resp_time == float('inf'):
            resp_str = "N/A"
        else:
            resp_str = f"{resp_time:.2f}s"
        
        return f"{display_name} ({intel_str}, {resp_str})"
    else:
        return display_name

# Function to sort models according to the specified logic
def sort_models_by_priority(models_list, free_providers):
    """
    Sorts models according to the specified priority logic:
    1. First, choose models with free API providers, sorted by response time
    2. For models with the same response time, prioritize by intelligence index
    3. For models with the same intelligence index, prioritize by context window size
    4. After all free models, add the remaining models
    """
    # Create a list to store models with their attributes
    model_data = []
    
    # Process each model
    for model_tuple in models_list:
        model_name, provider_count, intel_index, resp_time = model_tuple
        
        # Check if this model has a free API provider
        has_free_api = any(
            model_name.lower() == offering["model_key"].lower() 
            for offering in KNOWN_FREE_MODEL_OFFERINGS
        )
        
        # Get context window size
        context_window = 0
        for entry in PROVIDER_PERFORMANCE_CACHE:
            if entry.get('model_name_scraped', '').lower() == model_name.lower():
                context_window = entry.get('context_window_int', 0)
                break
        
        # Add to model data list
        model_data.append({
            'model_name': model_name,
            'provider_count': provider_count,
            'intel_index': intel_index if intel_index is not None else 0,
            'resp_time': resp_time if resp_time is not None else float('inf'),
            'context_window': context_window,
            'has_free_api': has_free_api,
            'original_tuple': model_tuple
        })
    
    # Split into free and non-free models
    free_models = [m for m in model_data if m['has_free_api']]
    non_free_models = [m for m in model_data if not m['has_free_api']]
    
    # Sort free models by the specified logic
    free_models.sort(key=lambda x: (
        x['resp_time'],  # First by response time (ascending)
        -x['intel_index'],  # Then by intelligence index (descending)
        -x['context_window']  # Then by context window (descending)
    ))
    
    # Sort non-free models by the same logic
    non_free_models.sort(key=lambda x: (
        x['resp_time'],
        -x['intel_index'],
        -x['context_window']
    ))
    
    # Combine the lists, with free models first
    sorted_models = [m['original_tuple'] for m in free_models] + [m['original_tuple'] for m in non_free_models]
    
    # Remove duplicates, keeping the first occurrence
    unique_models = []
    seen_normalized_names = set()
    
    for model_tuple in sorted_models:
        model_name = model_tuple[0]
        normalized_name = normalize_model_name(model_name)
        
        # Skip if we've already seen this normalized name
        if normalized_name in seen_normalized_names:
            continue
        
        # Check if this model is a duplicate of any model we've already added
        is_duplicate = False
        for added_model in unique_models:
            if are_duplicate_models(model_name, added_model[0]):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_models.append(model_tuple)
            seen_normalized_names.add(normalized_name)
    
    return unique_models
