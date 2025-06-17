def initialize_model_cache():
    """
    Initializes the global model cache by fetching available models and their providers.
    Falls back to default models if there are any errors.
    """
    global CACHED_AVAILABLE_MODELS_SORTED_LIST, CACHED_MODEL_PROVIDER_INFO, CACHED_PROVIDER_CLASS_MAP
    
    # Define default models with their properties
    default_models = [
        ("gpt-3.5-turbo", 1, 85, 0.5),      # OpenAI
        ("gpt-4", 1, 95, 1.0),              # OpenAI
        ("claude-3-opus", 1, 98, 1.2),      # Anthropic
        ("claude-3-sonnet", 1, 96, 0.8),    # Anthropic
        ("claude-3-haiku", 1, 94, 0.4),     # Anthropic
        ("gemini-pro", 1, 93, 0.6),         # Google
        ("llama-2-70b", 1, 92, 1.5),        # Meta
        ("llama-2-13b", 1, 90, 1.0),        # Meta
        ("llama-2-7b", 1, 88, 0.8),         # Meta
        ("mistral-7b", 1, 89, 0.7),         # Mistral AI
        ("mixtral-8x7b", 1, 91, 1.1),       # Mistral AI
        ("qwen-72b", 1, 93, 1.3),           # Alibaba
        ("qwen-14b", 1, 90, 0.9),           # Alibaba
        ("qwen-7b", 1, 88, 0.7),            # Alibaba
        ("deepseek-coder-33b", 1, 92, 1.2),  # DeepSeek
        ("deepseek-coder-6.7b", 1, 89, 0.8), # DeepSeek
        ("deepseek-llm-67b", 1, 93, 1.4),    # DeepSeek
        ("deepseek-llm-7b", 1, 90, 0.9),     # DeepSeek
        ("yi-34b", 1, 91, 1.2),             # 01.AI
        ("yi-6b", 1, 88, 0.7)               # 01.AI
    ]
    
    print("\n=== [CACHE] Starting model cache initialization ===")
    start_time = time.time()
    
    try:
        # Try to get models from g4f with timeout
        print("--- [CACHE] Fetching models from g4f...")
        list_data, info_data, map_data = get_available_models_with_provider_counts()
        
        # Validate the returned data
        if not list_data or not isinstance(list_data, list):
            raise ValueError("No valid model data returned from g4f")
            
        print(f"--- [CACHE] Successfully retrieved {len(list_data)} models from g4f")
        
        # Initialize caches with fetched data
        model_list = list_data
        CACHED_MODEL_PROVIDER_INFO = info_data if info_data and isinstance(info_data, dict) else {}
        CACHED_PROVIDER_CLASS_MAP = map_data if map_data and isinstance(map_data, dict) else {}
        
        # Add any default models that aren't already in the list
        existing_models = {model[0].lower() for model in model_list}
        added_defaults = 0
        
        for default_model in default_models:
            model_name = default_model[0]
            if model_name.lower() not in existing_models:
                model_list.append(default_model)
                CACHED_MODEL_PROVIDER_INFO[model_name] = []
                added_defaults += 1
                
        if added_defaults > 0:
            print(f"--- [CACHE] Added {added_defaults} default models not found in g4f")
        
        # Sort models using our custom sorting logic
        print("--- [CACHE] Sorting models with custom priority logic...")
        try:
            # First, sort models by our priority logic
            sorted_models = sort_models_by_priority(model_list, KNOWN_FREE_MODEL_OFFERINGS)
            
            # Update the display names to include performance metrics
            formatted_models = []
            for model_tuple in sorted_models:
                model_name, provider_count, intel_index, resp_time = model_tuple
                display_name = get_model_display_name(model_name, intel_index, resp_time)
                formatted_models.append((display_name, provider_count, intel_index, resp_time))
            
            CACHED_AVAILABLE_MODELS_SORTED_LIST = formatted_models
            
            # Create a mapping from display names to original model names
            global DISPLAY_NAME_TO_MODEL_MAP
            DISPLAY_NAME_TO_MODEL_MAP = {
                get_model_display_name(model[0], model[2], model[3]): model[0]
                for model in sorted_models
            }
            
        except Exception as sort_error:
            print(f"--- [CACHE] Error with custom sorting, falling back to default sort: {sort_error}")
            # Fallback to original sorting logic
            model_list.sort(key=lambda x: (
                -int(x[1]) if isinstance(x[1], (int, float)) else 0,  # Provider count
                -int(x[2]) if isinstance(x[2], (int, float)) else 0,  # Intelligence index
                float('inf') if not isinstance(x[3], (int, float)) else float(x[3]),  # Response time
                str(x[0])  # Model name
            ))
            CACHED_AVAILABLE_MODELS_SORTED_LIST = model_list
        
        # Log success
        elapsed = time.time() - start_time
        print(f"=== [CACHE] Model cache initialized in {elapsed:.2f}s ===")
        print(f"=== [CACHE] Total models: {len(CACHED_AVAILABLE_MODELS_SORTED_LIST)}")
        print(f"=== [CACHE] Top 5 models: {[m[0] for m in CACHED_AVAILABLE_MODELS_SORTED_LIST[:5]]}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        # Fall back to default models on any error
        elapsed = time.time() - start_time
        print(f"--- [CACHE] Error initializing model cache after {elapsed:.2f}s: {str(e)}")
        print("--- [CACHE] Falling back to default models")
        import traceback
        print(f"--- [CACHE] Traceback: {traceback.format_exc()}")
        
        CACHED_AVAILABLE_MODELS_SORTED_LIST = default_models
        CACHED_MODEL_PROVIDER_INFO = {model[0]: [] for model in default_models}
        CACHED_PROVIDER_CLASS_MAP = {}
        
        print(f"=== [CACHE] Initialized with {len(default_models)} default models ===\n")
