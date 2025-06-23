@app.route('/', methods=['GET', 'POST'])
def index():
    chats = load_chats()
    # Use cached model data
    available_models_sorted_list = CACHED_AVAILABLE_MODELS_SORTED_LIST
    print(f"--- [ROUTE] Number of models in dropdown: {len(available_models_sorted_list)} ---")
    print(f"--- [ROUTE] First few models: {[m[0] for m in available_models_sorted_list[:5]]} ---")
    
    # model_provider_info and provider_class_map will be accessed via their global cached versions
    # directly in the POST handler where needed.

    # Get unique model names
    available_model_names = {name for name, count, intel, rt in available_models_sorted_list}
    print(f"--- [ROUTE] Number of unique model names: {len(available_model_names)} ---")
    print(f"--- [ROUTE] First few unique model names: {list(available_model_names)[:5]} ---")

    # --- Set Default Model ---
    # Set default to the first model in the custom sorted list
    if available_models_sorted_list:
        default_model = available_models_sorted_list[0][0] # Index 0 is model name
        print(f"--- Setting default model to: {default_model} (top of custom sort) ---")
    else:
        default_model = "gpt-3.5-turbo"
        print(f"--- Setting default model to: {default_model} (fallback) ---")
