def prioritize_providers_for_model(model_name, providers):
    """
    Prioritizes providers for a given model, ensuring g4f is used as a last resort.
    Returns a sorted list of providers.
    """
    if not providers:
        return []
    
    # Create a list of (provider, priority) tuples
    provider_priorities = []
    
    for provider in providers:
        provider_name = getattr(provider, '__name__', str(provider)).lower()
        
        # Assign priority (lower is better)
        priority = 100  # Default priority
        
        # Check if this is a known free provider for this model
        for offering in KNOWN_FREE_MODEL_OFFERINGS:
            if (offering["provider_key"].lower() == provider_name.lower() and 
                (offering["model_key"].lower() == model_name.lower() or 
                 normalize_model_name(offering["model_key"]) == normalize_model_name(model_name))):
                priority = 10  # High priority for known free providers
                break
        
        # Specific provider priorities
        if "g4f" in provider_name.lower():
            priority = 999  # Lowest priority for g4f
        elif "openai" in provider_name.lower():
            priority = 20
        elif "anthropic" in provider_name.lower():
            priority = 30
        elif "google" in provider_name.lower():
            priority = 40
        elif "groq" in provider_name.lower():
            priority = 15  # Higher priority for Groq
        elif "cerebras" in provider_name.lower():
            priority = 15  # Higher priority for Cerebras
        
        provider_priorities.append((provider, priority))
    
    # Sort by priority (lower is better)
    provider_priorities.sort(key=lambda x: x[1])
    
    # Return just the providers in priority order
    return [p[0] for p in provider_priorities]
