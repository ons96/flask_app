    # Handle POST request (chat submission)
    if request.method == 'POST':
        try:
            # Get form data
            user_message = request.form.get('user_message', '').strip()
            selected_model = request.form.get('model', default_model).strip()
            
            # If the selected model is a display name, get the actual model name
            actual_model_name = DISPLAY_NAME_TO_MODEL_MAP.get(selected_model, selected_model)
            
            # Get the chat history from session
            chat_history = session.get('chat_history', [])
            
            # Add user message to chat history
            chat_history.append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Update session
            session['chat_history'] = chat_history
            
            # Get providers for the selected model
            model_providers = CACHED_MODEL_PROVIDER_INFO.get(actual_model_name, [])
            
            # Prioritize providers, ensuring g4f is used as a last resort
            prioritized_providers = prioritize_providers_for_model(actual_model_name, model_providers)
            
            if not prioritized_providers:
                print(f"--- [CHAT] No providers found for model: {actual_model_name} ---")
                # Add error message to chat history
                chat_history.append({
                    'role': 'assistant',
                    'content': f"Error: No providers available for the selected model ({actual_model_name}). Please try a different model.",
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'error': True
                })
                session['chat_history'] = chat_history
                return redirect(url_for('index'))
