# Changes Summary

## 1. Fixed Model Selection Persistence

- Modified the code to ensure the selected model persists between messages
- Added logic to use the model from the current chat if no model is explicitly provided in the form
- Updated the session and current chat with the selected model immediately
- Added debug logging to track model selection

## 2. Eliminated Duplicate Llama 4 Maverick Models

- Added deduplication logic in the `get_available_models_with_provider_counts` function
- Implemented a system that keeps only one model per display name, prioritizing models with:
  - More providers
  - Better performance (lower response time)
- Added additional Llama 4 Maverick variants to the `MODEL_DISPLAY_NAME_MAP`
- Added logging to show the reduction in model count after deduplication

## 3. Improved Provider Selection for Llama 4 Maverick

- Added Chutes.ai Llama 4 Maverick variants to the supported models lists for both Cerebras and Groq
- Enhanced the model ID mapping for Llama 4 Maverick to ensure it uses the correct IDs with Cerebras and Groq
- Added additional logging to track which model IDs are being used
- Improved the Groq model validation check to ensure it works with all Llama 4 Maverick variants

## 4. Added Last Modified Timestamp for Saved Chats

- Added a `last_modified` timestamp to each chat that gets updated whenever:
  - A user sends a message
  - The assistant responds
  - The model is changed
  - A chat is loaded from the saved chats list
- Changed the sorting logic in the saved chats view to use `last_modified` instead of `created_at`
- Added fallback to `created_at` for backward compatibility with existing chats
- Updated the display to show both "Last modified" and "Created" timestamps

These changes ensure that:
1. The model selection persists between messages
2. There are no duplicate Llama 4 Maverick models in the dropdown
3. Llama 4 Maverick uses fast APIs (Groq/Cerebras) when available
4. The most recently used chats appear at the top of the saved chats list