# Changes Made to Fix Issues

## 1. Removed Context Window Size from Model Dropdown

- Modified the model dropdown to only display intelligence index and response time
- Removed context window size from the display to make it more consistent and cleaner

## 2. Fixed Selected Model Persistence

- Added code to properly save the selected model to the session when a POST request is made
- Ensured the model is updated in the current chat immediately when selected
- Removed duplicate code that was updating the model in the chat

## 3. Improved Web Search Implementation

- Reordered the web search methods to prioritize free and unlimited options:
  1. DuckDuckGo (free and unlimited) as the first option
  2. Gemini grounding (if Google API key is available) as the second option
  3. SerpAPI (limited to 100 free searches/month) as the last resort
- Added better error handling and logging for each search method
- Improved the display of search results to indicate which search method was used

These changes should address all the issues mentioned while maintaining the functionality of the application.