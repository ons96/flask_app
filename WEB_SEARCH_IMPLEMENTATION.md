# Web Search Implementation - Enhanced Flask Chat App

## Overview
The Flask chat application now includes a robust, multi-layered web search system that provides current information to LLM responses. The system uses multiple fallback methods to ensure reliable search results even when primary services are rate-limited or unavailable.

## Search Methods (In Order of Priority)

### 1. DuckDuckGo Search (Primary)
- **Cost**: Free
- **Limit**: Rate-limited (varies)
- **Reliability**: High when not rate-limited
- **Implementation**: Uses `duckduckgo-search` library

### 2. SerpAPI (Google Search)
- **Cost**: Paid API (usually has free credits)
- **Limit**: Based on subscription plan
- **Reliability**: Very high
- **Implementation**: Uses `serpapi` library
- **Requirement**: `SERPAPI_API_KEY` in .env file

### 3. Brave Search API
- **Cost**: Free tier available (2000 searches/month)
- **Limit**: 2000 searches per month on free tier
- **Reliability**: High
- **Implementation**: Direct API calls to Brave Search
- **Requirement**: `BRAVE_API_KEY` in .env file

### 4. Bing Search (Scraping Fallback)
- **Cost**: Free (web scraping)
- **Limit**: Rate-limited by Bing
- **Reliability**: Medium (depends on page structure)
- **Implementation**: Web scraping with BeautifulSoup

### 5. Google Search (Final Fallback)
- **Cost**: Free (web scraping)
- **Limit**: Rate-limited by Google
- **Reliability**: Low-Medium (depends on page structure)
- **Implementation**: Web scraping with BeautifulSoup

## Smart Search Logic

The app includes intelligent search triggering that determines when to perform web searches based on the user's query. This prevents unnecessary searches for basic questions while ensuring current information is retrieved when needed.

### Triggers Search (Smart Mode)
- Time-sensitive keywords: "latest", "recent", "today", "current", "now", "breaking", "news"
- Question patterns: "what happened", "who is the current", "latest version", "stock price"
- Technology keywords: "version", "release", "announcement", "launched"
- Sports keywords: "score", "game", "season", "championship"
- Financial keywords: "stock", "crypto", "market", "inflation"
- Year references: 2020-2030
- Company names: Google, Apple, Microsoft, Tesla, OpenAI, etc.
- Date patterns: Month names

### Excludes from Search
- Basic math: "what is 2+2"
- Geography basics: "what is the capital of"
- How-to questions: "how to write code"
- Explanations: "explain machine learning"
- Philosophical: "what is the meaning of life"

## Web Search Modes

### Off
- No web search is performed
- LLM responds based only on training data

### Smart (Default)
- Automatically determines if web search is needed
- Uses enhanced keyword detection and exclusion patterns
- Balances accuracy with search efficiency

### On
- Always performs web search for every query
- Useful when you specifically want current information
- May use more API quotas/credits

## Integration with LLMs

When web search results are found, they are automatically integrated into the LLM prompt:

```
Current web search results:
• [Result Title]: [Snippet]
• [Result Title]: [Snippet]
• [Result Title]: [Snippet]

Based on the above search results and your knowledge, please answer the following question:
[User Question]

Please provide a comprehensive answer that incorporates both the current information from the search results and relevant background knowledge.
```

## Setup Instructions

### 1. Required Dependencies
```bash
pip install duckduckgo-search beautifulsoup4 requests
```

### 2. Optional API Keys (Add to .env file)
```env
# For SerpAPI (Google Search)
SERPAPI_API_KEY=your_serpapi_key_here

# For Brave Search API
BRAVE_API_KEY=your_brave_api_key_here
```

### 3. Optional Dependencies
```bash
# For SerpAPI
pip install google-search-results

# Brave Search uses standard requests (already included)
```

## Error Handling

The system includes comprehensive error handling:
- Graceful fallback when services are rate-limited
- Automatic detection of API quota exhaustion
- Continues operation even if all search methods fail
- Detailed logging for debugging

## Performance Features

- **Multiple Fallbacks**: Ensures search results even when primary services fail
- **Smart Caching**: Results are used immediately without unnecessary retries
- **Rate Limit Detection**: Automatically switches to alternatives when rate-limited
- **Timeout Protection**: All search requests have timeouts to prevent hanging
- **Blackberry Optimization**: Works efficiently on limited resources

## Testing

Use the provided test scripts to verify functionality:

```bash
# Test web search functionality
python test_web_search.py

# Test smart search logic
python test_smart_search.py

# Test Brave Search API (if configured)
python test_brave_search.py

# Test complete app functionality
python test_app_functionality.py
```

## Troubleshooting

### Common Issues

1. **DuckDuckGo Rate Limiting**
   - Error: "202 Ratelimit"
   - Solution: System automatically falls back to other methods

2. **SerpAPI Credits Exhausted**
   - Error: Messages containing "quota", "limit", or "credit"
   - Solution: System automatically tries Brave Search or scraping fallbacks

3. **All Search Methods Failing**
   - Check internet connection
   - Verify API keys if using paid services
   - Check if search terms are being blocked

4. **Brave Search API Issues**
   - Verify `BRAVE_API_KEY` is correctly set in .env
   - Check if monthly quota (2000 searches) is exceeded
   - Ensure API key has proper permissions

## Benefits for Blackberry Classic

- **Minimal Resource Usage**: Efficient fallback system
- **Old Browser Compatible**: All UI elements remain unchanged
- **Offline Graceful**: Works without search when all methods fail
- **Fast Response**: Smart search prevents unnecessary delays
- **Low Bandwidth**: Search results are limited and compressed

## Security Notes

- API keys are stored securely in .env file
- No sensitive data is transmitted in search queries
- Scraping fallbacks use standard user agents
- All requests include appropriate timeouts