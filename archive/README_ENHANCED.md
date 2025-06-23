# 🤖 Enhanced Flask LLM Chat Application

A comprehensive, production-ready Flask web application for chatting with multiple Large Language Models (LLMs) through various API providers.

## 🌟 Features

### Core Functionality
- **Multi-Provider Support**: Direct API integration with Groq, Cerebras, Google AI, OpenRouter, and G4F fallback
- **Smart Model Selection**: Intelligent prioritization based on performance, cost, and availability
- **Real-time Chat Interface**: Modern, responsive web interface with real-time messaging
- **Chat Management**: Save, load, and manage multiple chat sessions
- **Web Search Integration**: Automatic web search for time-sensitive queries with multiple search providers

### Advanced Features
- **Model Deduplication**: Eliminates duplicate models with different API names
- **Provider Prioritization**: Free providers prioritized over paid ones
- **Regenerate Functionality**: Properly replace previous responses instead of appending
- **Smart Web Search**: Automatically triggered for queries needing current information
- **Performance Optimization**: Async API calls with intelligent fallback mechanisms

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- API keys for desired providers (optional but recommended)

### Installation

1. **Clone or download the application files**
   ```bash
   cd Coding\ Projects/flask_app/
   ```

2. **Install dependencies**
   ```bash
   pip install flask flask-session g4f requests beautifulsoup4 duckduckgo-search aiohttp python-dotenv
   ```

3. **Set up environment variables** (create `.env` file)
   ```bash
   # Required for enhanced functionality
   GROQ_API_KEY=your_groq_api_key_here
   CEREBRAS_API_KEY=your_cerebras_api_key_here
   GOOGLE_API_KEY=your_google_ai_api_key_here
   
   # Optional providers
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   CHUTES_API_KEY=your_chutes_api_key_here
   SERPAPI_API_KEY=your_serpapi_key_here
   
   # Flask configuration
   SECRET_KEY=your_random_secret_key_here
   ```

4. **Run the application**
   ```bash
   python app_fixed_comprehensive_final_updated.py
   ```
   
   Or use the run script:
   ```bash
   python run_enhanced_app.py
   ```

5. **Access the application**
   Open your browser and go to: `http://localhost:5000`

## 🔑 API Key Setup

### Groq API (Recommended - Fast and Free)
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Generate an API key
4. Add to `.env`: `GROQ_API_KEY=your_key_here`

### Cerebras API (Recommended - Fast and Free)
1. Visit [inference.cerebras.ai](https://inference.cerebras.ai)
2. Create an account
3. Get your API key
4. Add to `.env`: `CEREBRAS_API_KEY=your_key_here`

### Google AI API (For Gemini Models)
1. Visit [aistudio.google.com](https://aistudio.google.com)
2. Get your API key
3. Add to `.env`: `GOOGLE_API_KEY=your_key_here`

### OpenRouter API (Optional - Access to Many Models)
1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up and get credits
3. Add to `.env`: `OPENROUTER_API_KEY=your_key_here`

## 🎯 Usage Guide

### Basic Chat
1. Select a model from the dropdown (automatically sorted by performance and availability)
2. Type your message in the text area
3. Click "Send Message" or press Enter
4. View the response with provider attribution

### Web Search
- **Smart Mode** (Default): Automatically searches for time-sensitive queries
- **Always On**: Searches for every query
- **Off**: No web search

### Chat Management
- **New Chat**: Start a fresh conversation
- **Saved Chats**: View and load previous conversations
- **Delete Chat**: Remove current or saved chats
- **Auto-naming**: Chats automatically named based on first message

### Advanced Features
- **Regenerate**: Click to get a new response for the last message
- **Model Switching**: Change models mid-conversation
- **Provider Info**: See which API provider generated each response

## 🔧 Configuration

### Model Prioritization
Models are automatically sorted by:
1. **Free availability** (free models prioritized)
2. **Intelligence index** (higher is better)
3. **Response time** (lower is better)
4. **Provider count** (more providers = higher availability)

### Provider Priority Order
1. **Groq** - Fast, free, reliable
2. **Cerebras** - Fast, free, reliable
3. **Google AI** - High quality, rate limited
4. **OpenRouter** - Paid but comprehensive
5. **G4F** - Free fallback, less reliable

### Web Search Providers
1. **DuckDuckGo** - Free, unlimited, primary choice
2. **Google AI Grounding** - High quality, requires API key
3. **SerpAPI** - High quality, limited free tier

## 📁 File Structure

```
flask_app/
├── app_fixed_comprehensive_final_updated.py  # Main enhanced application
├── run_enhanced_app.py                       # Run script
├── README_ENHANCED.md                        # This file
├── IMPLEMENTED_FIXES_SUMMARY.md             # Detailed fixes documentation
├── .env                                     # Environment variables (create this)
├── chats.json                              # Chat storage (auto-created)
├── flask_session/                          # Session storage (auto-created)
├── logs/                                   # Application logs (auto-created)
└── requirements.txt                        # Dependencies list
```

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install flask flask-session g4f requests beautifulsoup4 duckduckgo-search aiohttp python-dotenv
```

**2. API Key Not Working**
- Verify the key is correct in `.env`
- Check if the key has credits/quota remaining
- Ensure the model is supported by that provider

**3. No Models Showing**
- Check internet connection
- Verify G4F is working: `pip install -U g4f`
- Restart the application

**4. Web Search Not Working**
- DuckDuckGo might be rate-limited (temporary)
- Try setting SERPAPI_API_KEY for backup search

**5. Regenerate Not Working**
- Ensure there's at least one assistant response in the chat
- Try refreshing the page

### Debug Mode
Add this to your `.env` for detailed logging:
```
FLASK_DEBUG=True
```

## 🚀 Performance Tips

1. **Use Direct APIs**: Set up Groq and Cerebras API keys for best performance
2. **Model Selection**: Top models in the dropdown are optimized for speed and quality
3. **Web Search**: Use "Smart" mode to balance functionality and speed
4. **Chat Management**: Regularly clean up old chats to maintain performance

## 🔄 What's New in Enhanced Version

### Major Improvements
- ✅ **Fixed Dropdown Deduplication**: No more duplicate model entries
- ✅ **Real Provider Attribution**: Shows actual API provider used
- ✅ **Smart Web Search Default**: Automatically enabled with intelligent triggering
- ✅ **Fixed Regenerate**: Properly replaces responses instead of appending
- ✅ **Direct API Integration**: Real responses from multiple providers
- ✅ **Enhanced UI**: Modern, responsive design with better UX
- ✅ **Improved Error Handling**: Graceful fallbacks and meaningful error messages

### Technical Enhancements
- Async API architecture for better performance
- Comprehensive provider prioritization logic
- Smart model mapping and normalization
- Enhanced session and chat management
- Improved caching and data persistence

## 🆘 Support

If you encounter issues:

1. Check the console output for error messages
2. Verify your `.env` file configuration
3. Ensure all dependencies are installed
4. Check API key validity and quotas
5. Try restarting the application

## 📄 License

This project is provided as-is for educational and development purposes. Please respect the terms of service of all API providers used.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional API provider integrations
- Enhanced model performance tracking
- Advanced search functionality
- UI/UX improvements
- Mobile optimization

---

**Happy Chatting! 🎉**

For detailed technical information about the fixes implemented, see `IMPLEMENTED_FIXES_SUMMARY.md`.