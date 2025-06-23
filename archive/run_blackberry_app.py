#!/usr/bin/env python3
"""
Run script for BlackBerry Optimized LLM Chat Application
Optimized for small screens and low-end hardware
"""

import os
import sys
from pathlib import Path

def main():
    """Main function to run the BlackBerry optimized Flask app."""
    
    print("=" * 60)
    print("  BLACKBERRY OPTIMIZED LLM CHAT APPLICATION")
    print("=" * 60)
    print("  Optimized for:")
    print("  - Small screens (BlackBerry Classic)")
    print("  - Low-end hardware")
    print("  - No JavaScript required")
    print("  - Minimal bandwidth usage")
    print("=" * 60)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Set environment variables if not already set
    env_file = current_dir / '.env'
    if env_file.exists():
        print("📋 Loading environment variables from .env file...")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            print("⚠️  python-dotenv not installed. Please install it:")
            print("   pip install python-dotenv")
    else:
        print("⚠️  No .env file found.")
        print("   The app will work with limited functionality.")
        print("   For full features, create .env with API keys:")
        print("   GROQ_API_KEY=your_key")
        print("   CEREBRAS_API_KEY=your_key")
        print("   GOOGLE_API_KEY=your_key")
    
    print("\n🚀 Starting BlackBerry Optimized App...")
    
    # Import and run the BlackBerry optimized app
    try:
        import app_blackberry_optimized
        print("✅ App imported successfully")
        
    except ImportError as e:
        print(f"❌ Error importing BlackBerry app: {e}")
        print("\nMissing dependencies. Please install:")
        print("pip install flask flask-session g4f requests beautifulsoup4 duckduckgo-search aiohttp python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()