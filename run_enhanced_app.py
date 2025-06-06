#!/usr/bin/env python3
"""
Run script for the Enhanced Flask LLM Chat Application
"""

import os
import sys
from pathlib import Path

def main():
    """Main function to run the enhanced Flask app."""
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Set environment variables if not already set
    env_file = current_dir / '.env'
    if env_file.exists():
        print("📋 Loading environment variables from .env file...")
        from dotenv import load_dotenv
        load_dotenv(env_file)
    else:
        print("⚠️  No .env file found. Please ensure API keys are set as environment variables.")
    
    # Import and run the enhanced app
    try:
        print("🚀 Starting Enhanced LLM Chat Application...")
        print("="*60)
        
        # Import the enhanced app
        import app_fixed_comprehensive_final_updated as app
        
        # The app will run automatically when imported due to the if __name__ == '__main__' block
        
    except ImportError as e:
        print(f"❌ Error importing enhanced app: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install flask flask-session g4f requests beautifulsoup4 duckduckgo-search aiohttp python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()