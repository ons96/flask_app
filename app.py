"""
Vercel entrypoint for Flask app
Simply imports and exposes the app from your existing modules
"""

from app_enhanced_context_refreshed import app

# Vercel needs this to be exported as 'app'
__all__ = ['app']
