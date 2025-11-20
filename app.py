"""
Vercel entrypoint for Flask app
Imports the main app and registers chatbot routes
"""

from app_enhanced_context_refreshed import app
from routes import chatbot_bp

# Register the chatbot blueprint
app.register_blueprint(chatbot_bp)

# Vercel needs this to be exported as 'app'
__all__ = ['app']
