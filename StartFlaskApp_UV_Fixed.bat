@echo off
echo Starting Flask App in WSL...
echo.

REM Change to the correct directory and run the Flask app with uv for fast installation
wsl -e bash -c "cd /mnt/c/Users/owens/flask_app && export PATH=\"$HOME/.local/bin:$PATH\" && echo 'Creating virtual environment with uv...' && uv venv && echo 'Installing dependencies with uv (much faster)...' && uv pip install -r requirements.txt && echo 'Starting Flask app...' && source .venv/bin/activate && python app_enhanced_context.py"

echo.
echo Flask app has stopped. Press any key to close this window.
pause >nul