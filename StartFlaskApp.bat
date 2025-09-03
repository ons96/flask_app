@echo off
echo Starting Flask App in WSL...
echo.

REM Change to the correct directory and run the Flask app with virtual environment
wsl -e bash -c "cd /mnt/c/Users/owens/flask_app && echo 'Setting up virtual environment...' && python3 -m venv venv 2>/dev/null && echo 'Activating virtual environment...' && source venv/bin/activate && echo 'Installing dependencies...' && pip install -r requirements.txt && echo 'Starting Flask app...' && python app_enhanced_context.py"

echo.
echo Flask app has stopped. Press any key to close this window.
pause >nul