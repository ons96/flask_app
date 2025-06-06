@echo off
echo ================================
echo Enhanced Flask LLM Chat App
echo ================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Display Python version
echo Python version:
python --version
echo.

:: Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
if exist "requirements.txt" (
    echo Installing/updating dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo WARNING: Some dependencies failed to install
        echo Continuing anyway...
    )
) else (
    echo Installing core dependencies...
    pip install flask flask-session g4f requests beautifulsoup4 duckduckgo-search aiohttp python-dotenv
)

:: Check if .env file exists
if not exist ".env" (
    echo.
    echo WARNING: No .env file found!
    echo Creating sample .env file...
    echo # Enhanced Flask LLM Chat App Configuration > .env
    echo # Add your API keys here >> .env
    echo GROQ_API_KEY=your_groq_api_key_here >> .env
    echo CEREBRAS_API_KEY=your_cerebras_api_key_here >> .env
    echo GOOGLE_API_KEY=your_google_api_key_here >> .env
    echo SECRET_KEY=your_random_secret_key_here >> .env
    echo.
    echo Please edit .env file with your actual API keys
    echo You can run the app without API keys but functionality will be limited
    echo.
)

:: Run the enhanced app
echo.
echo ================================
echo Starting Enhanced Flask App...
echo ================================
echo.
echo The app will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app_fixed_comprehensive_final_updated.py

:: If the app exits, show a message
echo.
echo ================================
echo App has stopped
echo ================================
pause