import os
import asyncio
import sys
import logging
import argparse
from g4f.api import run_api
from g4f.Provider import RetryProvider, __all__ as all_providers  # Import all providers dynamically
import aiohttp  # Required for custom timeout handling
from g4f.cookies import set_cookies_dir, read_cookie_files
import g4f.debug
from g4f.gui import run_gui

# Set up argument parser
parser = argparse.ArgumentParser(description='Run gpt4free server')
parser.add_argument('--port', type=int, default=1337, help='Port to run the server on (default: 1337)')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('g4f_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Optional: Enable debug logging to see cookie loading messages
g4f.debug.logging = True
# g4f.debug.version_check = False # Optional: Disable version check noise

try:
    # --- Define the *absolute* path to the DIRECTORY containing your cookie file ---
    # Use forward slashes or raw strings for Windows paths in Python
    cookies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'har_and_cookies')
    # Create the directory if it doesn't exist
    os.makedirs(cookies_dir, exist_ok=True)
    if not os.path.isdir(cookies_dir):
        print(f"Error: Cookies directory not found at '{cookies_dir}'")
        print("Please ensure the path is correct and the directory exists.")
    else:
        print(f"Setting cookies directory to: {cookies_dir}")
        set_cookies_dir(cookies_dir)

        # --- Rename your file for potentially better compatibility ---
        # It's often best if the filename matches the domain, e.g., google.com.json
        original_file = os.path.join(cookies_dir, "google_cookies.json")
        renamed_file = os.path.join(cookies_dir, "google.com.json") # Or just google.json

        if os.path.exists(original_file) and not os.path.exists(renamed_file):
             try:
                 os.rename(original_file, renamed_file)
                 print(f"Renamed '{original_file}' to '{renamed_file}'")
             except OSError as e:
                 print(f"Warning: Could not rename cookie file: {e}")

        # --- Read the cookie files from the specified directory ---
        print("Attempting to read cookie files...")
        # read_cookie_files() will read all valid .json and .har files in the directory
        read_cookie_files(cookies_dir)
        # If debug logging is on, you should see messages about cookies being added.
        print("Finished attempting to read cookie files. Check debug logs above for details.")

except Exception as e:
    print(f"An error occurred during cookie setup: {e}")
    import traceback
    traceback.print_exc() # Print full traceback for debugging

# 1. Configure debug mode
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("g4f")

# 2. Set API keys (optional)
os.environ["HUGGINGFACE_TOKEN"] = "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"  # Optional, verify if valid

# 3. Custom timeout for provider attempts (in seconds)
TIMEOUT_SECONDS = 10  # Adjust this value as needed

# 4. Dynamically get all available providers
provider_classes = []
for provider_name in all_providers:
    provider = getattr(sys.modules["g4f.Provider"], provider_name)
    if isinstance(provider, type) and provider_name != "RetryProvider":  # Exclude RetryProvider itself
        provider_classes.append(provider)

# 5. Patch RetryProvider to include timeout
async def patched_get_completion(self, *args, **kwargs):
    """Patch for RetryProvider to add timeout per provider attempt."""
    for provider in self.providers:
        logger.debug(f"Trying provider: {provider.__name__}")
        try:
            async with aiohttp.ClientSession() as session:
                # Wrap the provider call in a timeout
                async with asyncio.timeout(TIMEOUT_SECONDS):
                    response = await provider.create_async(*args, **kwargs)
                    if response:
                        logger.debug(f"Success with provider: {provider.__name__}")
                        return response
        except asyncio.TimeoutError:
            logger.warning(f"Provider {provider.__name__} timed out after {TIMEOUT_SECONDS} seconds")
            continue
        except Exception as e:
            logger.error(f"Provider {provider.__name__} failed: {str(e)}")
            continue
    raise Exception("All providers failed or timed out")

# Apply the patch to RetryProvider
RetryProvider.create_async = patched_get_completion

# 6. Main API setup
if __name__ == "__main__":
    # Windows event loop fix
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the GUI in a separate thread if needed
    # run_gui()  # Commented out to run in headless mode for server use
    
    logger.info(f"Starting gpt4free server on {args.host}:{args.port}")
    
    # Start the API server with the specified host and port
    run_api(
        host=args.host,
        port=args.port,
        debug=True
    )

    