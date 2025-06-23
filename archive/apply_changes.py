# Script to apply all changes to app.py
import re
import os

# Read the original app.py file
with open('C:/Users/owens/Coding Projects/flask_app/app.py', 'r', encoding='utf-8') as f:
    app_content = f.read()

# Read our new functions
with open('C:/Users/owens/Coding Projects/model_utils.py', 'r', encoding='utf-8') as f:
    model_utils_content = f.read()

with open('C:/Users/owens/Coding Projects/initialize_model_cache.py', 'r', encoding='utf-8') as f:
    initialize_model_cache_content = f.read()

with open('C:/Users/owens/Coding Projects/scrape_provider_performance.py', 'r', encoding='utf-8') as f:
    scrape_provider_performance_content = f.read()

with open('C:/Users/owens/Coding Projects/save_performance_to_csv.py', 'r', encoding='utf-8') as f:
    save_performance_to_csv_content = f.read()

with open('C:/Users/owens/Coding Projects/load_performance_from_csv.py', 'r', encoding='utf-8') as f:
    load_performance_from_csv_content = f.read()

with open('C:/Users/owens/Coding Projects/prioritize_providers.py', 'r', encoding='utf-8') as f:
    prioritize_providers_content = f.read()

# Replace the normalize_model_name function
app_content = re.sub(
    r'def normalize_model_name\(name\):.*?return.*?\n',
    model_utils_content + '\n',
    app_content,
    flags=re.DOTALL
)

# Replace the initialize_model_cache function
app_content = re.sub(
    r'def initialize_model_cache\(\):.*?print\(f"=== \[CACHE\] Initialized with \{len\(default_models\)\} default models ===\\n"\)',
    initialize_model_cache_content,
    app_content,
    flags=re.DOTALL
)

# Replace the scrape_provider_performance function
app_content = re.sub(
    r'def scrape_provider_performance\(url=PROVIDER_PERFORMANCE_URL\):.*?return performance_data',
    scrape_provider_performance_content,
    app_content,
    flags=re.DOTALL
)

# Replace the save_performance_to_csv function
app_content = re.sub(
    r'def save_performance_to_csv\(data, filepath=PERFORMANCE_CSV_PATH\):.*?return False',
    save_performance_to_csv_content,
    app_content,
    flags=re.DOTALL
)

# Replace the load_performance_from_csv function
app_content = re.sub(
    r'def load_performance_from_csv\(filepath=PERFORMANCE_CSV_PATH\):.*?return data',
    load_performance_from_csv_content,
    app_content,
    flags=re.DOTALL
)

# Add the prioritize_providers_for_model function after the load_performance_from_csv function
app_content = app_content.replace(
    'def load_performance_from_csv(filepath=PERFORMANCE_CSV_PATH):',
    'def load_performance_from_csv(filepath=PERFORMANCE_CSV_PATH):'
)
app_content = app_content.replace(
    'return data\n',
    'return data\n\n' + prioritize_providers_content + '\n'
)

# Add DISPLAY_NAME_TO_MODEL_MAP to global variables
app_content = app_content.replace(
    'CACHED_AVAILABLE_MODELS_SORTED_LIST = []',
    'CACHED_AVAILABLE_MODELS_SORTED_LIST = []\nDISPLAY_NAME_TO_MODEL_MAP = {}'
)

# Write the modified content to a new file
with open('C:/Users/owens/Coding Projects/flask_app/app_new.py', 'w', encoding='utf-8') as f:
    f.write(app_content)

print("Changes applied successfully to app_new.py")
