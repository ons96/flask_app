#!/usr/bin/env python3
"""
Update LLM performance data by running the existing scraper and copying the data to the flask app.
"""
import os
import sys
import csv
import shutil
import datetime
import subprocess
from pathlib import Path

# Define paths
FLASK_APP_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_LEADERBOARD_DIR = os.path.join(os.path.dirname(FLASK_APP_DIR), 'LLM-Performance-Leaderboard')
SCRAPER_PATH = os.path.join(LLM_LEADERBOARD_DIR, 'llm_leaderboard_scraper.py')
PROVIDER_PERFORMANCE_PATH = os.path.join(FLASK_APP_DIR, 'provider_performance.csv')

def run_leaderboard_scraper():
    """Run the existing LLM leaderboard scraper script."""
    print(f"Running LLM leaderboard scraper from {SCRAPER_PATH}...")

    # Change to the leaderboard directory
    original_dir = os.getcwd()
    os.chdir(LLM_LEADERBOARD_DIR)

    try:
        # Run the scraper script
        result = subprocess.run([sys.executable, SCRAPER_PATH],
                               capture_output=True, text=True, check=True)
        print(result.stdout)

        # Find the latest CSV file in the directory
        csv_files = list(Path(LLM_LEADERBOARD_DIR).glob('llm_leaderboard_*.csv'))
        if not csv_files:
            print("No leaderboard CSV files found after running the scraper.")
            return None

        # Get the latest file by modification time
        latest_csv = max(csv_files, key=os.path.getmtime)
        print(f"Latest leaderboard file: {latest_csv}")

        return latest_csv
    except subprocess.CalledProcessError as e:
        print(f"Error running the scraper: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None
    finally:
        # Change back to the original directory
        os.chdir(original_dir)

def convert_to_provider_performance(leaderboard_path):
    """Convert the leaderboard data to provider_performance.csv format."""
    if not leaderboard_path or not os.path.exists(leaderboard_path):
        print("No valid leaderboard file to convert.")
        return

    print(f"Converting {leaderboard_path} to provider_performance.csv format...")

    # Read the leaderboard data
    leaderboard_data = []
    with open(leaderboard_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            leaderboard_data.append(row)

    # Convert to provider_performance.csv format
    provider_performance_data = []
    for row in leaderboard_data:
        provider_name = row.get('API Provider', '')
        model_name = row.get('Model', '')

        if not provider_name or not model_name:
            continue

        # Parse context window
        context_window = row.get('ContextWindow', '0')
        if context_window.lower().endswith('k'):
            context_window = int(float(context_window[:-1]) * 1000)
        elif context_window.lower().endswith('m'):
            context_window = int(float(context_window[:-1]) * 1000000)
        else:
            try:
                context_window = int(context_window)
            except (ValueError, TypeError):
                context_window = 0

        # Parse intelligence index
        intelligence_index = row.get('Artificial AnalysisIntelligence Index', 'N/A')
        if intelligence_index != 'N/A':
            try:
                intelligence_index = float(intelligence_index)
            except (ValueError, TypeError):
                intelligence_index = 'N/A'

        # Parse response time
        response_time = row.get('Total Response (s)', 'N/A')
        if response_time != 'N/A':
            try:
                response_time = float(response_time)
            except (ValueError, TypeError):
                response_time = 'N/A'

        # Parse tokens per second
        tokens_per_s = row.get('MedianTokens/s', 'N/A')
        if tokens_per_s != 'N/A':
            tokens_per_s = tokens_per_s.replace(',', '')
            try:
                tokens_per_s = float(tokens_per_s)
            except (ValueError, TypeError):
                tokens_per_s = 'N/A'

        # Create a record
        record = {
            'provider_name_scraped': provider_name,
            'model_name_scraped': model_name,
            'context_window': context_window,
            'intelligence_index': intelligence_index,
            'response_time_s': response_time,
            'tokens_per_s': tokens_per_s,
            'source_url': '',
            'last_updated_utc': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'is_free_source': 'true'  # Assuming all models are free as per your request
        }

        provider_performance_data.append(record)

    # Write to provider_performance.csv
    fieldnames = [
        'provider_name_scraped', 'model_name_scraped', 'context_window',
        'intelligence_index', 'response_time_s', 'tokens_per_s',
        'source_url', 'last_updated_utc', 'is_free_source'
    ]

    with open(PROVIDER_PERFORMANCE_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(provider_performance_data)

    print(f"Converted {len(provider_performance_data)} models to {PROVIDER_PERFORMANCE_PATH}")

def main():
    """Main function to update LLM data."""
    print("Updating LLM performance data...")

    # Always update the provider_performance.csv to include all models from the leaderboard
    if os.path.exists(PROVIDER_PERFORMANCE_PATH):
        with open(PROVIDER_PERFORMANCE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for _ in reader) - 1  # Subtract 1 for header
        
        print(f"Found existing provider_performance.csv with {row_count} entries.")
        print("Updating with the latest data from the leaderboard...")

    # Run the leaderboard scraper
    latest_leaderboard = run_leaderboard_scraper()

    if latest_leaderboard:
        # Convert to provider_performance.csv format
        convert_to_provider_performance(latest_leaderboard)
        print("LLM data update complete!")
    else:
        print("Failed to update LLM data.")

if __name__ == "__main__":
    main()
