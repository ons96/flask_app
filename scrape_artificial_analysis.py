#!/usr/bin/env python3
"""
Scrape model data from Artificial Analysis website.
"""

import os
import csv
import json
import time
import datetime
import requests
from bs4 import BeautifulSoup

# Output path for the scraped data
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join('..', 'llm-leaderboard', f'artificial_analysis_models_{timestamp}.csv')

# URL to scrape
ARTIFICIAL_ANALYSIS_URL = "https://artificialanalysis.ai/leaderboard"

def scrape_artificial_analysis():
    """Scrape model data from Artificial Analysis website."""
    print(f"Scraping data from {ARTIFICIAL_ANALYSIS_URL}...")
    
    try:
        # Send a GET request to the website
        response = requests.get(ARTIFICIAL_ANALYSIS_URL, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the script tag containing the model data
        script_tags = soup.find_all('script')
        model_data = []
        
        for script in script_tags:
            if script.string and 'window.__NEXT_DATA__' in script.string:
                # Extract the JSON data
                json_str = script.string.strip()
                json_str = json_str.replace('window.__NEXT_DATA__ = ', '')
                
                # Parse the JSON data
                data = json.loads(json_str)
                
                # Extract model data from the JSON
                if 'props' in data and 'pageProps' in data['props'] and 'models' in data['props']['pageProps']:
                    model_data = data['props']['pageProps']['models']
                    break
        
        if not model_data:
            print("Could not find model data in the page.")
            return []
        
        # Process the model data
        processed_data = []
        for model in model_data:
            # Extract model information
            model_id = model.get('id', '')
            model_name = model.get('name', '')
            model_slug = model.get('slug', '')
            model_creator_name = model.get('model_creator', {}).get('name', '')
            model_creator_slug = model.get('model_creator', {}).get('slug', '')
            
            # Extract evaluations
            evaluations = {}
            for eval_name, eval_value in model.get('evaluations', {}).items():
                evaluations[eval_name] = eval_value
            
            # Extract pricing
            pricing = {}
            for price_name, price_value in model.get('pricing', {}).items():
                pricing[price_name] = price_value
            
            # Extract median output tokens per second
            median_output_tokens_per_second = model.get('median_output_tokens_per_second', None)
            
            # Create a record
            record = {
                'id': model_id,
                'name': model_name,
                'slug': model_slug,
                'model_creator_name': model_creator_name,
                'model_creator_slug': model_creator_slug,
                'evaluations': json.dumps(evaluations),
                'pricing': json.dumps(pricing),
                'median_output_tokens_per_second': median_output_tokens_per_second
            }
            
            processed_data.append(record)
        
        print(f"Scraped {len(processed_data)} models from Artificial Analysis.")
        return processed_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error scraping Artificial Analysis: {e}")
        return []

def write_output(data):
    """Write the scraped data to a CSV file."""
    if not data:
        print("No data to write.")
        return
    
    fieldnames = [
        'id', 'name', 'slug', 'model_creator_name', 'model_creator_slug',
        'evaluations', 'pricing', 'median_output_tokens_per_second'
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Wrote {len(data)} models to {OUTPUT_PATH}")

def main():
    """Main function to scrape Artificial Analysis."""
    data = scrape_artificial_analysis()
    write_output(data)
    print("Done!")

if __name__ == "__main__":
    main()