#!/usr/bin/env python3
"""
Updated scraper for LLM provider performance data from artificialanalysis.ai
This uses multiple approaches to get the data and updates the CSV file.
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import os
import time
from datetime import datetime
import re

# Try to import selenium, handle if not available
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available, will use requests-only scraping")

# Configuration
PROVIDER_PERFORMANCE_URL = "https://artificialanalysis.ai/leaderboards/providers"
HUGGINGFACE_URL = "https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard"
CSV_PATH = os.path.join(os.path.dirname(__file__), "provider_performance.csv")

def scrape_with_selenium(url, timeout=30):
    """
    Scrape using Selenium to handle dynamic content.
    """
    if not SELENIUM_AVAILABLE:
        print("--- Selenium not available, skipping ---")
        return None
        
    print(f"--- Attempting to scrape with Selenium: {url} ---")
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Wait for the page to load
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        
        # Get the page source after JavaScript execution
        page_source = driver.page_source
        driver.quit()
        
        return page_source
        
    except Exception as e:
        print(f"--- Selenium scraping failed: {e} ---")
        if 'driver' in locals():
            driver.quit()
        return None

def scrape_with_requests(url):
    """
    Try basic requests scraping first.
    """
    print(f"--- Attempting to scrape with requests: {url} ---")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        return response.text
        
    except Exception as e:
        print(f"--- Requests scraping failed: {e} ---")
        return None

def extract_json_data(html_content):
    """
    Try to extract JSON data from script tags (Next.js data).
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Look for Next.js data
    script_tags = soup.find_all('script')
    for script in script_tags:
        if script.string and 'window.__NEXT_DATA__' in script.string:
            try:
                json_str = script.string.strip()
                json_str = json_str.replace('window.__NEXT_DATA__ = ', '')
                if json_str.endswith(';'):
                    json_str = json_str[:-1]
                
                data = json.loads(json_str)
                print("--- Found Next.js data ---")
                return data
            except json.JSONDecodeError as e:
                print(f"--- JSON decode error: {e} ---")
                continue
    
    # Look for other potential data sources
    for script in script_tags:
        if script.string and ('models' in script.string or 'leaderboard' in script.string):
            try:
                # Try to extract JSON-like data
                text = script.string
                json_matches = re.findall(r'\{[^{}]*"models"[^{}]*\}', text)
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        print("--- Found embedded JSON data ---")
                        return data
                    except:
                        continue
            except:
                continue
    
    return None

def parse_table_data(html_content):
    """
    Enhanced table parsing for current artificial analysis structure.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try multiple selectors for the table
    table = None
    selectors = [
        'table',
        '[role="table"]',
        '.table',
        'div[data-testid="table"]'
    ]
    
    for selector in selectors:
        table = soup.select_one(selector)
        if table:
            print(f"--- Found table using selector: {selector} ---")
            break
    
    if not table:
        # Try finding in main content
        main_content = soup.find('main')
        if main_content:
            table = main_content.find('table')
            if table:
                print("--- Found table in main content ---")
    
    if not table:
        print("--- No table found in HTML ---")
        return []
    
    print("--- Found table, parsing data ---")
    
    performance_data = []
    
    # Enhanced row parsing
    rows = table.find_all('tr') or table.find_all('[role="row"]')
    if not rows:
        tbody = table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
    
    print(f"--- Found {len(rows)} rows ---")
    
    for row_index, row in enumerate(rows[1:], 1):  # Skip header
        cells = row.find_all(['td', 'th']) or row.find_all('[role="cell"]')
        if len(cells) < 4:
            continue
            
        try:
            # Extract data with better error handling
            provider = extract_provider_name(cells[0])
            model = cells[1].get_text(strip=True)
            context_window = cells[2].get_text(strip=True) if len(cells) > 2 else ''
            
            # Intelligence index (usually 4th column)
            intelligence_str = cells[3].get_text(strip=True) if len(cells) > 3 else '0'
            intelligence_index = parse_float(intelligence_str)
            
            # Look for response time in later columns
            response_time = float('inf')
            tokens_per_s = 0
            
            for i, cell in enumerate(cells[4:], 4):
                text = cell.get_text(strip=True)
                
                # Response time column (usually contains 's' suffix)
                if 's' in text.lower() and any(c.isdigit() for c in text):
                    response_time = parse_float(text.replace('s', ''))
                    break
                
                # Tokens per second (usually a number with commas)
                if ',' in text or (text.replace('.', '').replace(',', '').isdigit() and len(text) > 2):
                    tokens_per_s = parse_float(text.replace(',', ''))
            
            if provider and model and intelligence_index > 0:
                performance_data.append({
                    'provider_name_scraped': provider,
                    'model_name_scraped': model,
                    'context_window': context_window,
                    'intelligence_index': intelligence_index,
                    'response_time_s': response_time,
                    'tokens_per_s': tokens_per_s,
                    'source_url': PROVIDER_PERFORMANCE_URL,
                    'last_updated_utc': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'is_free_source': 'true'
                })
                
        except Exception as e:
            print(f"--- Error parsing row {row_index}: {e} ---")
            continue
    
    print(f"--- Successfully parsed {len(performance_data)} entries ---")
    return performance_data

def extract_provider_name(cell):
    """Extract provider name from cell, handling images and text."""
    # Try image alt text first
    img = cell.find('img')
    if img and img.get('alt'):
        return img['alt'].replace(' logo', '').strip()
    
    # Fallback to text content
    return cell.get_text(strip=True)

def parse_float(text_str):
    """Safely parse float from text with fallback."""
    if not text_str or text_str.lower() in ['n/a', 'na', '']:
        return 0.0
    
    # Remove common non-numeric characters
    cleaned = re.sub(r'[^0-9.-]', '', str(text_str))
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0

def parse_context_window(context_str):
    """Parse context window string to integer."""
    if not context_str:
        return 0
    
    context_str = str(context_str).lower().strip()
    
    if context_str in ['', 'n/a', 'none']:
        return 0
    
    # Remove common suffixes and convert
    multiplier = 1
    if context_str.endswith('k'):
        multiplier = 1000
        context_str = context_str[:-1]
    elif context_str.endswith('m'):
        multiplier = 1000000
        context_str = context_str[:-1]
    
    try:
        return int(float(context_str) * multiplier)
    except (ValueError, TypeError):
        return 0

def save_performance_data(data, filepath=CSV_PATH):
    """Save performance data to CSV."""
    if not data:
        print("--- No data to save ---")
        return False
    
    fieldnames = [
        'provider_name_scraped', 'model_name_scraped', 'context_window',
        'intelligence_index', 'agentic_coding_score', 'response_time_s', 'tokens_per_s',
        'source_url', 'last_updated_utc', 'is_free_source'
    ]
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"--- Saved {len(data)} entries to {filepath} ---")
        return True
        
    except Exception as e:
        print(f"--- Error saving CSV: {e} ---")
        return False

def scrape_provider_performance(max_retries=3):
    """
    Enhanced main scraping function with retry logic and better error handling.
    """
    print("--- Starting provider performance scraping ---")
    
    for attempt in range(max_retries):
        print(f"--- Attempt {attempt + 1} of {max_retries} ---")
        
        # Try scraping with requests first
        html_content = scrape_with_requests(PROVIDER_PERFORMANCE_URL)
        
        if html_content:
            # Try to extract JSON data first
            json_data = extract_json_data(html_content)
            if json_data:
                print("--- Successfully extracted JSON data ---")
                # Process JSON data if needed (implement based on structure)
                
            # Try to parse table data
            performance_data = parse_table_data(html_content)
            
            if performance_data:
                print(f"--- Successfully scraped {len(performance_data)} entries with requests ---")
                return performance_data
        
        # If requests failed, try Selenium
        print("--- Requests method failed, trying Selenium ---")
        try:
            html_content = scrape_with_selenium(PROVIDER_PERFORMANCE_URL)
            
            if html_content:
                performance_data = parse_table_data(html_content)
                
                if performance_data:
                    print(f"--- Successfully scraped {len(performance_data)} entries with Selenium ---")
                    return performance_data
        except Exception as e:
            print(f"--- Selenium method failed: {e} ---")
        
        # Try the Hugging Face version as fallback
        print("--- Trying Hugging Face version ---")
        try:
            html_content = scrape_with_selenium(HUGGINGFACE_URL)
            
            if html_content:
                performance_data = parse_table_data(html_content)
                
                if performance_data:
                    print(f"--- Successfully scraped {len(performance_data)} entries from Hugging Face ---")
                    return performance_data
        except Exception as e:
            print(f"--- Hugging Face method failed: {e} ---")
        
        # Wait before retrying (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"--- Waiting {wait_time} seconds before retry ---")
            time.sleep(wait_time)
    
    print("--- All scraping methods failed after all retries ---")
    return []

def update_csv_file():
    """
    Update the CSV file with fresh data.
    """
    print("--- Updating provider performance CSV ---")
    
    data = scrape_provider_performance()
    
    if data:
        success = save_performance_data(data)
        if success:
            print(f"--- Successfully updated CSV with {len(data)} entries ---")
            return True
    
    print("--- Failed to update CSV ---")
    return False

if __name__ == "__main__":
    update_csv_file()