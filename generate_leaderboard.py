#!/usr/bin/env python3
"""
Generate a comprehensive LLM performance leaderboard by combining data from multiple sources.
"""

import os
import csv
import json
import datetime
from collections import defaultdict

# Define paths to source data files
ARTIFICIAL_ANALYSIS_PATH = os.path.join('..', 'llm-leaderboard', 'artificial_analysis_models_20250522_174848.csv')
PROVIDER_PERFORMANCE_PATH = 'provider_performance.csv'
EXISTING_LEADERBOARD_PATH = os.path.join('..', 'LLM-Performance-Leaderboard', 'llm_leaderboard_20250521_013630.csv')

# Output path for the new leaderboard
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join('..', 'LLM-Performance-Leaderboard', f'llm_leaderboard_{timestamp}.csv')

def parse_context_window(context_window_str):
    """Parse context window size from string format."""
    if not context_window_str or context_window_str == '0':
        return 0
    
    try:
        if 'k' in context_window_str.lower():
            return int(float(context_window_str.lower().replace('k', '')) * 1000)
        elif 'm' in context_window_str.lower():
            return int(float(context_window_str.lower().replace('m', '')) * 1000000)
        else:
            return int(context_window_str)
    except (ValueError, TypeError):
        return 0

def format_price(price):
    """Format price as a string with $ sign."""
    if price is None or price == '' or price == 'N/A':
        return 'N/A'
    try:
        price_float = float(price)
        return f"${price_float:.2f}"
    except (ValueError, TypeError):
        return price

def format_tokens_per_second(tokens_per_s):
    """Format tokens per second with commas for thousands."""
    if tokens_per_s is None or tokens_per_s == '' or tokens_per_s == 'N/A':
        return 'N/A'
    try:
        tokens_float = float(tokens_per_s)
        return f"{tokens_float:,.1f}"
    except (ValueError, TypeError):
        return tokens_per_s

def load_artificial_analysis_data():
    """Load data from the Artificial Analysis CSV file."""
    models_data = {}
    
    try:
        with open(ARTIFICIAL_ANALYSIS_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get('name', '')
                if not model_name:
                    continue
                
                # Parse evaluations JSON
                evaluations = {}
                try:
                    evaluations_str = row.get('evaluations', '{}')
                    evaluations = json.loads(evaluations_str)
                except json.JSONDecodeError:
                    pass
                
                # Parse pricing JSON
                pricing = {}
                try:
                    pricing_str = row.get('pricing', '{}')
                    pricing = json.loads(pricing_str)
                except json.JSONDecodeError:
                    pass
                
                # Get intelligence index
                intel_index = evaluations.get('artificial_analysis_intelligence_index')
                if intel_index is not None:
                    intel_index = round(float(intel_index))
                
                # Get price per 1M tokens (blended)
                price_1m = pricing.get('price_1m_blended_3_to_1')
                
                # Get tokens per second
                tokens_per_s = row.get('median_output_tokens_per_second', '')
                if tokens_per_s:
                    tokens_per_s = float(tokens_per_s)
                
                # Store data
                models_data[model_name] = {
                    'model_creator': row.get('model_creator_name', ''),
                    'intelligence_index': intel_index,
                    'price_1m': price_1m,
                    'tokens_per_s': tokens_per_s
                }
        
        print(f"Loaded {len(models_data)} models from Artificial Analysis data")
    except FileNotFoundError:
        print(f"Warning: Could not find Artificial Analysis data file at {ARTIFICIAL_ANALYSIS_PATH}")
    
    return models_data

def load_provider_performance_data():
    """Load data from the provider performance CSV file."""
    performance_data = {}
    
    try:
        with open(PROVIDER_PERFORMANCE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                provider = row.get('provider_name_scraped', '')
                model = row.get('model_name_scraped', '')
                
                if not provider or not model:
                    continue
                
                key = f"{provider}:{model}"
                
                # Parse context window
                context_window = parse_context_window(row.get('context_window', '0'))
                
                # Get intelligence index
                intel_index = row.get('intelligence_index', '')
                if intel_index and intel_index != 'N/A':
                    intel_index = round(float(intel_index))
                else:
                    intel_index = None
                
                # Get response time
                response_time = row.get('response_time_s', '')
                if response_time and response_time != 'N/A':
                    response_time = float(response_time)
                else:
                    response_time = None
                
                # Get tokens per second
                tokens_per_s = row.get('tokens_per_s', '')
                if tokens_per_s and tokens_per_s != 'N/A':
                    tokens_per_s = float(tokens_per_s)
                else:
                    tokens_per_s = None
                
                # Store data
                performance_data[key] = {
                    'provider': provider,
                    'model': model,
                    'context_window': context_window,
                    'intelligence_index': intel_index,
                    'response_time': response_time,
                    'tokens_per_s': tokens_per_s,
                    'is_free': row.get('is_free_source', '').lower() == 'true'
                }
        
        print(f"Loaded {len(performance_data)} provider-model combinations from performance data")
    except FileNotFoundError:
        print(f"Warning: Could not find provider performance data file at {PROVIDER_PERFORMANCE_PATH}")
    
    return performance_data

def load_existing_leaderboard_data():
    """Load data from the existing leaderboard CSV file."""
    leaderboard_data = {}
    
    try:
        with open(EXISTING_LEADERBOARD_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                provider = row.get('API Provider', '')
                model = row.get('Model', '')
                
                if not provider or not model:
                    continue
                
                key = f"{provider}:{model}"
                
                # Parse context window
                context_window_str = row.get('ContextWindow', '')
                context_window = parse_context_window(context_window_str)
                
                # Get intelligence index
                intel_index = row.get('Artificial AnalysisIntelligence Index', '')
                if intel_index and intel_index != 'N/A':
                    intel_index = int(intel_index)
                else:
                    intel_index = None
                
                # Get price
                price_str = row.get('BlendedUSD/1M Tokens', '')
                if price_str and price_str.startswith('$'):
                    price = float(price_str[1:])
                else:
                    price = None
                
                # Get tokens per second
                tokens_per_s_str = row.get('MedianTokens/s', '')
                if tokens_per_s_str and tokens_per_s_str != 'N/A':
                    tokens_per_s = float(tokens_per_s_str.replace(',', ''))
                else:
                    tokens_per_s = None
                
                # Get response time
                response_time = row.get('Total Response (s)', '')
                if response_time and response_time != 'N/A':
                    response_time = float(response_time)
                else:
                    response_time = None
                
                # Get first chunk time
                first_chunk = row.get('MedianFirst Chunk (s)', '')
                if first_chunk and first_chunk != 'N/A':
                    first_chunk = float(first_chunk)
                else:
                    first_chunk = None
                
                # Get reasoning time
                reasoning_time = row.get('ReasoningTime (s)', '')
                
                # Store data
                leaderboard_data[key] = {
                    'provider': provider,
                    'model': model,
                    'context_window': context_window,
                    'context_window_str': context_window_str,
                    'intelligence_index': intel_index,
                    'price': price,
                    'tokens_per_s': tokens_per_s,
                    'response_time': response_time,
                    'first_chunk': first_chunk,
                    'reasoning_time': reasoning_time,
                    'further_analysis': row.get('FurtherAnalysis', '')
                }
        
        print(f"Loaded {len(leaderboard_data)} models from existing leaderboard")
    except FileNotFoundError:
        print(f"Warning: Could not find existing leaderboard file at {EXISTING_LEADERBOARD_PATH}")
    
    return leaderboard_data

def merge_data(artificial_data, performance_data, leaderboard_data):
    """Merge data from all sources into a comprehensive dataset."""
    merged_data = []
    
    # First, process existing leaderboard data
    for key, data in leaderboard_data.items():
        provider = data['provider']
        model = data['model']
        
        # Create a new entry
        entry = {
            'API Provider': provider,
            'Model': model,
            'ContextWindow': data['context_window_str'],
            'Artificial AnalysisIntelligence Index': data['intelligence_index'],
            'BlendedUSD/1M Tokens': format_price(data['price']),
            'MedianTokens/s': format_tokens_per_second(data['tokens_per_s']),
            'MedianFirst Chunk (s)': data['first_chunk'],
            'Total Response (s)': data['response_time'],
            'ReasoningTime (s)': data['reasoning_time'],
            'FurtherAnalysis': data['further_analysis']
        }
        
        merged_data.append(entry)
    
    # Process performance data to add new entries or update existing ones
    for key, data in performance_data.items():
        provider = data['provider']
        model = data['model']
        
        # Check if this provider-model combination already exists
        existing_entry = None
        for entry in merged_data:
            if entry['API Provider'] == provider and entry['Model'] == model:
                existing_entry = entry
                break
        
        if existing_entry:
            # Update existing entry with new data if available
            if data['intelligence_index'] is not None and (existing_entry['Artificial AnalysisIntelligence Index'] == '' or existing_entry['Artificial AnalysisIntelligence Index'] == 'N/A'):
                existing_entry['Artificial AnalysisIntelligence Index'] = data['intelligence_index']
            
            if data['tokens_per_s'] is not None and (existing_entry['MedianTokens/s'] == '' or existing_entry['MedianTokens/s'] == 'N/A'):
                existing_entry['MedianTokens/s'] = format_tokens_per_second(data['tokens_per_s'])
            
            if data['response_time'] is not None and (existing_entry['Total Response (s)'] == '' or existing_entry['Total Response (s)'] == 'N/A'):
                existing_entry['Total Response (s)'] = data['response_time']
            
            if data['context_window'] > 0 and (existing_entry['ContextWindow'] == '' or existing_entry['ContextWindow'] == 'N/A'):
                if data['context_window'] >= 1000000:
                    context_str = f"{data['context_window'] // 1000000}m"
                elif data['context_window'] >= 1000:
                    context_str = f"{data['context_window'] // 1000}k"
                else:
                    context_str = str(data['context_window'])
                existing_entry['ContextWindow'] = context_str
        else:
            # Create a new entry
            if data['context_window'] >= 1000000:
                context_str = f"{data['context_window'] // 1000000}m"
            elif data['context_window'] >= 1000:
                context_str = f"{data['context_window'] // 1000}k"
            else:
                context_str = str(data['context_window']) if data['context_window'] > 0 else 'N/A'
            
            entry = {
                'API Provider': provider,
                'Model': model,
                'ContextWindow': context_str,
                'Artificial AnalysisIntelligence Index': data['intelligence_index'] if data['intelligence_index'] is not None else 'N/A',
                'BlendedUSD/1M Tokens': 'N/A',  # We don't have price in performance data
                'MedianTokens/s': format_tokens_per_second(data['tokens_per_s']),
                'MedianFirst Chunk (s)': 'N/A',  # We don't have first chunk time in performance data
                'Total Response (s)': data['response_time'] if data['response_time'] is not None else 'N/A',
                'ReasoningTime (s)': 'N/A',  # We don't have reasoning time in performance data
                'FurtherAnalysis': 'ModelProviders'
            }
            
            merged_data.append(entry)
    
    # Process artificial analysis data to add price information and other details
    for model_name, data in artificial_data.items():
        # Try to find matching entries
        for entry in merged_data:
            model_match = False
            
            # Check for exact match
            if entry['Model'].lower() == model_name.lower():
                model_match = True
            
            # Check for partial match (e.g., "llama-3-1-8b" vs "Llama 3.1 8B")
            if not model_match:
                normalized_entry_model = entry['Model'].lower().replace(' ', '-').replace('.', '-')
                normalized_model_name = model_name.lower().replace(' ', '-').replace('.', '-')
                
                if normalized_entry_model in normalized_model_name or normalized_model_name in normalized_entry_model:
                    model_match = True
            
            if model_match:
                # Update price if available
                if data['price_1m'] is not None and (entry['BlendedUSD/1M Tokens'] == '' or entry['BlendedUSD/1M Tokens'] == 'N/A'):
                    entry['BlendedUSD/1M Tokens'] = format_price(data['price_1m'])
                
                # Update intelligence index if available
                if data['intelligence_index'] is not None and (entry['Artificial AnalysisIntelligence Index'] == '' or entry['Artificial AnalysisIntelligence Index'] == 'N/A'):
                    entry['Artificial AnalysisIntelligence Index'] = data['intelligence_index']
                
                # Update tokens per second if available
                if data['tokens_per_s'] is not None and (entry['MedianTokens/s'] == '' or entry['MedianTokens/s'] == 'N/A'):
                    entry['MedianTokens/s'] = format_tokens_per_second(data['tokens_per_s'])
    
    # Sort the merged data by response time (ascending)
    merged_data.sort(key=lambda x: float(x['Total Response (s)']) if x['Total Response (s)'] != 'N/A' else float('inf'))
    
    return merged_data

def write_output(merged_data):
    """Write the merged data to a CSV file."""
    fieldnames = [
        'API Provider', 'Model', 'ContextWindow', 'Artificial AnalysisIntelligence Index',
        'BlendedUSD/1M Tokens', 'MedianTokens/s', 'MedianFirst Chunk (s)',
        'Total Response (s)', 'ReasoningTime (s)', 'FurtherAnalysis'
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)
    
    print(f"Wrote {len(merged_data)} models to {OUTPUT_PATH}")

def main():
    """Main function to generate the leaderboard."""
    print("Loading data from sources...")
    artificial_data = load_artificial_analysis_data()
    performance_data = load_provider_performance_data()
    leaderboard_data = load_existing_leaderboard_data()
    
    print("Merging data...")
    merged_data = merge_data(artificial_data, performance_data, leaderboard_data)
    
    print("Writing output...")
    write_output(merged_data)
    
    print("Done!")

if __name__ == "__main__":
    main()