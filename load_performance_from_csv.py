def load_performance_from_csv(filepath=PERFORMANCE_CSV_PATH):
    """Loads performance data from a CSV file."""
    data = []
    if not os.path.exists(filepath):
        print(f"--- Performance data CSV not found at {filepath}. ---")
        return data
    
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert numeric fields
                try:
                    if 'intelligence_index' in row:
                        row['intelligence_index'] = int(row['intelligence_index']) if row['intelligence_index'].isdigit() else 0
                    if 'response_time_s' in row:
                        row['response_time_s'] = float(row['response_time_s']) if row['response_time_s'] else float('inf')
                    if 'tokens_per_s' in row:
                        row['tokens_per_s'] = float(row['tokens_per_s']) if row['tokens_per_s'] else 0.0
                    if 'context_window_int' in row:
                        row['context_window_int'] = int(row['context_window_int']) if row['context_window_int'].isdigit() else 0
                    if 'is_free_source' in row:
                        row['is_free_source'] = row['is_free_source'].lower() == 'true'
                except (ValueError, TypeError) as e:
                    print(f"--- Warning: Error converting numeric fields in row {row}: {e} ---")
                
                data.append(row)
        
        print(f"--- Successfully loaded {len(data)} performance data entries from {filepath} ---")
    except IOError as e:
        print(f"--- Error loading performance data from CSV: {e} ---")
    except Exception as e:
        print(f"--- Unexpected error loading performance data: {e} ---")
    
    return data
