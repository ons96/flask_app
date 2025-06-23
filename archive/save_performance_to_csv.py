def save_performance_to_csv(data, filepath=PERFORMANCE_CSV_PATH):
    """Saves the performance data list to a CSV file."""
    if not data:
        print("--- No performance data to save. --- ")
        return False
    # Ensure all expected keys are present in the first row for the header
    expected_keys = [
        'provider_name_scraped', 'model_name_scraped', 'context_window', 
        'context_window_int', 'intelligence_index', 'response_time_s', 
        'tokens_per_s', 'is_free_source'
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=expected_keys)
            writer.writeheader()
            for row in data:
                # Ensure all expected keys are present
                row_to_write = {key: row.get(key, '') for key in expected_keys}
                writer.writerow(row_to_write)
        print(f"--- Successfully saved {len(data)} performance data entries to {filepath} ---")
        return True
    except IOError as e:
        print(f"--- Error saving performance data to CSV: {e} ---")
        return False
    except Exception as e:
        print(f"--- Unexpected error saving performance data: {e} ---")
        return False
