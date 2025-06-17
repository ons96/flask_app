def scrape_provider_performance(url=PROVIDER_PERFORMANCE_URL):
    """Fetches and parses the provider performance table."""
    print(f"--- Scraping provider performance data from: {url} ---")
    performance_data = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table')
        if not table:
             main_content = soup.find('main')
             if main_content:
                 tables = main_content.find_all('table')
                 if tables: table = tables[0]
        if not table:
            print("--- Error: Could not find the performance table on the page. ---")
            return []
        tbody = table.find('tbody')
        if not tbody:
            print("--- Error: Found table but could not find tbody. ---")
            return []
        rows = tbody.find_all('tr')
        print(f"--- Found {len(rows)} rows in the table body. ---")
        # Expected column indices (adjust if the table layout changes)
        PROVIDER_COL = 0
        MODEL_COL = 1
        CONTEXT_WINDOW_COL = 2
        INTELLIGENCE_INDEX_COL = 3
        TOKENS_PER_S_COL = 5      # Assuming 6th column
        RESPONSE_TIME_COL = 7   # Assuming 8th column
        EXPECTED_COLS = max(PROVIDER_COL, MODEL_COL, INTELLIGENCE_INDEX_COL, TOKENS_PER_S_COL, RESPONSE_TIME_COL) + 1

        for row_index, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= EXPECTED_COLS:
                try:
                    provider_img = cols[PROVIDER_COL].find('img')
                    provider = provider_img['alt'].replace(' logo', '').strip() if provider_img and provider_img.has_attr('alt') else cols[PROVIDER_COL].get_text(strip=True)
                    model = cols[MODEL_COL].get_text(strip=True)
                    tokens_per_s_str = cols[TOKENS_PER_S_COL].get_text(strip=True)
                    response_time_str = cols[RESPONSE_TIME_COL].get_text(strip=True).lower().replace('s', '').strip()
                    intelligence_index_str = cols[INTELLIGENCE_INDEX_COL].get_text(strip=True)
                    context_window_str = cols[CONTEXT_WINDOW_COL].get_text(strip=True)

                    # Convert tokens per second
                    try: tokens_per_s = float(tokens_per_s_str) if tokens_per_s_str.lower() != 'n/a' else 0.0
                    except ValueError: tokens_per_s = 0.0
                    # Convert response time
                    try: response_time_s = float(response_time_str) if response_time_str.lower() != 'n/a' else float('inf')
                    except ValueError: response_time_s = float('inf')
                    # Convert intelligence index
                    try: intelligence_index = int(intelligence_index_str) if intelligence_index_str.isdigit() else 0 # Default to 0 if not a digit
                    except ValueError: intelligence_index = 0 # Default to 0 on conversion error
                    # Parse context window
                    context_window_int = parse_context_window(context_window_str)

                    if provider and model:
                        # Check if this model has a free API provider
                        is_free_source = False
                        for offering in KNOWN_FREE_MODEL_OFFERINGS:
                            if (offering["provider_key"].lower() == provider.lower() and 
                                (offering["model_key"].lower() == model.lower() or 
                                 normalize_model_name(offering["model_key"]) == normalize_model_name(model))):
                                is_free_source = True
                                break

                        performance_data.append({
                            'provider_name_scraped': provider,
                            'model_name_scraped': model,
                            'intelligence_index': intelligence_index if intelligence_index > 0 else 50,
                            'context_window': context_window_str,
                            'context_window_int': context_window_int if context_window_int is not None else 0,
                            'response_time_s': response_time_s,
                            'tokens_per_s': tokens_per_s,
                            'is_free_source': is_free_source
                        })
                except IndexError as e:
                     print(f"--- Warning: Skipping row {row_index} due to missing columns (IndexError): {row}. Error: {e} ---")
                except Exception as e:
                    print(f"--- Warning: Could not parse row {row_index} content: {row}. Error: {e} ---")
            else:
                 print(f"--- Warning: Skipping row {row_index} with insufficient columns ({len(cols)} found): {row} ---")
    except requests.exceptions.RequestException as e:
        print(f"--- Error fetching performance data URL: {e} ---")
        return []
    except Exception as e:
        print(f"--- Error processing performance data: {e} ---")
        return []

    if not performance_data:
         print("--- Warning: Scraping finished but no performance data was extracted. ---")
    else:
        print(f"--- Successfully extracted {len(performance_data)} model performance entries. ---")
        
    # Try to supplement context window data from the LLM-Performance-Leaderboard
    try:
        leaderboard_path = os.path.join(os.path.dirname(APP_DIR), "LLM-Performance-Leaderboard", "llm_leaderboard_20250521_013630.csv")
        if os.path.exists(leaderboard_path):
            print(f"--- Found LLM-Performance-Leaderboard CSV, supplementing context window data... ---")
            with open(leaderboard_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                leaderboard_data = list(reader)
                
            # Create a mapping of model names to context windows
            leaderboard_context_windows = {}
            for entry in leaderboard_data:
                model_name = entry.get('Model', '')
                context_window = entry.get('ContextWindow', '')
                if model_name and context_window:
                    leaderboard_context_windows[model_name.lower()] = context_window
            
            # Update performance data with context window information
            for entry in performance_data:
                model_name = entry['model_name_scraped']
                if entry['context_window_int'] == 0 and model_name.lower() in leaderboard_context_windows:
                    context_str = leaderboard_context_windows[model_name.lower()]
                    context_int = parse_context_window(context_str)
                    if context_int:
                        entry['context_window'] = context_str
                        entry['context_window_int'] = context_int
                        print(f"--- Updated context window for {model_name} to {context_str} ({context_int}) ---")
    except Exception as e:
        print(f"--- Error supplementing context window data: {e} ---")
    
    return performance_data
