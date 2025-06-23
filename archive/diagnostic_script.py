import os
import pandas as pd
import logging

IMAGE_DIR = 'dashboard-light_scraper/data/images'
OUTPUT_CSV = 'dashboard-light_scraper/chart_data_filtered.csv'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

image_models = set()
logger.info("--- Analyzing image directory ---")

if not os.path.exists(IMAGE_DIR):
    logger.error(f"Image directory not found: {IMAGE_DIR}")
    exit()

for make_dir in os.listdir(IMAGE_DIR):
    make_path = os.path.join(IMAGE_DIR, make_dir)
    if os.path.isdir(make_path):
        for model_dir in os.listdir(make_path):
            image_models.add((make_dir.strip(), model_dir.strip())) 

logger.info(f"Total unique Make/Model combinations in image directory: {len(image_models)}")

csv_models_present = set()
logger.info("--- Analyzing output CSV ---")
if not os.path.exists(OUTPUT_CSV):
    logger.warning(f"Output CSV not found: {OUTPUT_CSV}")
else:
    try:
        df = pd.read_csv(OUTPUT_CSV, usecols=['Make', 'Model'], dtype=str, skipinitialspace=True)
        df.dropna(subset=['Make', 'Model'], inplace=True)
        for index, row in df.iterrows():
            make_csv = str(row['Make']).strip()
            model_csv = str(row['Model']).strip()
            if make_csv and model_csv:
                 csv_models_present.add((make_csv, model_csv))
    except Exception as e:
        logger.error(f"Error reading CSV {OUTPUT_CSV}: {e}")

logger.info(f"Total unique Make/Model combinations found at least once in CSV: {len(csv_models_present)}")

# Now find models from image dir that are NOT in the set of models present in CSV
models_completely_missing_from_csv = sorted(list(image_models - csv_models_present))

logger.info("--- Make/Model combinations in image directory but completely missing from CSV ---")
if models_completely_missing_from_csv:
    for make, model in models_completely_missing_from_csv:
        logger.info(f"{make}\\{model}")
else:
    logger.info("All Make/Model combinations found in the image directory have at least one corresponding row in the CSV.")

logger.info(f"Count of Make/Model combinations completely missing from CSV: {len(models_completely_missing_from_csv)}")
print(f"\nFound {len(models_completely_missing_from_csv)} Make/Model combinations in the image directory that appear to have NO corresponding entries in {OUTPUT_CSV}.")
if models_completely_missing_from_csv:
    print("List of these Make\\Model combinations:")
    for make, model in models_completely_missing_from_csv:
        print(f"  {make}\\{model}") 