import csv

def check_model():
    found = False
    with open('C:/Users/owens/Coding Projects/flask_app/provider_performance.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Qwen3 32B (Reasoning)' in row.get('model_name_scraped', ''):
                print(f\
Found
model:
row['model_name_scraped']
from
row['provider_name_scraped']
with
intel_index=
row['intelligence_index']
and
response_time=
row['response_time_s']
\)
                found = True
    
    if not found:
        print('Model not found')

if __name__ == \__main__\:
    check_model()
