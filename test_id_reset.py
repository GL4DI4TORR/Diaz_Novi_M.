import requests
import json

# Test clearing and reloading records
base_url = "http://127.0.0.1:8001/api/records/"

# 1. Check current records
print("=== Current Records ===")
response = requests.get(f"{base_url}")
if response.status_code == 200:
    data = response.json()
    print(f"Total records: {data['total_records']}")
    if data['total_records'] > 0:
        # Get first and last record IDs
        records = requests.get(f"{base_url}?page=1").json()['results']
        print(f"First record ID: {records[0]['id']}")
        
        # Get last page
        last_page = data['pages']
        last_records = requests.get(f"{base_url}?page={last_page}").json()['results']
        print(f"Last record ID: {last_records[-1]['id']}")

# 2. Clear records
print("\n=== Clearing Records ===")
response = requests.post(f"{base_url}clear_records/")
if response.status_code == 200:
    result = response.json()
    print(f"Message: {result['message']}")
else:
    print(f"Error: {response.status_code} - {response.text}")

# 3. Load dataset
print("\n=== Loading Dataset ===")
response = requests.post(f"{base_url}load_dataset/")
if response.status_code == 200:
    result = response.json()
    print(f"Message: {result['message']}")
    print(f"Records loaded: {result['records_loaded']}")
    
    # Check new IDs
    print("\n=== New Records After Reload ===")
    response = requests.get(f"{base_url}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total records: {data['total_records']}")
        
        # Get first and last record IDs
        records = requests.get(f"{base_url}?page=1").json()['results']
        print(f"First record ID: {records[0]['id']}")
        
        # Get last page
        last_page = data['pages']
        last_records = requests.get(f"{base_url}?page={last_page}").json()['results']
        print(f"Last record ID: {last_records[-1]['id']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
