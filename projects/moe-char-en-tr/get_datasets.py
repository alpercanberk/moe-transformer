import requests
import json
from pathlib import Path

# Constants
BASE_URL = "https://datasets-server.huggingface.co/rows?dataset=opus100&config=en-tr&split=test"
LIMIT = 100
TARGET_ROWS = 10000
DIRECTORY = Path("en-tr")
ENGLISH_FILENAME = DIRECTORY / "english-train.txt"
TURKISH_FILENAME = DIRECTORY / "turkish-train.txt"

def get_data(offset):
    """Fetch data from the server with the given offset."""
    response = requests.get(f"{BASE_URL}&offset={offset}&limit={LIMIT}")
    
    if response.status_code == 200:
        return response.json()["rows"]
    else:
        print(f"Error {response.status_code} when fetching data with offset {offset}. Response: {response.text}")
        return []

def main():
    # Ensure directory exists
    DIRECTORY.mkdir(exist_ok=True)

    # Opening files for writing
    with open(ENGLISH_FILENAME, 'w', encoding='utf-8') as ef, open(TURKISH_FILENAME, 'w', encoding='utf-8') as tf:
        for offset in range(0, TARGET_ROWS, LIMIT):
            print(f"Fetching data with offset {offset}...")
            rows = get_data(offset)
            
            # Checking if any rows were fetched
            if not rows:
                print(f"No rows fetched for offset {offset}. Continuing to the next offset...")
                continue
            
            # Extracting translations and writing to the files
            for row in rows:
                english_sentence = row["row"]["translation"]["en"]
                turkish_sentence = row["row"]["translation"]["tr"]
                
                ef.write(english_sentence + "\n")
                tf.write(turkish_sentence + "\n")
            
            print(f"Written {LIMIT if offset + LIMIT <= TARGET_ROWS else TARGET_ROWS % LIMIT} rows for offset {offset}")

if __name__ == "__main__":
    main()
