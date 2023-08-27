import json
from pathlib import Path

# Constants
DIRECTORY = Path("opus-100-corpus/v1.0/supervised/en-tr")
ENGLISH_FILE = DIRECTORY / "opus.en-tr-train.en"
TURKISH_FILE = DIRECTORY / "opus.en-tr-train.tr"
ENCODER_FILENAME = DIRECTORY / "encoder.json"

def main():
    unique_chars = set()  # Initialize a set to keep track of unique characters

    # Process the English file
    with open(ENGLISH_FILE, 'r', encoding='utf-8') as ef:
        for index, line in enumerate(ef):
            unique_chars = unique_chars.union(set(line))
            if index % 5000 == 0:  # Print progress every 5000 lines
                print(f"Processed {index} lines from the English file...")
    
    # Process the Turkish file
    with open(TURKISH_FILE, 'r', encoding='utf-8') as tf:
        for index, line in enumerate(tf):
            unique_chars = unique_chars.union(set(line))
            if index % 5000 == 0:  # Print progress every 5000 lines
                print(f"Processed {index} lines from the Turkish file...")

    # Convert the set to a dictionary and then save to a JSON file
    encoder = {char: index for index, char in enumerate(sorted(unique_chars))}
    with open(ENCODER_FILENAME, 'w', encoding='utf-8') as enc_file:
        json.dump(encoder, enc_file, ensure_ascii=False, indent=4)

    print("Finished creating encoder.json!")

if __name__ == "__main__":
    main()
