"""
Script: process_batch_raw.py
Purpose: Process each file in batch_testing_results/raw and generate results and metadata files in batch_testing_results/processed.
"""
import os
import pandas as pd
from pathlib import Path

# Define paths relative to the root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = ROOT_DIR / 'batch_testing_results' / 'raw'
PROCESSED_DIR = ROOT_DIR / 'batch_testing_results' / 'processed'

def process_raw_file(input_path, output_dir):
    # Read all lines
    with open(input_path, encoding='utf-8') as f:
        lines = f.readlines()
    # Find header line by first cell matching Utterance
    header_line = None
    for i, line in enumerate(lines):
        first_cell = line.split(',')[0].strip().strip('"')
        if first_cell == "Utterance":
            header_line = i
            break
    if header_line is None:
        raise ValueError("No data header found in file: " + str(input_path))
    # Parse metadata
    metadata = {}
    for line in lines[:header_line]:
        if ',' in line:
            key, value = line.split(',', 1)
            metadata[key.strip().strip('"')] = value.strip().strip('"\n')
    # Parse data table
    from io import StringIO
    data_str = ''.join(lines[header_line:])
    df = pd.read_csv(StringIO(data_str))
    # Save results and metadata
    results_path = output_dir / (input_path.stem + '_results.csv')
    metadata_path = output_dir / (input_path.stem + '_metadata.csv')
    df.to_csv(results_path, index=False)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[:header_line])

def process_all_raw_files():
    """Process all raw batch testing files."""
    # Create processed directory if it doesn't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for file in RAW_DIR.glob('*.csv'):
        process_raw_file(file, PROCESSED_DIR)

def main():
    process_all_raw_files()

if __name__ == '__main__':
    main()
