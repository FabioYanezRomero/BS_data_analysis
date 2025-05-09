"""
Script: process_batch_raw.py
Purpose: Process each file in batch_testing_results/raw and generate results and metadata files in batch_testing_results/processed.
"""
import os
import pandas as pd
from pathlib import Path

def process_raw_file(input_path, output_dir):
    # Read all lines
    with open(input_path, encoding='utf-8') as f:
        lines = f.readlines()
    # Find header line
    header_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Utterance"):
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
    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)

def main():
    raw_dir = Path(__file__).parent.parent / 'batch_testing_results' / 'raw'
    processed_dir = Path(__file__).parent.parent / 'batch_testing_results' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    for file in raw_dir.glob('*.csv'):
        process_raw_file(file, processed_dir)

if __name__ == '__main__':
    main()
