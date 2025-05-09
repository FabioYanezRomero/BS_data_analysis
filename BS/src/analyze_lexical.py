"""
Script: analyze_lexical.py
Purpose: Perform lexical analysis (TTR intra-set and inter-set).
"""
import os
import pandas as pd
from pathlib import Path

def analyze_lexical(input_path, output_dir):
    # Placeholder: load and analyze lexical features
    df = pd.read_csv(input_path)
    # TODO: Add TTR intra-set and inter-set logic
    output_path = output_dir / (input_path.stem + '_lexical_analysis.csv')
    df.to_csv(output_path, index=False)  # Dummy output

def main():
    processed_dir = Path(__file__).parent.parent / 'batch_testing_results' / 'processed'
    results_dir = Path(__file__).parent.parent / 'Results' / 'data'
    results_dir.mkdir(exist_ok=True, parents=True)
    for file in processed_dir.glob('*results.csv'):
        analyze_lexical(file, results_dir)

if __name__ == '__main__':
    main()
