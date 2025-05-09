"""
Script: analyze_intents.py
Purpose: Analyze intent-related results (semantic similarity, intra-intent similarity).
"""
import os
import pandas as pd
from pathlib import Path

def analyze_intent_results(input_path, output_dir):
    # Placeholder: load and analyze intent results
    df = pd.read_csv(input_path)
    # TODO: Add intent similarity and intra-intent logic
    output_path = output_dir / (input_path.stem + '_intent_analysis.csv')
    df.to_csv(output_path, index=False)  # Dummy output

def main():
    processed_dir = Path(__file__).parent.parent / 'batch_testing_results' / 'processed'
    results_dir = Path(__file__).parent.parent / 'Results' / 'intents'
    results_dir.mkdir(exist_ok=True, parents=True)
    for file in processed_dir.glob('*results.csv'):
        analyze_intent_results(file, results_dir)

if __name__ == '__main__':
    main()
