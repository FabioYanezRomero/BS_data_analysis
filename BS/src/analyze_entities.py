"""
Script: analyze_entities.py
Purpose: Process entities and compute metrics (entity accuracy, utterance-level success, etc.).
"""
import os
import pandas as pd
from pathlib import Path

def analyze_entities(input_path, output_dir):
    # Placeholder: load and analyze entities
    df = pd.read_csv(input_path)
    # TODO: Add entity metric logic (accuracy, micro, utterance-level, etc.)
    output_path = output_dir / (input_path.stem + '_entity_metrics.csv')
    df.to_csv(output_path, index=False)  # Dummy output

def main():
    processed_dir = Path(__file__).parent.parent / 'batch_testing_results' / 'processed'
    results_dir = Path(__file__).parent.parent / 'Results' / 'entities'
    results_dir.mkdir(exist_ok=True, parents=True)
    for file in processed_dir.glob('*results.csv'):
        analyze_entities(file, results_dir)

if __name__ == '__main__':
    main()
