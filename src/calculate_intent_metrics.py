"""
Script: calculate_intent_metrics.py
Purpose: Calculate intent metrics (precision, recall, f1, support, accuracy) from batch testing results.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_csv_data

# Define paths relative to the root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_RESULTS_DIR = ROOT_DIR / 'batch_testing_results' / 'processed'
RESULTS_DIR = ROOT_DIR / 'Results' / 'Intents'

def calculate_intent_metrics(results_file):
    """
    Calculate intent metrics from batch testing results.
    
    Args:
        results_file: Path to the batch testing results CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with intent metrics
    """
    # Load results data
    results_df = load_csv_data(results_file)
    
    # Normalize intent names to lowercase for consistent comparison
    results_df['ExpectedIntent_norm'] = results_df['ExpectedIntent'].str.lower() if 'ExpectedIntent' in results_df.columns else ''
    results_df['MatchedIntent_norm'] = results_df['MatchedIntent'].str.lower() if 'MatchedIntent' in results_df.columns else ''
    
    # Get unique intents from expected intents (these are the ones we want metrics for)
    # We'll keep the original case for display purposes
    intent_mapping = {}  # Maps normalized intent name to original case
    for intent in results_df['ExpectedIntent'].dropna().unique():
        if pd.notna(intent) and intent != '':
            intent_mapping[intent.lower()] = intent
    
    # Initialize metrics dictionary
    metrics = []
    
    # Calculate total number of samples
    total_samples = len(results_df)
    
    # Process each intent
    for norm_intent, original_intent in sorted(intent_mapping.items()):
        # Skip empty intents
        if pd.isna(norm_intent) or norm_intent == '':
            continue
        
        # Calculate metrics based on the ResultType field
        # TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative
        
        # Special handling for 'none' intent
        if norm_intent.lower() == 'none':
            # For 'none' intent:
            # TP: When expected is 'none' and result is TN (true negative)
            tp = len(results_df[(results_df['ExpectedIntent_norm'] == 'none') & 
                              (results_df['ResultType'] == 'TN')])
            
            # FP: When expected is not 'none' but matched is 'none' (or nothing matched)
            # This is when the system incorrectly thinks an utterance has no intent
            fp = len(results_df[(results_df['ExpectedIntent_norm'] != 'none') & 
                              (results_df['MatchedIntent_norm'] == 'none') & 
                              (results_df['ResultType'] == 'FN')])
            
            # FN: When expected is 'none' but something was matched (false positive in results)
            fn = len(results_df[(results_df['ExpectedIntent_norm'] == 'none') & 
                              (results_df['ResultType'] == 'FP')])
            
            # TN: When expected is not 'none' and matched is not 'none'
            tn = len(results_df[(results_df['ExpectedIntent_norm'] != 'none') & 
                              (results_df['MatchedIntent_norm'] != 'none') & 
                              (results_df['ResultType'] != 'FN')])
        else:
            # Regular intent handling
            # Get all rows where this is the expected intent
            expected_rows = results_df[results_df['ExpectedIntent_norm'] == norm_intent]
            
            # Count TP, FN for this intent
            tp = len(expected_rows[expected_rows['ResultType'] == 'TP'])
            fn = len(expected_rows[expected_rows['ResultType'] == 'FN'])
            
            # Count FP for this intent (where it was matched but not expected)
            fp = len(results_df[(results_df['MatchedIntent_norm'] == norm_intent) & 
                               (results_df['ExpectedIntent_norm'] != norm_intent) & 
                               (results_df['ResultType'] == 'FP')])
            
            # TN is all other samples
            tn = total_samples - (tp + fp + fn)
        
        # Calculate metrics
        support = tp + fn  # Total number of expected samples for this intent
        
        # Handle division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        
        # Add to metrics list
        metrics.append({
            'intent': original_intent,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': float(support),
            'accuracy': accuracy
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Calculate micro and macro averages
    if len(metrics) > 0:
        # Count total TP, FP, FN across all intents
        total_tp = sum(len(expected_rows[expected_rows['ResultType'] == 'TP']) 
                for norm_intent, original_intent in intent_mapping.items() 
                if pd.notna(norm_intent) and norm_intent != '' 
                for expected_rows in [results_df[results_df['ExpectedIntent_norm'] == norm_intent]])
    
        total_fp = sum(len(results_df[(results_df['MatchedIntent_norm'] == norm_intent) & 
                               (results_df['ExpectedIntent_norm'] != norm_intent) & 
                               (results_df['ResultType'] == 'FP')])
                for norm_intent, original_intent in intent_mapping.items() 
                if pd.notna(norm_intent) and norm_intent != '')
    
        total_fn = sum(len(expected_rows[expected_rows['ResultType'] == 'FN'])
                for norm_intent, original_intent in intent_mapping.items() 
                if pd.notna(norm_intent) and norm_intent != '' 
                for expected_rows in [results_df[results_df['ExpectedIntent_norm'] == norm_intent]])
    
        # Micro average calculations
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        micro_support = float(total_tp + total_fn)
        micro_accuracy = (total_tp + (total_samples - (total_tp + total_fp + total_fn))) / total_samples if total_samples > 0 else 0
        
        # Macro average (average of metrics)
        macro_precision = metrics_df['precision'].mean()
        macro_recall = metrics_df['recall'].mean()
        macro_f1 = metrics_df['f1'].mean()
        macro_support = float(metrics_df['support'].mean())
        macro_accuracy = metrics_df['accuracy'].mean()
    
        # Add micro and macro averages to the DataFrame
        metrics_df = pd.concat([metrics_df, pd.DataFrame([
            {
                'intent': 'micro',
                'precision': micro_precision,
                'recall': micro_recall,
                'f1': micro_f1,
                'support': micro_support,
                'accuracy': micro_accuracy
            },
            {
                'intent': 'macro',
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1,
                'support': macro_support,
                'accuracy': macro_accuracy
            }
        ])])
    
    return metrics_df

def process_all_batch_results():
    """
    Process all batch testing results files and generate intent metrics.
    """
    # Create output directory if it doesn't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each results file
    for results_file in BATCH_RESULTS_DIR.glob('*_results.csv'):
        # Get model name
        model_name = results_file.stem.split('_')[0]
        
        # Calculate metrics
        metrics_df = calculate_intent_metrics(results_file)
        
        # Save to CSV
        output_file = RESULTS_DIR / f'{model_name}_intent_metrics.csv'
        metrics_df.to_csv(output_file, index=False)
        print(f"Generated intent metrics for {model_name} at {output_file}")

def main():
    process_all_batch_results()

if __name__ == '__main__':
    main()
