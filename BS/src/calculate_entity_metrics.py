"""
Script: calculate_entity_metrics.py
Purpose: Calculate entity metrics from batch testing results.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_csv_data

def calculate_entity_metrics(results_file):
    # Initialize DataFrames that will be returned
    entity_metrics_df = None
    entity_identification_df = None
    entity_method_df = None
    entity_engine_summary_df = None
    entity_success_df = None
    """
    Calculate entity metrics from batch testing results.
    
    Args:
        results_file: Path to the batch testing results CSV file
        
    Returns:
        tuple: (entity_metrics_df, entity_identification_df, entity_method_df)
    """
    # Extract model name from file path
    model_name = Path(results_file).stem.split('_')[0]
    
    # Load results data
    results_df = load_csv_data(results_file)
    
    # Filter rows with entity information
    entity_rows = results_df[results_df['Entity Name'].notna() & (results_df['Entity Name'] != '')]
    
    # Get unique entity types
    entity_types = entity_rows['Entity Name'].unique()
    
    # Calculate metrics for each entity type
    entity_metrics = []
    total_entities = len(entity_rows)
    
    # Calculate overall metrics
    for entity_type in sorted(entity_types):
        # Filter rows for this entity type
        entity_type_rows = entity_rows[entity_rows['Entity Name'] == entity_type]
        
        # Calculate support (total occurrences of this entity type)
        support = len(entity_type_rows)
        
        # Calculate accuracy (True/False in Entity Result column)
        true_matches = entity_type_rows[entity_type_rows['Entity Result'] == True].shape[0]
        accuracy = true_matches / support if support > 0 else 0
        
        # Add to metrics list
        entity_metrics.append({
            'entity': entity_type,
            'accuracy': accuracy,
            'support': support
        })
    
    # Add a row for empty entity (utterances without entities)
    non_entity_rows = results_df[results_df['Entity Name'].isna() | (results_df['Entity Name'] == '')]
    entity_metrics.append({
        'entity': '',
        'accuracy': 1.0,  # By default, these are correct (no entity expected, none found)
        'support': len(non_entity_rows)
    })
    
    # Calculate micro average
    total_true_matches = entity_rows[entity_rows['Entity Result'] == True].shape[0]
    micro_accuracy = total_true_matches / total_entities if total_entities > 0 else 0
    
    # Add micro average to metrics
    entity_metrics.append({
        'entity': 'micro',
        'accuracy': micro_accuracy,
        'support': len(results_df)  # Total number of utterances
    })
    
    # Convert to DataFrame
    entity_metrics_df = pd.DataFrame(entity_metrics)
    
    # Calculate entity identification methods with proper relationship between EntityIdentifiedBy and entityIdentifiedUsing
    # Now calculating percentages per entity type
    
    # First create a simplified version of entityIdentifiedUsing
    entity_rows['simplified_using'] = entity_rows['entityIdentifiedUsing'].apply(
        lambda x: 'Full Match' if isinstance(x, str) and x.startswith('Full Match') else
                  'Partial Match' if isinstance(x, str) and x.startswith('Partial Match') else
                  x
    )
    
    # Create a DataFrame to store entity identification data per entity type
    entity_identification_df = pd.DataFrame()
    
    # Process each entity type separately
    for entity_type in sorted(entity_types):
        # Filter rows for this entity type
        entity_type_rows = entity_rows[entity_rows['Entity Name'] == entity_type]
        entity_type_count = len(entity_type_rows)
        
        if entity_type_count == 0:
            continue
        
        # Calculate total counts for each EntityIdentifiedBy method for this entity type
        identified_by_counts = entity_type_rows['EntityIdentifiedBy'].value_counts().reset_index()
        identified_by_counts.columns = ['method', 'count']
        
        # Calculate the total count of all methods for this entity type
        total_methods_count = identified_by_counts['count'].sum()
        
        # Calculate percentage based on the total methods count, not entity count
        identified_by_counts['percentage'] = identified_by_counts['count'] / total_methods_count * 100
        identified_by_counts['category'] = 'EntityIdentifiedBy'
        identified_by_counts['entity'] = entity_type
        
        # Add rows for each EntityIdentifiedBy method
        for _, row in identified_by_counts.iterrows():
            entity_identification_df = pd.concat([entity_identification_df, pd.DataFrame([{
                'entity': entity_type,
                'method': row['method'],
                'count': row['count'],
                'percentage': row['percentage'],
                'category': 'EntityIdentifiedBy',
                'sub_method': 'Total'
            }])])
            
            # Group by simplified entityIdentifiedUsing for this entity type and EntityIdentifiedBy
            sub_rows = entity_type_rows[entity_type_rows['EntityIdentifiedBy'] == row['method']]
            sub_methods = sub_rows['simplified_using'].value_counts().reset_index()
            sub_methods.columns = ['sub_method', 'sub_count']
            
            # Calculate the total count for this specific method
            method_total_count = sub_methods['sub_count'].sum()
            
            # Add rows for each entityIdentifiedUsing method under this EntityIdentifiedBy
            for _, sub_row in sub_methods.iterrows():
                entity_identification_df = pd.concat([entity_identification_df, pd.DataFrame([{
                    'entity': entity_type,
                    'method': row['method'],
                    'count': sub_row['sub_count'],
                    'percentage': sub_row['sub_count'] / method_total_count * 100,
                    'category': 'entityIdentifiedUsing',
                    'sub_method': sub_row['sub_method']
                }])])
    
    # Calculate identification methods per entity type with proper relationship
    entity_method_data = []
    entity_engine_summary_data = []
    
    for entity_type in sorted(entity_types):
        # Filter rows for this entity type
        entity_type_rows = entity_rows[entity_rows['Entity Name'] == entity_type]
        entity_type_count = len(entity_type_rows)
        
        # Count by EntityIdentifiedBy
        identified_by = entity_type_rows['EntityIdentifiedBy'].value_counts().reset_index()
        identified_by.columns = ['method', 'count']
        
        # Calculate the total count of all engines for this entity type
        total_engines_count = identified_by['count'].sum()
        
        # Store engine summary in a separate list for a different CSV file
        for _, row in identified_by.iterrows():
            if pd.notna(row['method']) and row['method'] != '':
                entity_engine_summary_data.append({
                    'entity': entity_type,
                    'engine': row['method'],
                    'count': row['count'],
                    'percentage': row['count'] / total_engines_count * 100 if total_engines_count > 0 else 0
                })
                
                # For each EntityIdentifiedBy, get the breakdown by entityIdentifiedUsing
                sub_rows = entity_type_rows[entity_type_rows['EntityIdentifiedBy'] == row['method']]
                
                # Process entityIdentifiedUsing to cluster Full Match and Partial Match
                # Create a new column with simplified method names
                sub_rows['simplified_method'] = sub_rows['entityIdentifiedUsing'].apply(
                    lambda x: 'Full Match' if isinstance(x, str) and x.startswith('Full Match') else
                              'Partial Match' if isinstance(x, str) and x.startswith('Partial Match') else
                              x
                )
                
                # Count by simplified method
                sub_methods = sub_rows['simplified_method'].value_counts().reset_index()
                sub_methods.columns = ['sub_method', 'sub_count']
                
                # Calculate the total count for this specific engine
                engine_total_count = sub_methods['sub_count'].sum()
                
                for _, sub_row in sub_methods.iterrows():
                    if pd.notna(sub_row['sub_method']) and sub_row['sub_method'] != '':
                        entity_method_data.append({
                            'entity': entity_type,
                            'identified_by': row['method'],
                            'identified_using': sub_row['sub_method'],
                            'count': sub_row['sub_count'],
                            'percentage': sub_row['sub_count'] / engine_total_count * 100 if engine_total_count > 0 else 0
                        })
    
    # Convert to DataFrame
    entity_method_df = pd.DataFrame(entity_method_data)
    entity_engine_summary_df = pd.DataFrame(entity_engine_summary_data)
    
    # Rename columns to make the output more clear
    entity_method_df = entity_method_df.rename(columns={
        'identified_by': 'engine',
        'identified_using': 'method'
    })
    
    # Calculate utterance-level entity success (KoreAI metric)
    # An utterance is successful if all its entities are correctly identified
    utterance_success_count = 0
    utterances_with_entities = 0
    
    # Group by utterance and check if all entities are correctly identified
    for utterance, group in entity_rows.groupby('Utterance'):
        utterances_with_entities += 1
        if all(group['Entity Result'] == True):
            utterance_success_count += 1
    
    # Calculate success ratio
    utterance_success_ratio = utterance_success_count / utterances_with_entities if utterances_with_entities > 0 else 0
    
    # Create KoreAI success metric DataFrame
    entity_success_df = pd.DataFrame([
        {'metric': 'utterance_level_success', 'value': utterance_success_ratio}
    ])
    
    return entity_metrics_df, entity_identification_df, entity_method_df, entity_engine_summary_df, entity_success_df

def process_all_batch_results():
    """
    Process all batch testing results files and generate entity metrics.
    """
    # Define paths
    base_dir = Path(__file__).parent.parent
    batch_results_dir = base_dir / 'batch_testing_results' / 'processed'
    entity_metrics_dir = base_dir / 'Results' / 'Entities'
    
    # Create output directory if it doesn't exist
    entity_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each results file
    for results_file in batch_results_dir.glob('*_results.csv'):
        # Get model name
        model_name = results_file.stem.split('_')[0]

        # Calculate metrics
        entity_metrics_df, entity_identification_df, entity_method_df, entity_engine_summary_df, entity_success_df = calculate_entity_metrics(results_file)

        # Save to CSV files
        entity_metrics_df.to_csv(entity_metrics_dir / f'{model_name}_entity_metrics.csv', index=False)

        # For entity identification file, group by entity type for better readability
        entity_identification_df.to_csv(entity_metrics_dir / f'{model_name}_entity_identification.csv', index=False)

        
        # Save entity methods without the Total rows for better readability
        entity_method_df.to_csv(entity_metrics_dir / f'{model_name}_entity_methods.csv', index=False)
        
        # Save engine summary to a separate file
        entity_engine_summary_df.to_csv(entity_metrics_dir / f'{model_name}_entity_engine_summary.csv', index=False)
        
        entity_success_df.to_csv(entity_metrics_dir / f'{model_name}_entity_success_koreai.csv', index=False)
        
        print(f"Generated entity metrics for {model_name} at {entity_metrics_dir}")

def main():
    process_all_batch_results()

if __name__ == '__main__':
    main()
