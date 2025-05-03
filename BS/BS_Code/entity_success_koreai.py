import pandas as pd
import sys
import os
from process_koreai_results import parse_metadata_and_data
import re

def compute_koreai_entity_success(df: pd.DataFrame) -> pd.DataFrame:
    # Filter valid matched entities (not null, not None, not empty string)
    valid_entities = df['Matched EntityValue'].dropna()
    valid_entities = valid_entities[valid_entities.astype(str).str.lower().ne('none')]
    valid_entities = valid_entities[valid_entities.astype(str).str.strip() != '']
    recognized_entities = valid_entities    #.unique()
    recognized_count = len(recognized_entities)

    # Filter rows where Entity Result is exactly 'TRUE' or 'FALSE' (case-insensitive, ignore nulls and other values)
    filtered_df = df[df['Entity Result'].astype(str).str.upper().isin(['TRUE', 'FALSE'])]

    # For those, count how many have Entity Result == TRUE (case-insensitive)
    matched_entities = 0
    for entity in recognized_entities:
        entity_rows = filtered_df[filtered_df['Matched EntityValue'] == entity]
        # If any row for this entity has Entity Result == TRUE, count as matched
        if (entity_rows['Entity Result'].astype(str).str.upper() == 'TRUE').any():
            matched_entities += 1

    # Calculate utterance-level entity success rate
    utterances = filtered_df['Utterance'].unique()
    correct_utterances = 0
    for utterance in utterances:
        utterance_df = filtered_df[filtered_df['Utterance'] == utterance]
        utterance_entities = utterance_df['Matched EntityValue'].unique()
        if all(utterance_df[utterance_df['Matched EntityValue'] == entity]['Entity Result'].astype(str).str.upper().eq('TRUE').any() for entity in utterance_entities):
            correct_utterances += 1

    result = pd.DataFrame([{
        'metric': 'entity_level_success',
        'value': matched_entities / recognized_count if recognized_count > 0 else 0.0
    }, {
        'metric': 'utterance_level_success',
        'value': correct_utterances / len(utterances) if len(utterances) > 0 else 0.0
    }])
    return result

if __name__ == '__main__':
    csvs = os.listdir("/usrvol/BS/BS_data")
    for input_csv in csvs:
        csv_path = os.path.join("/usrvol/BS/BS_data", input_csv)
        metadata, df = parse_metadata_and_data(csv_path)
        if df is not None:
            result = compute_koreai_entity_success(df)
            notes = metadata.get('Notes:', '')
            safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', notes)
            output_csv = f"{safe_filename}_entity_success_koreai.csv"
            output_path = os.path.join("/usrvol/BS/BS_dataframes", output_csv)
            result.to_csv(output_path, index=False)
