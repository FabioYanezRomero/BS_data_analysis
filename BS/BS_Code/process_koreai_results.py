import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BS_DATA = Path(__file__).parent.parent / 'BS_data'
BS_DATAFRAMES = Path(__file__).parent.parent / 'BS_dataframes'
BS_DATAFRAMES.mkdir(exist_ok=True)


def parse_metadata_and_data(filepath: Path) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Parses the metadata at the top of the CSV and loads the main data table.
    Handles multiline and quoted values (especially for 'Notes').
    Returns a tuple (metadata_dict, data_df).
    """
    metadata = {}
    header_line = None
    lines = []
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('"Utterance"'):
            header_line = i
            break
        if ',' in line:
            key, value = line.split(',', 1)
            key = key.strip().strip('"')
            value = value.strip()
            # Handle multiline Notes field
            if key == 'Notes:':
                value_lines = [value.strip('"\n')]
                i += 1
                while i < len(lines) and not lines[i].startswith('"'):
                    value_lines.append(lines[i].strip('"\n'))
                    i += 1
                metadata[key] = ' '.join([v for v in value_lines if v]).strip()
                continue  # already incremented i
            else:
                value = value.strip('"\n')
                metadata[key] = value
        i += 1
    if header_line is None:
        raise ValueError(f"No data header found in {filepath}")
    # Read the data section
    data_str = ''.join(lines[header_line:])
    from io import StringIO
    df = pd.read_csv(StringIO(data_str), na_filter=False)
    return metadata, df


def compute_intent_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute micro and per-intent metrics for each ExpectedIntent label.
    Support = count of rows with that expected label.
    """
    # Compute per-utterance metrics to properly count each example once
    ut = df.dropna(subset=['Utterance','ExpectedIntent','MatchedIntent'])
    ut = ut.drop_duplicates(subset='Utterance')
    exp = ut['ExpectedIntent'].astype(str).str.strip().str.lower().replace('', 'none')
    mt  = ut['MatchedIntent'].astype(str).str.strip().str.lower().replace('', 'none')
    N = len(ut)
    labels = sorted(set(exp))
    metrics_list = []
    for label in labels:
        tp = int(((exp == label) & (mt == label)).sum())
        fp = int(((exp != label) & (mt == label)).sum())
        fn = int(((exp == label) & (mt != label)).sum())
        tn = N - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0
        recall    = tp / (tp + fn) if tp + fn else 0
        f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0
        support   = int((exp == label).sum())
        accuracy  = (tp + tn) / N if N else 0
        metrics_list.append({
            'intent': label,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'accuracy': accuracy
        })
    # Micro metrics
    micro_tp = int((exp == mt).sum())
    micro_fp = int((exp != mt).sum())
    micro_fn = micro_fp
    micro_support = N
    micro_prec = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0
    micro_rec  = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if micro_prec + micro_rec else 0
    metrics_list.append({
        'intent': 'micro',
        'precision': micro_prec,
        'recall': micro_rec,
        'f1': micro_f1,
        'support': micro_support,
        'accuracy': micro_tp / N if N else 0
    })
    # Macro metrics: average of per-intent (exclude micro)
    class_metrics = [m for m in metrics_list if m['intent'] != 'micro']
    count = len(class_metrics)
    macro_prec = sum(m['precision'] for m in class_metrics) / count
    macro_rec = sum(m['recall']    for m in class_metrics) / count
    macro_f1 = sum(m['f1']        for m in class_metrics) / count
    macro_support = sum(m['support'] for m in class_metrics) / count
    macro_acc = sum(m['accuracy']   for m in class_metrics) / count
    metrics_list.append({
        'intent': 'macro',
        'precision': macro_prec,
        'recall': macro_rec,
        'f1': macro_f1,
        'support': macro_support,
        'accuracy': macro_acc
    })
    return pd.DataFrame(metrics_list)


def compute_entity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """

    Evalua la accuracy por tipo de entidad. En este caso se consideran las columnas:
    
    Entity Name
    Expected EntityValue
    Matched EntityValue
    
    La primera columna nos indica el nombre de la entidad general, las otras dos nos indican si coinciden o no
    obteniendo la métrica al dividir el número de aciertos por el número total de apariciones de la entidad.
    
    """
    # Filter rows where expected and matched entities are available
    entity_rows = df.dropna(subset=['Entity Name', 'Expected EntityValue', 'Matched EntityValue'])
    entities = sorted(set(entity_rows['Entity Name']))
    metrics_list = []
    for entity in entities:
        entity = entity.lower()
        accuracy = ((entity_rows['Expected EntityValue'] == entity_rows['Matched EntityValue']) & (entity_rows['Entity Name'].str.lower() == entity)).sum() / len(entity_rows[entity_rows['Entity Name'].str.lower() == entity])
        support = len(entity_rows[entity_rows['Entity Name'].str.lower() == entity])
        metrics_list.append({
            'entity': entity,
            'accuracy': accuracy,
            'support': support
        })
    # Micro metrics (global)
    total_correct = (entity_rows['Expected EntityValue'] == entity_rows['Matched EntityValue']).sum()
    total_count = len(entity_rows)
    micro_acc = total_correct / total_count if total_count > 0 else 0
    micro_support = total_count
    metrics_list.append({
        'entity': 'micro',
        'accuracy': micro_acc,
        'support': micro_support
    })
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


def compute_koreai_entity_success(df: pd.DataFrame) -> pd.DataFrame:
    # Utterance-level entity success: require every Entity Result in each utterance to be TRUE
    filtered = df[df['Entity Result'].astype(str).str.upper().isin(['TRUE','FALSE'])]
    utterances = filtered['Utterance'].unique()
    correct = 0
    for utt in utterances:
        df_utt = filtered[filtered['Utterance'] == utt]
        if df_utt['Entity Result'].astype(str).str.upper().eq('TRUE').all():
            correct += 1
    value = correct / len(utterances) if len(utterances) > 0 else 0.0
    return pd.DataFrame([{'metric': 'utterance_level_success', 'value': value}])


def process_file(filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Process a single CSV file and return:
    - intent_metrics_df
    - entity_metrics_df
    - raw data df
    - metadata dict
    """
    metadata, df = parse_metadata_and_data(filepath)
    intent_metrics_df = compute_intent_metrics(df)
    entity_metrics_df = compute_entity_metrics(df)
    return intent_metrics_df, entity_metrics_df, df, metadata


import re

def clean_notes_for_filename(notes: str) -> str:
    """Clean the Notes field to create a safe filename."""
    if not notes:
        return None
    # Remove newlines and non-filename-safe chars
    cleaned = re.sub(r'[^\w\-]+', '_', notes.strip())
    return cleaned[:50]  # Limit length for safety

def save_combined_entity_metrics(filename_prefix: str, entity_metrics: pd.DataFrame, utterance_metrics: pd.DataFrame, metadata: Dict[str, str]):
    """
    Save all entity-related metrics (per-entity, micro/global, utterance-level) to a single CSV.
    """
    # Try both possible keys for Notes
    notes = metadata.get('Notes')
    if not notes:
        notes = metadata.get('Notes:')
    if notes:
        notes = notes.strip().replace('\n', ' ')
    cleaned_notes = clean_notes_for_filename(notes) if notes else None
    base_name = cleaned_notes if cleaned_notes else filename_prefix

    # Add a column to distinguish metric type
    entity_metrics_cp = entity_metrics.copy()
    entity_metrics_cp['metric_type'] = 'per_entity_or_micro'
    # For utterance-level, add compatible columns
    utterance_metrics_cp = utterance_metrics.copy()
    utterance_metrics_cp['entity'] = None
    utterance_metrics_cp['accuracy'] = utterance_metrics_cp['value']
    utterance_metrics_cp['support'] = None
    utterance_metrics_cp['metric_type'] = utterance_metrics_cp['metric']
    # Select columns to match entity_metrics
    utterance_metrics_cp = utterance_metrics_cp[['entity', 'accuracy', 'support', 'metric_type']]
    entity_metrics_cp = entity_metrics_cp[['entity', 'accuracy', 'support', 'metric_type']]
    # Combine
    combined = pd.concat([entity_metrics_cp, utterance_metrics_cp], ignore_index=True)
    combined.to_csv(BS_DATAFRAMES / f'{base_name}_all_entity_metrics.csv', index=False)
    # Also save metadata for traceability
    pd.DataFrame([metadata]).to_csv(BS_DATAFRAMES / f'{base_name}_metadata.csv', index=False)


def main():
    csv_files = list(BS_DATA.glob('*.csv'))
    if not csv_files:
        logging.warning(f"No CSV files found in {BS_DATA}")
        return
    for csv_file in csv_files:
        logging.info(f"Processing {csv_file.name}")
        try:
            intent_metrics, entity_metrics, raw_df, metadata = process_file(csv_file)
            filename_prefix = csv_file.stem
            # Compute utterance-level entity success
            utterance_metrics = compute_koreai_entity_success(raw_df)
            # Save all entity-related metrics in a single CSV
            save_combined_entity_metrics(filename_prefix, entity_metrics, utterance_metrics, metadata)
            # Optionally, still save intent metrics and raw for traceability
            intent_metrics.to_csv(BS_DATAFRAMES / f'{filename_prefix}_intent_metrics.csv', index=False)
            raw_df.to_csv(BS_DATAFRAMES / f'{filename_prefix}_raw.csv', index=False)
            logging.info(f"Saved combined entity metrics for {csv_file.name}")
        except Exception as e:
            logging.error(f"Error processing {csv_file.name}: {e}")

if __name__ == '__main__':
    main()
