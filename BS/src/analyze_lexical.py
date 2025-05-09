"""
Script: analyze_lexical.py
Purpose: Perform lexical analysis (TTR intra-set and inter-set).
"""
import itertools
import pandas as pd
from pathlib import Path

def analyze_lexical(processed_dir, results_dir):
    """
    Compute TTR intra-set for each results file and inter-set TTR differences.
    """
    processed_files = list(Path(processed_dir).glob('*results.csv'))
    # Collect vocab and TTR per file
    stats = {}
    for f in processed_files:
        df = pd.read_csv(f)
        # tokenise utterances
        tokens = df['Utterance'].astype(str).str.split().explode().tolist()
        vocab = set(tokens)
        vocab_size = len(vocab)
        ttr = vocab_size / len(tokens) if tokens else 0.0
        stats[f.stem] = {'vocab_size': vocab_size, 'type_token_ratio': ttr}
        # save intra-set metrics
        intra_df = pd.DataFrame([{
            'file': f.name,
            'vocab_size': vocab_size,
            'type_token_ratio': ttr
        }])
        intra_df.to_csv(Path(results_dir) / (f.stem + '_lexical_analysis.csv'), index=False)
    # inter-set: pairwise TTR differences
    if len(stats) >= 2:
        rows = []
        for a, b in itertools.combinations(stats.keys(), 2):
            t1 = stats[a]['type_token_ratio']
            t2 = stats[b]['type_token_ratio']
            rows.append({
                'file1': a + '_results.csv',
                'file2': b + '_results.csv',
                'type_token_ratio_diff': abs(t1 - t2)
            })
        pd.DataFrame(rows).to_csv(Path(results_dir) / 'ttr_inter_set_diff.csv', index=False)

def main():
    processed_dir = Path(__file__).parent.parent / 'batch_testing_results' / 'processed'
    results_dir = Path(__file__).parent.parent / 'Results' / 'data'
    results_dir.mkdir(exist_ok=True, parents=True)
    analyze_lexical(processed_dir, results_dir)

if __name__ == '__main__':
    main()
