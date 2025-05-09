"""
Script: analyze_lexical.py
Purpose: Perform lexical analysis (TTR intra-set and inter-set).
"""
import itertools
import pandas as pd
from pathlib import Path

def analyze_lexical(test_suites_dir, utterances_dir, results_dir):
    """
    Compute TTR intra-set for test_suites and utterances, and inter-set TTR differences.
    """
    set_dirs = {
        'test_suites': Path(test_suites_dir),
        'utterances': Path(utterances_dir)
    }
    stats = {}
    for set_name, dir_path in set_dirs.items():
        files = list(dir_path.glob('*.csv'))
        stats[set_name] = {}
        rows = []
        for f in files:
            df = pd.read_csv(f, header=0)
            # robust tokenization: select first column, lowercase, remove punctuation
            utterance_col = df.columns[0]
            series = df[utterance_col].dropna().astype(str)
            series = series.str.lower().str.replace(r'[^\w\s]', '', regex=True)
            tokens = series.str.split().explode().tolist()
            vocab = set(tokens)
            vocab_size = len(vocab)
            ttr = vocab_size / len(tokens) if tokens else 0.0
            stats[set_name][f.name] = {
                'vocab_size': vocab_size,
                'type_token_ratio': ttr
            }
            rows.append({
                'file': f.name,
                'vocab_size': vocab_size,
                'type_token_ratio': ttr
            })
        pd.DataFrame(rows).to_csv(Path(results_dir) / f'{set_name}_intra_lexical_analysis.csv', index=False)
    # inter-set: differences between sets for common files
    common_files = set(stats['test_suites'].keys()) & set(stats['utterances'].keys())
    if common_files:
        rows = []
        for fname in sorted(common_files):
            t1 = stats['test_suites'][fname]['type_token_ratio']
            t2 = stats['utterances'][fname]['type_token_ratio']
            rows.append({
                'file': fname,
                'test_suites_ttr': t1,
                'utterances_ttr': t2,
                'type_token_ratio_diff': abs(t1 - t2)
            })
        pd.DataFrame(rows).to_csv(Path(results_dir) / 'ttr_inter_set_diff.csv', index=False)

def main():
    base = Path(__file__).parent.parent
    test_suites_dir = base / 'test_suites'
    utterances_dir = base / 'utterances'
    results_dir = base / 'Results' / 'data' / 'lexical'
    results_dir.mkdir(exist_ok=True, parents=True)
    analyze_lexical(test_suites_dir, utterances_dir, results_dir)

if __name__ == '__main__':
    main()
