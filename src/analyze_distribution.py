"""
Script: analyze_distribution.py
Purpose: Perform distributional analysis (kurtosis, EMD, JS divergence).
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_csv_data, load_csv_directory
from scipy.stats import kurtosis, entropy, wasserstein_distance

# Define paths relative to the root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UTTERANCES_DIR = ROOT_DIR / 'utterances'
TEST_SUITES_DIR = ROOT_DIR / 'test_suites'
RESULTS_DIR = ROOT_DIR / 'Results' / 'Distribution'

def analyze_distribution(test_suites_dir, utterances_dir, results_dir):
    """Compute intra-set kurtosis and inter-set EMD/JS divergence per intent."""
    set_dirs = {'test_suites': Path(test_suites_dir), 'utterances': Path(utterances_dir)}
    distributions = {}
    # Intra-set: collect sentence/word distributions
    for set_name, dir_path in set_dirs.items():
        distributions[set_name] = {}
        for f in dir_path.glob('*.csv'):
            df = pd.read_csv(f, header=0)
            col = df.columns[0]
            series = df[col].dropna().astype(str)
            series = series.str.lower().str.replace(r'[^\w\s]', '', regex=True)
            tokens = series.str.split()
            sl = tokens.apply(len).tolist()
            wl = tokens.explode().map(len).tolist()
            distributions[set_name][f.name] = {'sl': sl, 'wl': wl}
    # Combined kurtosis CSV: test vs train for each file
    kurt_rows = []
    all_files = set(distributions['test_suites']) | set(distributions['utterances'])
    for fname in sorted(all_files):
        sl_ts = distributions['test_suites'].get(fname, {}).get('sl', [])
        wl_ts = distributions['test_suites'].get(fname, {}).get('wl', [])
        sl_ut = distributions['utterances'].get(fname, {}).get('sl', [])
        wl_ut = distributions['utterances'].get(fname, {}).get('wl', [])
        kurt_rows.append({
            'file': fname,
            'test_sentence_length_kurtosis': float(kurtosis(sl_ts)) if len(sl_ts) > 3 else None,
            'test_word_length_kurtosis': float(kurtosis(wl_ts)) if len(wl_ts) > 3 else None,
            'train_sentence_length_kurtosis': float(kurtosis(sl_ut)) if len(sl_ut) > 3 else None,
            'train_word_length_kurtosis': float(kurtosis(wl_ut)) if len(wl_ut) > 3 else None
        })
    pd.DataFrame(kurt_rows).to_csv(Path(results_dir) / 'distribution_kurtosis.csv', index=False)
    # Inter-set: EMD only between train and test per file
    common = set(distributions['test_suites']) & set(distributions['utterances'])
    emd_rows = []
    for fname in sorted(common):
        sl_ts = distributions['test_suites'][fname]['sl']
        sl_ut = distributions['utterances'][fname]['sl']
        wl_ts = distributions['test_suites'][fname]['wl']
        wl_ut = distributions['utterances'][fname]['wl']
        emd_sl = wasserstein_distance(sl_ts, sl_ut)
        emd_wl = wasserstein_distance(wl_ts, wl_ut)
        emd_rows.append({
            'file': fname,
            'emd_sentence_length': emd_sl,
            'emd_word_length': emd_wl
        })
    if emd_rows:
        pd.DataFrame(emd_rows).to_csv(Path(results_dir) / 'distribution_emd.csv', index=False)

    # Inter-set: JS divergence between train and test
    js_rows = []
    for fname in sorted(common):
        sl_ts = distributions['test_suites'][fname]['sl']
        sl_ut = distributions['utterances'][fname]['sl']
        wl_ts = distributions['test_suites'][fname]['wl']
        wl_ut = distributions['utterances'][fname]['wl']
        # JS divergence for sentence lengths
        lengths_sl = sorted(set(sl_ts) | set(sl_ut))
        c_ts_sl = np.array([sl_ts.count(x) for x in lengths_sl], dtype=float)
        c_ut_sl = np.array([sl_ut.count(x) for x in lengths_sl], dtype=float)
        eps = 1e-9
        P_sl = (c_ts_sl + eps) / (c_ts_sl.sum() + eps * len(lengths_sl))
        Q_sl = (c_ut_sl + eps) / (c_ut_sl.sum() + eps * len(lengths_sl))
        M_sl = 0.5 * (P_sl + Q_sl)
        js_sl = float(0.5 * entropy(P_sl, M_sl) + 0.5 * entropy(Q_sl, M_sl))
        # JS divergence for word lengths
        lengths_wl = sorted(set(wl_ts) | set(wl_ut))
        c_ts_wl = np.array([wl_ts.count(x) for x in lengths_wl], dtype=float)
        c_ut_wl = np.array([wl_ut.count(x) for x in lengths_wl], dtype=float)
        P_wl = (c_ts_wl + eps) / (c_ts_wl.sum() + eps * len(lengths_wl))
        Q_wl = (c_ut_wl + eps) / (c_ut_wl.sum() + eps * len(lengths_wl))
        M_wl = 0.5 * (P_wl + Q_wl)
        js_wl = float(0.5 * entropy(P_wl, M_wl) + 0.5 * entropy(Q_wl, M_wl))
        js_rows.append({
            'file': fname,
            'js_sentence_length': js_sl,
            'js_word_length': js_wl
        })
    if js_rows:
        pd.DataFrame(js_rows).to_csv(Path(results_dir) / 'distribution_js.csv', index=False)

def main():
    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    analyze_distribution(TEST_SUITES_DIR, UTTERANCES_DIR, RESULTS_DIR)

if __name__ == '__main__':
    main()
