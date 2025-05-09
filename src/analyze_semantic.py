"""
Script: analyze_semantic.py
Purpose: Perform semantic cohesion and similarity analysis for training and test datasets
         using sentence-transformers library.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_csv_data, load_csv_directory
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Define paths relative to the root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UTTERANCES_DIR = ROOT_DIR / 'utterances'
TEST_SUITES_DIR = ROOT_DIR / 'test_suites'
RESULTS_DIR = ROOT_DIR / 'Results' / 'Semantic'


def compute_cohesion(dir_path, model):
    """Compute cohesion (avg cosine to centroid) and return stats & centroids."""
    stats = []
    centroids = {}
    for f in Path(dir_path).glob('*.csv'):
        df = pd.read_csv(f, header=0)
        col = df.columns[0]
        texts = df[col].dropna().astype(str).tolist()
        if texts:
            # Encode texts with SentenceTransformer
            embs = model.encode(texts, convert_to_numpy=True)
            # Calculate centroid
            centroid = np.mean(embs, axis=0)
            # Calculate cosine similarity between each text and centroid
            sims = cosine_similarity(embs, centroid.reshape(1, -1)).flatten()
            cohesion = float(np.mean(sims))
        else:
            centroid = None
            cohesion = None
        stats.append({'file': f.name, 'cohesion': cohesion})
        centroids[f.name] = centroid
    return pd.DataFrame(stats), centroids


def compute_pairwise_similarity(centroids):
    """Flatten pairwise cosine similarity between centroids."""
    rows = []
    keys = list(centroids.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            f1, f2 = keys[i], keys[j]
            v1, v2 = centroids.get(f1), centroids.get(f2)
            if v1 is not None and v2 is not None:
                sim = float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])
            else:
                sim = None
            rows.append({'file1': f1, 'file2': f2, 'similarity': sim})
    return pd.DataFrame(rows)


def analyze_semantic(utterances_dir, test_suites_dir, results_dir, model_name='all-MiniLM-L6-v2'):
    """Run semantic cohesion and similarity analysis using SentenceTransformer."""
    # Initialize the sentence transformer model
    model = SentenceTransformer(model_name)
    results_dir = Path(results_dir)
    # Intra-set cohesion
    train_stats, train_centroids = compute_cohesion(utterances_dir, model)
    test_stats, test_centroids = compute_cohesion(test_suites_dir, model)
    # save
    train_stats.to_csv(results_dir / 'utterances_intra_semantic_cohesion.csv', index=False)
    test_stats.to_csv(results_dir / 'test_suites_intra_semantic_cohesion.csv', index=False)
    # Intra-set similarity
    train_sim = compute_pairwise_similarity(train_centroids)
    test_sim = compute_pairwise_similarity(test_centroids)
    train_sim.to_csv(results_dir / 'utterances_intra_semantic_similarity.csv', index=False)
    test_sim.to_csv(results_dir / 'test_suites_intra_semantic_similarity.csv', index=False)
    # Cross-set same-file similarity
    cross = []
    common = set(train_centroids) & set(test_centroids)
    for fname in sorted(common):
        v1, v2 = train_centroids.get(fname), test_centroids.get(fname)
        if v1 is not None and v2 is not None:
            sim = float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])
        else:
            sim = None
        cross.append({'file': fname, 'cross_similarity': sim})
    pd.DataFrame(cross).to_csv(results_dir / 'cross_semantic_similarity.csv', index=False)


def main():
    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    analyze_semantic(TEST_SUITES_DIR, UTTERANCES_DIR, RESULTS_DIR)


if __name__ == '__main__':
    main()
