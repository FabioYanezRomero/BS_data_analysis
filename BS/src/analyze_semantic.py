"""
Script: analyze_semantic.py
Purpose: Perform semantic cohesion and similarity analysis for training and test datasets.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom text encoder to replace SentenceTransformer
class TextEncoder:
    def __init__(self, model_name=None):
        # We ignore model_name to maintain compatibility
        self.vectorizer = CountVectorizer(min_df=1, binary=True, 
                                        ngram_range=(1, 2))
        self.fitted = False
        
    def encode(self, sentences, convert_to_numpy=True):
        # Fit vectorizer if not already fitted
        if not self.fitted:
            self.vectorizer.fit(sentences)
            self.fitted = True
        # Transform sentences to vectors
        return self.vectorizer.transform(sentences).toarray()


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


def analyze_semantic(utterances_dir, test_suites_dir, results_dir):
    """Run semantic cohesion and similarity analysis."""
    # Use our custom encoder instead of SentenceTransformer
    model = TextEncoder('all-MiniLM-L6-v2')  # model name is ignored but kept for compatibility
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
    base = Path(__file__).parent.parent
    utterances_dir = base / 'utterances'
    test_suites_dir = base / 'test_suites'
    results_dir = base / 'Results' / 'data' / 'semantic'
    results_dir.mkdir(exist_ok=True, parents=True)
    analyze_semantic(utterances_dir, test_suites_dir, results_dir)


if __name__ == '__main__':
    main()
