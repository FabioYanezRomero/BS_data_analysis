#!/usr/bin/env python3
"""
analyze_direct.py

Direct implementation using SentenceTransformer with all-MiniLM-L6-v2 model.
Analyzes semantic cohesion and similarity across datasets.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TransformerAnalyzer:
    """Analyzes semantic cohesion using SentenceTransformer."""
    
    def __init__(self):
        """Initialize with all-MiniLM-L6-v2 model."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2', token=False)
        
    def compute_cohesion(self, texts):
        """Compute cohesion (avg similarity to centroid) for texts."""
        if not texts:
            return None, None
            
        # Encode texts using the transformer model
        embeddings = self.model.encode(texts)
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Compute similarities to centroid
        similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        cohesion = float(np.mean(similarities))
        
        return cohesion, centroid


def analyze_datasets(test_dir, train_dir, results_dir):
    """Analyze cohesion and similarity across datasets."""
    analyzer = TransformerAnalyzer()
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Process datasets
    test_cohesion = []
    train_cohesion = []
    test_centroids = {}
    train_centroids = {}
    
    # Process test files
    print("Processing test files...")
    for f in Path(test_dir).glob('*.csv'):
        df = pd.read_csv(f, header=0)
        col = df.columns[0]
        texts = df[col].dropna().astype(str).tolist()
        
        cohesion, centroid = analyzer.compute_cohesion(texts)
        test_cohesion.append({'file': f.name, 'cohesion': cohesion})
        test_centroids[f.name] = centroid
    
    # Process train files
    print("Processing train files...")
    for f in Path(train_dir).glob('*.csv'):
        df = pd.read_csv(f, header=0)
        col = df.columns[0]
        texts = df[col].dropna().astype(str).tolist()
        
        cohesion, centroid = analyzer.compute_cohesion(texts)
        train_cohesion.append({'file': f.name, 'cohesion': cohesion})
        train_centroids[f.name] = centroid
    
    # Save cohesion scores
    pd.DataFrame(test_cohesion).to_csv(
        results_dir / 'test_cohesion.csv', index=False)
    pd.DataFrame(train_cohesion).to_csv(
        results_dir / 'train_cohesion.csv', index=False)
    
    # Compute cross-set similarity
    print("Computing cross-set similarity...")
    cross_sim = []
    common = set(test_centroids) & set(train_centroids)
    for fname in sorted(common):
        v1 = test_centroids[fname]
        v2 = train_centroids[fname]
        if v1 is not None and v2 is not None:
            sim = float(cosine_similarity(
                v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])
        else:
            sim = None
        cross_sim.append({'file': fname, 'similarity': sim})
    
    pd.DataFrame(cross_sim).to_csv(
        results_dir / 'cross_similarity.csv', index=False)


def main():
    """Main function."""
    base = Path(__file__).parent.parent
    test_dir = base / 'test_suites'
    train_dir = base / 'utterances'
    results_dir = base / 'Results' / 'data' / 'direct'
    
    analyze_datasets(test_dir, train_dir, results_dir)
    print(f"Analysis complete. Results saved to {results_dir}")


if __name__ == '__main__':
    main()
