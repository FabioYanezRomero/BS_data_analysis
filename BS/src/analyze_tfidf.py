#!/usr/bin/env python3
"""
analyze_tfidf.py

Semantic analysis using TF-IDF vectorization with n-grams to approximate 
the behavior of sentence-transformers.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticAnalyzer:
    """Analyzes semantic cohesion using TF-IDF with n-grams."""
    
    def __init__(self):
        """Initialize with TF-IDF vectorizer configured for semantic analysis."""
        # Use character n-grams and word n-grams for better semantic representation
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # Character n-grams within word boundaries
            ngram_range=(2, 5),  # 2-5 character sequences
            max_features=10000,  # Limit vocabulary size
            sublinear_tf=True    # Apply sublinear scaling (log scaling)
        )
        
    def compute_cohesion(self, texts):
        """Compute cohesion (avg similarity to centroid) for texts."""
        if not texts:
            return None, None
            
        # Create document vectors
        X = self.vectorizer.fit_transform(texts)
        
        # Compute centroid (mean of all vectors)
        centroid = X.mean(axis=0)
        
        # Compute cosine similarity between each text and centroid
        similarities = cosine_similarity(X, centroid).flatten()
        cohesion = float(np.mean(similarities))
        
        return cohesion, centroid
        
    def compute_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return None
        return float(cosine_similarity(vec1, vec2)[0, 0])


def analyze_datasets(test_dir, train_dir, results_dir):
    """Analyze cohesion and similarity across datasets."""
    analyzer = SemanticAnalyzer()
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
        print(f"  {f.name}")
        df = pd.read_csv(f, header=0)
        col = df.columns[0]
        texts = df[col].dropna().astype(str).tolist()
        
        cohesion, centroid = analyzer.compute_cohesion(texts)
        test_cohesion.append({'file': f.name, 'cohesion': cohesion})
        test_centroids[f.name] = centroid
    
    # Process train files
    print("Processing train files...")
    for f in Path(train_dir).glob('*.csv'):
        print(f"  {f.name}")
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
            sim = analyzer.compute_similarity(v1, v2)
        else:
            sim = None
        cross_sim.append({'file': fname, 'similarity': sim})
    
    pd.DataFrame(cross_sim).to_csv(
        results_dir / 'cross_similarity.csv', index=False)
    
    # Compute pairwise similarity within test set
    print("Computing test set pairwise similarity...")
    test_pairs = []
    test_files = sorted(test_centroids.keys())
    for i in range(len(test_files)):
        for j in range(i+1, len(test_files)):
            f1, f2 = test_files[i], test_files[j]
            v1, v2 = test_centroids[f1], test_centroids[f2]
            if v1 is not None and v2 is not None:
                sim = analyzer.compute_similarity(v1, v2)
            else:
                sim = None
            test_pairs.append({'file1': f1, 'file2': f2, 'similarity': sim})
    
    pd.DataFrame(test_pairs).to_csv(
        results_dir / 'test_pairwise_similarity.csv', index=False)
    
    # Compute pairwise similarity within train set
    print("Computing training set pairwise similarity...")
    train_pairs = []
    train_files = sorted(train_centroids.keys())
    for i in range(len(train_files)):
        for j in range(i+1, len(train_files)):
            f1, f2 = train_files[i], train_files[j]
            v1, v2 = train_centroids[f1], train_centroids[f2]
            if v1 is not None and v2 is not None:
                sim = analyzer.compute_similarity(v1, v2)
            else:
                sim = None
            train_pairs.append({'file1': f1, 'file2': f2, 'similarity': sim})
    
    pd.DataFrame(train_pairs).to_csv(
        results_dir / 'train_pairwise_similarity.csv', index=False)


def main():
    """Main function."""
    base = Path(__file__).parent.parent
    test_dir = base / 'test_suites'
    train_dir = base / 'utterances'
    results_dir = base / 'Results' / 'data' / 'semantic'
    
    analyze_datasets(test_dir, train_dir, results_dir)
    print(f"Analysis complete. Results saved to {results_dir}")


if __name__ == '__main__':
    main()
