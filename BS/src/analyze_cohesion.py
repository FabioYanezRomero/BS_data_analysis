#!/usr/bin/env python3
"""
analyze_cohesion.py

Analyzes semantic cohesion within datasets and similarity between datasets:
- Intra-set cohesion: How semantically similar are texts within each file
- Inter-set similarity: How similar are the same intents across train/test sets
- Pairwise similarity: How similar are different intents to each other

Uses CountVectorizer with n-grams as a reliable alternative to neural embeddings.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CohesionAnalyzer:
    """Analyzes semantic cohesion and similarity using count-based embeddings."""
    
    def __init__(self):
        """Initialize the analyzer with a count vectorizer."""
        # Use binary counts and unigrams+bigrams for better semantic representation
        self.vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        
    def compute_cohesion(self, texts):
        """
        Compute semantic cohesion (avg similarity to centroid) for a set of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            tuple: (cohesion score, centroid vector)
        """
        if not texts:
            return None, None
            
        # Create document vectors
        X = self.vectorizer.fit_transform(texts)
        X_array = X.toarray()
        
        # Compute centroid (mean of all vectors)
        centroid = np.mean(X_array, axis=0).reshape(1, -1)
        
        # Compute cosine similarity between each text and centroid
        similarities = cosine_similarity(X_array, centroid).flatten()
        cohesion = float(np.mean(similarities))
        
        return cohesion, centroid
        
    def compute_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return None
        return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0])


def analyze_datasets(test_suites_dir, utterances_dir, results_dir):
    """
    Analyze cohesion and similarity across datasets.
    
    Args:
        test_suites_dir: Directory containing test suite CSV files
        utterances_dir: Directory containing utterance CSV files
        results_dir: Directory to save results
    """
    analyzer = CohesionAnalyzer()
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Store cohesion scores and centroids
    test_cohesion = []
    train_cohesion = []
    test_centroids = {}
    train_centroids = {}
    
    # Process test suites (test data)
    for f in Path(test_suites_dir).glob('*.csv'):
        df = pd.read_csv(f, header=0)
        col = df.columns[0]  # First column contains utterances
        texts = df[col].dropna().astype(str).tolist()
        
        cohesion, centroid = analyzer.compute_cohesion(texts)
        test_cohesion.append({'file': f.name, 'cohesion': cohesion})
        test_centroids[f.name] = centroid
    
    # Process utterances (training data)
    for f in Path(utterances_dir).glob('*.csv'):
        df = pd.read_csv(f, header=0)
        col = df.columns[0]
        texts = df[col].dropna().astype(str).tolist()
        
        cohesion, centroid = analyzer.compute_cohesion(texts)
        train_cohesion.append({'file': f.name, 'cohesion': cohesion})
        train_centroids[f.name] = centroid
    
    # Save intra-set cohesion scores
    pd.DataFrame(test_cohesion).to_csv(
        results_dir / 'test_suites_cohesion.csv', index=False)
    pd.DataFrame(train_cohesion).to_csv(
        results_dir / 'utterances_cohesion.csv', index=False)
    
    # Compute cross-set similarity for matching files
    cross_similarity = []
    common_files = set(test_centroids) & set(train_centroids)
    for fname in sorted(common_files):
        sim = analyzer.compute_similarity(test_centroids[fname], train_centroids[fname])
        cross_similarity.append({'file': fname, 'similarity': sim})
    
    pd.DataFrame(cross_similarity).to_csv(
        results_dir / 'cross_set_similarity.csv', index=False)
    
    # Compute pairwise similarity within test set
    test_pairs = []
    test_files = sorted(test_centroids.keys())
    for i in range(len(test_files)):
        for j in range(i+1, len(test_files)):
            f1, f2 = test_files[i], test_files[j]
            sim = analyzer.compute_similarity(test_centroids[f1], test_centroids[f2])
            test_pairs.append({'file1': f1, 'file2': f2, 'similarity': sim})
    
    pd.DataFrame(test_pairs).to_csv(
        results_dir / 'test_pairwise_similarity.csv', index=False)
    
    # Compute pairwise similarity within train set
    train_pairs = []
    train_files = sorted(train_centroids.keys())
    for i in range(len(train_files)):
        for j in range(i+1, len(train_files)):
            f1, f2 = train_files[i], train_files[j]
            sim = analyzer.compute_similarity(train_centroids[f1], train_centroids[f2])
            train_pairs.append({'file1': f1, 'file2': f2, 'similarity': sim})
    
    pd.DataFrame(train_pairs).to_csv(
        results_dir / 'train_pairwise_similarity.csv', index=False)


def main():
    """Main function to run the cohesion analysis."""
    base = Path(__file__).parent.parent
    test_suites_dir = base / 'test_suites'
    utterances_dir = base / 'utterances'
    results_dir = base / 'Results' / 'data' / 'cohesion'
    
    analyze_datasets(test_suites_dir, utterances_dir, results_dir)
    print(f"Cohesion analysis complete. Results saved to {results_dir}")


if __name__ == '__main__':
    main()
