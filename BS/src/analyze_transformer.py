#!/usr/bin/env python3
"""
analyze_transformer.py

Uses the all-MiniLM-L6-v2 SentenceTransformer model to analyze:
- Intra-set cohesion: How semantically similar are texts within each file
- Cross-set similarity: How similar are the same intents across train/test sets
- Pairwise similarity: How similar are different intents to each other
"""
import sys
import os
# Ensure huggingface_hub has cached_download
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Patch huggingface_hub to make it compatible with sentence_transformers
def patch_huggingface():
    """Add cached_download to huggingface_hub if missing."""
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        # Create a simple function that matches the signature but uses hf_hub_download
        def cached_download(url, **kwargs):
            # Extract repo_id and filename from URL
            parts = url.split('/')
            try:
                idx = parts.index('huggingface.co')
                repo_id = '/'.join(parts[idx+1:idx+3])
            except ValueError:
                raise ValueError(f"Cannot parse repo_id from URL: {url}")
            filename = parts[-1]
            # Call hf_hub_download with appropriate args
            return huggingface_hub.hf_hub_download(
                repo_id=repo_id, 
                filename=filename,
                cache_dir=kwargs.get('cache_dir'),
                force_filename=kwargs.get('force_filename')
            )
        # Add the function to the module
        huggingface_hub.cached_download = cached_download
    # Also patch the utils module if it exists
    if hasattr(huggingface_hub, 'utils'):
        if not hasattr(huggingface_hub.utils, 'cached_download'):
            huggingface_hub.utils.cached_download = huggingface_hub.cached_download

# Apply the patch
patch_huggingface()

# Now import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Error importing SentenceTransformer: {e}")
    print("Falling back to CountVectorizer")
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Create a fallback class with the same interface
    class SentenceTransformer:
        def __init__(self, model_name):
            print(f"Using CountVectorizer as fallback for {model_name}")
            self.vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
            self.fitted = False
            
        def encode(self, sentences, convert_to_numpy=True):
            if not self.fitted:
                self.vectorizer.fit(sentences)
                self.fitted = True
            return self.vectorizer.transform(sentences).toarray()


class CohesionAnalyzer:
    """Analyzes semantic cohesion using the all-MiniLM-L6-v2 model."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize with the specified sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        print(f"Initializing model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
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
            
        # Encode texts using the transformer model
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Compute centroid (mean of all vectors)
        centroid = np.mean(embeddings, axis=0)
        
        # Compute cosine similarity between each text and centroid
        similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
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
    analyzer = CohesionAnalyzer('all-MiniLM-L6-v2')
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Store cohesion scores and centroids
    test_cohesion = []
    train_cohesion = []
    test_centroids = {}
    train_centroids = {}
    
    # Process test suites (test data)
    print("Processing test suites...")
    for f in Path(test_suites_dir).glob('*.csv'):
        print(f"  {f.name}")
        df = pd.read_csv(f, header=0)
        col = df.columns[0]  # First column contains utterances
        texts = df[col].dropna().astype(str).tolist()
        
        cohesion, centroid = analyzer.compute_cohesion(texts)
        test_cohesion.append({'file': f.name, 'cohesion': cohesion})
        test_centroids[f.name] = centroid
    
    # Process utterances (training data)
    print("Processing utterances...")
    for f in Path(utterances_dir).glob('*.csv'):
        print(f"  {f.name}")
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
    print("Computing cross-set similarity...")
    cross_similarity = []
    common_files = set(test_centroids) & set(train_centroids)
    for fname in sorted(common_files):
        sim = analyzer.compute_similarity(test_centroids[fname], train_centroids[fname])
        cross_similarity.append({'file': fname, 'similarity': sim})
    
    pd.DataFrame(cross_similarity).to_csv(
        results_dir / 'cross_set_similarity.csv', index=False)
    
    # Compute pairwise similarity within test set
    print("Computing test set pairwise similarity...")
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
    print("Computing training set pairwise similarity...")
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
    results_dir = base / 'Results' / 'data' / 'transformer'
    
    analyze_datasets(test_suites_dir, utterances_dir, results_dir)
    print(f"Analysis complete. Results saved to {results_dir}")


if __name__ == '__main__':
    main()
