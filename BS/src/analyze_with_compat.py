#!/usr/bin/env python3
"""
analyze_with_compat.py

Semantic analysis with compatibility layer for newer huggingface-hub versions.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Add compatibility layer for huggingface_hub
import huggingface_hub

# Create a backward compatibility layer for sentence_transformers
if not hasattr(huggingface_hub, 'cached_download'):
    print("Adding cached_download compatibility layer")
    def cached_download(url, **kwargs):
        """
        Compatibility function that maps cached_download to hf_hub_download.
        This bridges older sentence-transformers with newer huggingface_hub.
        """
        # Parse repo_id and filename from URL
        parts = url.split('/')
        try:
            idx = parts.index('huggingface.co')
            repo_id = '/'.join(parts[idx+1:idx+3])
            filename = parts[-1]
        except ValueError:
            raise ValueError(f"Cannot parse URL: {url}")
            
        # Map to new API
        return huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=kwargs.get('cache_dir'),
            local_files_only=kwargs.get('local_files_only', False)
        )
    
    # Add to module
    huggingface_hub.cached_download = cached_download
    
    # Also add to utils if it exists
    if hasattr(huggingface_hub, 'utils'):
        huggingface_hub.utils.cached_download = cached_download

# Now import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    print("Successfully imported SentenceTransformer")
except ImportError as e:
    print(f"Error importing SentenceTransformer: {e}")
    sys.exit(1)


class TransformerAnalyzer:
    """Analyzes semantic cohesion using SentenceTransformer."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with the specified model."""
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device='cpu')
        print("Model loaded successfully")
        
    def compute_cohesion(self, texts):
        """Compute cohesion (avg similarity to centroid) for texts."""
        if not texts:
            return None, None
            
        # Encode texts using the transformer model
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
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
    results_dir = base / 'Results' / 'data' / 'transformer'
    
    analyze_datasets(test_dir, train_dir, results_dir)
    print(f"Analysis complete. Results saved to {results_dir}")


if __name__ == '__main__':
    main()
