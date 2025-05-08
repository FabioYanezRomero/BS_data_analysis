#!/usr/bin/env python3
"""
semantic_intent_similarity.py

Computes semantic similarity between chatbot intents using sentence embeddings:
 - Intra-set similarity matrices for train and test
 - Cross-set (train vs test) same-intent similarity
Outputs JSON and heatmap PNGs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticIntentAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the semantic intent analyzer with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def load_intents(self, json_path):
        """
        Load intent-to-utterances mapping from JSON.
        
        Args:
            json_path: Path to the JSON file containing intent data
            
        Returns:
            dict: Mapping of intent names to utterance lists
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compute_centroids(self, intents):
        """
        Compute embedding centroid for each intent.
        
        Args:
            intents: Dictionary mapping intent names to utterance lists
            
        Returns:
            dict: Mapping of intent names to embedding centroids
        """
        centroids = {}
        for intent, utterances in intents.items():
            texts = [u for u in utterances if isinstance(u, str) and u.strip()]
            if texts:
                embs = self.model.encode(texts, convert_to_numpy=True)
                centroids[intent] = np.mean(embs, axis=0)
            else:
                centroids[intent] = None
        return centroids
    
    def similarity_matrix(self, centroids):
        """
        Return list of intents and pairwise cosine-similarity matrix.
        
        Args:
            centroids: Dictionary mapping intent names to embedding centroids
            
        Returns:
            tuple: (list of intent names, similarity matrix)
        """
        intents = list(centroids.keys())
        mat = np.zeros((len(intents), len(intents)))
        for i, a in enumerate(intents):
            for j, b in enumerate(intents):
                v1 = centroids.get(a)
                v2 = centroids.get(b)
                if v1 is not None and v2 is not None:
                    mat[i, j] = float(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0])
                else:
                    mat[i, j] = None
        return intents, mat
    
    def compute_cross_similarity(self, train_centroids, test_centroids):
        """
        Compute cross-set similarity between same intents in train and test sets.
        
        Args:
            train_centroids: Training set centroids
            test_centroids: Test set centroids
            
        Returns:
            dict: Mapping of intent names to similarity scores
        """
        cross_sim = {}
        for intent in set(train_centroids) & set(test_centroids):
            v1 = train_centroids[intent]
            v2 = test_centroids[intent]
            if v1 is not None and v2 is not None:
                cross_sim[intent] = float(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0])
            else:
                cross_sim[intent] = None
        return cross_sim
    
    def create_heatmap(self, labels, matrix, title, output_path):
        """
        Create and save a heatmap visualization from a similarity matrix.
        
        Args:
            labels: List of intent names
            matrix: Similarity matrix
            title: Title for the heatmap
            output_path: Path to save the output image
        """
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='viridis')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def main():
    """Main function to run the semantic intent similarity analysis"""
    # Set up paths
    base = Path(__file__).parent.parent / 'utterances_tests'
    train_path = base / 'utterances.json'
    test_path = base / 'tests.json'
    
    # Initialize analyzer
    analyzer = SemanticIntentAnalyzer()
    
    # Load intent data
    print("Loading intent data...")
    train_intents = analyzer.load_intents(train_path)
    test_intents = analyzer.load_intents(test_path)
    
    # Compute intent centroids
    print("Computing intent centroids...")
    train_centroids = analyzer.compute_centroids(train_intents)
    test_centroids = analyzer.compute_centroids(test_intents)
    
    # Calculate intra-set similarity matrices
    print("Calculating similarity matrices...")
    train_labels, train_mat = analyzer.similarity_matrix(train_centroids)
    test_labels, test_mat = analyzer.similarity_matrix(test_centroids)
    
    # Calculate cross-set similarity
    print("Calculating cross-set similarity...")
    cross_sim = analyzer.compute_cross_similarity(train_centroids, test_centroids)
    
    # Save JSON outputs
    print("Saving similarity data to JSON...")
    train_json = {train_labels[i]: {train_labels[j]: train_mat[i, j] for j in range(len(train_labels))} for i in range(len(train_labels))}
    test_json = {test_labels[i]: {test_labels[j]: test_mat[i, j] for j in range(len(test_labels))} for i in range(len(test_labels))}
    
    (base / 'train_intent_similarity.json').write_text(json.dumps(train_json, ensure_ascii=False, indent=2), encoding='utf-8')
    (base / 'test_intent_similarity.json').write_text(json.dumps(test_json, ensure_ascii=False, indent=2), encoding='utf-8')
    (base / 'train_test_intent_similarity.json').write_text(json.dumps(cross_sim, ensure_ascii=False, indent=2), encoding='utf-8')
    
    # Create heatmap visualizations
    print("Creating heatmap visualizations...")
    analyzer.create_heatmap(
        train_labels, 
        train_mat, 
        'Train Intents Semantic Similarity', 
        base / 'train_intent_similarity.png'
    )
    
    analyzer.create_heatmap(
        test_labels, 
        test_mat, 
        'Test Intents Semantic Similarity', 
        base / 'test_intent_similarity.png'
    )
    
    print('Saved intent similarity matrices and heatmaps.')


if __name__ == '__main__':
    main()
