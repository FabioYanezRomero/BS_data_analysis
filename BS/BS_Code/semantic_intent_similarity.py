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


def load_intents(json_path):
    """Load intent-to-utterances mapping from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_centroids(intents, model):
    """Compute embedding centroid for each intent."""
    centroids = {}
    for intent, utterances in intents.items():
        texts = [u for u in utterances if isinstance(u, str) and u.strip()]
        if texts:
            embs = model.encode(texts, convert_to_numpy=True)
            centroids[intent] = np.mean(embs, axis=0)
        else:
            centroids[intent] = None
    return centroids


def similarity_matrix(centroids):
    """Return list of intents and pairwise cosine-similarity matrix."""
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


def main():
    base = Path(__file__).parent.parent / 'utterances_tests'
    train_path = base / 'utterances.json'
    test_path = base / 'tests.json'

    model = SentenceTransformer('all-MiniLM-L6-v2')

    train_intents = load_intents(train_path)
    test_intents = load_intents(test_path)

    train_centroids = compute_centroids(train_intents, model)
    test_centroids = compute_centroids(test_intents, model)

    # Intra-set similarity
    train_labels, train_mat = similarity_matrix(train_centroids)
    test_labels, test_mat = similarity_matrix(test_centroids)

    # Cross-set same-intent similarity
    cross_sim = {}
    for intent in set(train_centroids) & set(test_centroids):
        v1 = train_centroids[intent]
        v2 = test_centroids[intent]
        if v1 is not None and v2 is not None:
            cross_sim[intent] = float(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0])
        else:
            cross_sim[intent] = None

    # Save JSON outputs
    train_json = {train_labels[i]: {train_labels[j]: train_mat[i, j] for j in range(len(train_labels))} for i in range(len(train_labels))}
    test_json = {test_labels[i]: {test_labels[j]: test_mat[i, j] for j in range(len(test_labels))} for i in range(len(test_labels))}

    (base / 'train_intent_similarity.json').write_text(json.dumps(train_json, ensure_ascii=False, indent=2), encoding='utf-8')
    (base / 'test_intent_similarity.json').write_text(json.dumps(test_json, ensure_ascii=False, indent=2), encoding='utf-8')
    (base / 'train_test_intent_similarity.json').write_text(json.dumps(cross_sim, ensure_ascii=False, indent=2), encoding='utf-8')

    # Heatmap visualizations
    df_train = pd.DataFrame(train_mat, index=train_labels, columns=train_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_train, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Train Intents Semantic Similarity')
    plt.tight_layout()
    plt.savefig(base / 'train_intent_similarity.png')
    plt.close()

    df_test = pd.DataFrame(test_mat, index=test_labels, columns=test_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_test, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Test Intents Semantic Similarity')
    plt.tight_layout()
    plt.savefig(base / 'test_intent_similarity.png')
    plt.close()

    print('Saved intent similarity matrices and heatmaps.')


if __name__ == '__main__':
    main()
