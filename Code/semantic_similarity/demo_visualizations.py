"""
Demo script for semantic similarity visualizations.

This script demonstrates how to use the distinctiveness and consistency visualization modules.
It generates synthetic data and creates visualizations for demonstration purposes.
"""

import numpy as np
from distinctiveness import DistinctivenessVisualizer
from consistency import ConsistencyVisualizer
from sentence_transformers import SentenceTransformer

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_centroids(num_categories=5, embedding_dim=384):
    """Generate synthetic category centroids for demonstration."""
    # Create some base centroids that are well-separated
    base_centroids = np.eye(num_categories, embedding_dim)
    
    # Add some random noise to make them more realistic
    centroids = {}
    for i in range(num_categories):
        noise = np.random.normal(0, 0.2, embedding_dim)
        centroid = base_centroids[i] + noise
        # Normalize to unit length
        centroid = centroid / np.linalg.norm(centroid)
        centroids[f"Category {i+1}"] = centroid
    
    return centroids

def generate_split_centroids(base_centroids, noise_level=0.1):
    """Generate a second set of centroids by adding noise to the base centroids."""
    split_centroids = {}
    for name, centroid in base_centroids.items():
        noise = np.random.normal(0, noise_level, len(centroid))
        new_centroid = centroid + noise
        # Normalize to unit length
        new_centroid = new_centroid / np.linalg.norm(new_centroid)
        split_centroids[name] = new_centroid
    return split_centroids

def main():
    print("Generating synthetic data...")
    
    # Generate synthetic centroids for two different splits
    base_centroids = generate_synthetic_centroids(num_categories=6)
    split1_centroids = base_centroids
    split2_centroids = generate_split_centroids(base_centroids, noise_level=0.15)
    
    print("\n=== Distinctiveness Visualization ===")
    print("Visualizing distinctiveness between categories...")
    dv = DistinctivenessVisualizer()
    
    # Create and save distinctiveness visualizations
    similarity_df, heatmap_fig = dv.compute_pairwise_distinctiveness(base_centroids)
    print(f"\nPairwise similarity matrix:")
    print(similarity_df.round(2))
    
    # Create centroid distance visualization
    print("\nGenerating centroid distance visualization...")
    dv.plot_centroid_distances(base_centroids, method='mds')
    
    print("\n=== Consistency Visualization ===")
    print("Visualizing consistency between splits...")
    cv = ConsistencyVisualizer()
    
    # Create and save consistency visualizations
    consistency_df, consistency_fig = cv.compute_cross_split_consistency(
        split1_centroids, 
        split2_centroids,
        split1_name="Training Split",
        split2_name="Test Split"
    )
    
    print("\nCross-split consistency scores:")
    print(consistency_df.round(3))
    
    # Create centroid shift visualization
    print("\nGenerating centroid shift visualization...")
    cv.visualize_centroid_shifts(
        split1_centroids,
        split2_centroids,
        method='mds',
        split1_name="Training",
        split2_name="Test"
    )
    
    print("\nVisualizations have been saved to the 'output' directory.")
    print("Look for files starting with 'distinctiveness_' and 'consistency_'")

if __name__ == "__main__":
    main()
