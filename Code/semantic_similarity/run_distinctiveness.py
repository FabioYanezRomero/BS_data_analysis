"""
Example script to run distinctiveness analysis on your data.
"""

import numpy as np
from distinctiveness import DistinctivenessVisualizer

# Example data structure - replace this with your actual data
# This is just a template showing the expected format

def load_your_data():
    """
    Replace this function with your actual data loading logic.
    Should return a dictionary mapping category names to their centroid embeddings.
    """
    # Example: Replace this with your actual data loading code
    # centroids = {
    #     "Category 1": np.array([...]),
    #     "Category 2": np.array([...]),
    #     ...
    # }
    # return centroids
    
    # For demo purposes, we'll generate some synthetic data
    print("Generating synthetic data for demonstration...")
    np.random.seed(42)
    
    # Create 5 categories with 384-dimensional embeddings
    categories = [f"Category {i+1}" for i in range(5)]
    centroids = {}
    
    # Create some base patterns
    base_patterns = np.eye(5, 384)
    
    for i, category in enumerate(categories):
        # Start with a base pattern and add some noise
        centroid = base_patterns[i] + np.random.normal(0, 0.15, 384)
        # Normalize to unit length
        centroid = centroid / np.linalg.norm(centroid)
        centroids[category] = centroid
    
    return centroids

def main():
    # Load your data
    print("Loading data...")
    centroids = load_your_data()
    
    print(f"Loaded {len(centroids)} categories")
    
    # Initialize the visualizer
    print("Creating visualizations...")
    visualizer = DistinctivenessVisualizer()
    
    # 1. Create pairwise distinctiveness heatmap
    print("Generating distinctiveness heatmap...")
    similarity_df, heatmap_fig = visualizer.compute_pairwise_distinctiveness(
        centroids,
        metric='cosine'  # or 'euclidean'
    )
    
    print("\nPairwise similarity matrix:")
    print(similarity_df.round(2))
    
    # 2. Create centroid distance visualization
    print("\nGenerating centroid distance visualization...")
    visualizer.plot_centroid_distances(
        centroids,
        method='mds'  # or 'tsne', 'pca'
    )
    
    print("\nVisualizations have been saved to the 'output' directory.")
    print("Look for files starting with 'distinctiveness_' and 'centroid_distances_'")

if __name__ == "__main__":
    main()
