"""
Consistency Visualization Module

This module provides visualization tools for analyzing the consistency of category centroids
across different data splits or time periods. It helps in understanding how stable the semantic
representations of categories are across different contexts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsistencyVisualizer:
    """
    A class for visualizing the consistency of category centroids across different data splits.
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the ConsistencyVisualizer.
        
        Args:
            output_dir: Directory to save output visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_centroid_shifts(
        self,
        data_split1: Dict[str, np.ndarray],
        data_split2: Dict[str, np.ndarray],
        method: str = 'pca',
        split1_name: str = "Train",
        split2_name: str = "Test",
        max_points_per_category: int = 100
    ) -> go.Figure:
        """
        Visualize the data points and centroid shifts between train and test splits for each category.
        
        Args:
            data_split1: Dictionary mapping category names to their data points in the first split (train)
            data_split2: Dictionary mapping category names to their data points in the second split (test)
            method: Dimensionality reduction method ('pca', 'tsne', 'mds')
            split1_name: Name for the first split (e.g., 'Train')
            split2_name: Name for the second split (e.g., 'Test')
            max_points_per_category: Maximum number of points to show per category (for performance)
            
        Returns:
            Plotly Figure object with the visualization
        """
        from sklearn.manifold import MDS, TSNE
        from sklearn.decomposition import PCA
        
        # Find common categories
        common_categories = sorted(set(data_split1.keys()) & set(data_split2.keys()))
        if not common_categories:
            raise ValueError("No common categories found between the two splits")
        
        # Prepare data for dimensionality reduction
        all_points = []
        point_labels = []
        point_splits = []
        
        # Process data from both splits
        for split_idx, (split_data, split_name) in enumerate([(data_split1, split1_name), (data_split2, split2_name)]):
            for cat in common_categories:
                points = split_data[cat]
                # Limit number of points for performance
                if len(points) > max_points_per_category:
                    points = points[np.random.choice(len(points), max_points_per_category, replace=False)]
                all_points.extend(points)
                point_labels.extend([cat] * len(points))
                point_splits.extend([split_name] * len(points))
        
        # Calculate centroids
        centroids1 = {cat: np.mean(points, axis=0) for cat, points in data_split1.items()}
        centroids2 = {cat: np.mean(points, axis=0) for cat, points in data_split2.items()}
        
        # Add centroids to the points for dimensionality reduction
        centroid_points = []
        centroid_labels = []
        centroid_splits = []
        
        for cat in common_categories:
            centroid_points.append(centroids1[cat])
            centroid_labels.append(f"{cat} (Centroid)")
            centroid_splits.append(f"{split1_name} Centroid")
            
            centroid_points.append(centroids2[cat])
            centroid_labels.append(f"{cat} (Centroid)")
            centroid_splits.append(f"{split2_name} Centroid")
        
        # Combine all points
        all_points = np.array(all_points + centroid_points)
        point_labels = point_labels + centroid_labels
        point_splits = point_splits + centroid_splits
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(all_points)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = reducer.fit_transform(all_points)
        elif method.lower() == 'mds':
            reducer = MDS(n_components=2, random_state=42)
            coords = reducer.fit_transform(all_points)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        # Separate points and centroids
        is_centroid = np.array(['Centroid' in split for split in point_splits])
        point_coords = coords[~is_centroid]
        point_labels_filtered = [label for i, label in enumerate(point_labels) if not is_centroid[i]]
        point_splits_filtered = [split for i, split in enumerate(point_splits) if not is_centroid[i]]
        
        centroid_coords = coords[is_centroid]
        centroid_labels_filtered = [label for i, label in enumerate(point_labels) if is_centroid[i]]
        centroid_splits_filtered = [split for i, split in enumerate(point_splits) if is_centroid[i]]
        
        # Create figure
        fig = go.Figure()
        
        # Add points for split 1 (Train)
        mask1 = np.array([s == split1_name for s in point_splits_filtered])
        fig.add_trace(go.Scatter(
            x=point_coords[mask1, 0],
            y=point_coords[mask1, 1],
            mode='markers',
            name=split1_name,
            text=[f"{cat}" for cat in np.array(point_labels_filtered)[mask1]],
            marker=dict(size=8, color='blue', opacity=0.3),
            hoverinfo='text',
            showlegend=True
        ))
        
        # Add points for split 2 (Test)
        mask2 = np.array([s == split2_name for s in point_splits_filtered])
        fig.add_trace(go.Scatter(
            x=point_coords[mask2, 0],
            y=point_coords[mask2, 1],
            mode='markers',
            name=split2_name,
            text=[f"{cat}" for cat in np.array(point_labels_filtered)[mask2]],
            marker=dict(size=8, color='red', opacity=0.3),
            hoverinfo='text',
            showlegend=True
        ))
        
        # Add centroids and connecting lines
        for cat in common_categories:
            # Get centroid coordinates
            train_centroid = centroid_coords[[i for i, (l, s) in enumerate(zip(centroid_labels_filtered, centroid_splits_filtered)) 
                                           if cat in l and split1_name in s]][0]
            test_centroid = centroid_coords[[i for i, (l, s) in enumerate(zip(centroid_labels_filtered, centroid_splits_filtered)) 
                                          if cat in l and split2_name in s]][0]
            
            # Add connecting line
            fig.add_trace(go.Scatter(
                x=[train_centroid[0], test_centroid[0]],
                y=[train_centroid[1], test_centroid[1]],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='none'
            ))
            
            # Add train centroid
            fig.add_trace(go.Scatter(
                x=[train_centroid[0]],
                y=[train_centroid[1]],
                mode='markers+text',
                name=f"{split1_name} Centroid",
                text=[f"{cat} ({split1_name} Centroid)"],
                textposition="top center",
                marker=dict(size=16, color='blue', symbol='circle'),
                textfont=dict(size=12, family="sans serif"),
                showlegend=cat == common_categories[0]  # Only show in legend once
            ))
            
            # Add test centroid
            fig.add_trace(go.Scatter(
                x=[test_centroid[0]],
                y=[test_centroid[1]],
                mode='markers+text',
                name=f"{split2_name} Centroid",
                text=[f"{cat} ({split2_name} Centroid)"],
                textposition="bottom center",
                marker=dict(size=16, color='red', symbol='x'),
                textfont=dict(size=12, family="sans serif"),
                showlegend=cat == common_categories[0]  # Only show in legend once
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{split1_name} vs {split2_name} Data Points and Centroid Shifts ({method.upper()})",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            plot_bgcolor='white',
            width=1200,
            height=1000,
            margin=dict(l=20, r=20, t=80, b=20),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            )
        )
        
        # Configure figure for high-resolution export
        config = {
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'filename': f'centroid_shifts_{method}',
                'height': 2000,
                'width': 2400,
                'scale': 4  # Multiply title/legend/axis/canvas sizes by this factor
            },
            'scrollZoom': True
        }
        
        # Save the figure with high resolution
        html_file = self.output_dir / f'centroid_shifts_{method}.html'
        png_file = self.output_dir / f'centroid_shifts_{method}.png'
        
        # Save HTML with config
        fig.write_html(str(html_file), config=config)
        
        # Update layout for high-res PNG
        fig.update_layout(
            width=2400,
            height=2000,
            font=dict(size=36),  # Larger font for high-res
            margin=dict(l=100, r=100, t=150, b=100)  # Adjust margins for larger plot
        )
        
        # Save high-res PNG
        fig.write_image(str(png_file), scale=4, engine='kaleido')
        
        logger.info(f"Saved centroid shifts visualization to {html_file} and {png_file}")
        
        return fig
    
    def compute_cross_split_consistency(
        self,
        centroids_split1: Dict[str, np.ndarray],
        centroids_split2: Dict[str, np.ndarray],
        split1_name: str = "Split 1",
        split2_name: str = "Split 2"
    ) -> Tuple[pd.DataFrame, go.Figure]:
        """
        Compute and visualize consistency of category centroids across two data splits.
        
        Args:
            centroids_split1: Dictionary mapping category names to their centroid embeddings in first split
            centroids_split2: Dictionary mapping category names to their centroid embeddings in second split
            split1_name: Name for the first split (for display purposes)
            split2_name: Name for the second split (for display purposes)
            
        Returns:
            Tuple containing:
                - DataFrame with consistency scores for each category
                - Plotly Figure object with the consistency visualization
        """
        # Find common categories
        common_categories = set(centroids_split1.keys()) & set(centroids_split2.keys())
        if not common_categories:
            raise ValueError("No common categories found between the two splits")
        
        # Compute similarity scores
        results = []
        for category in common_categories:
            vec1 = centroids_split1[category]
            vec2 = centroids_split2[category]
            
            # Skip invalid vectors
            if vec1 is None or vec2 is None or np.all(vec1 == 0) or np.all(vec2 == 0):
                continue
                
            # Compute cosine similarity
            similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
            
            # Ensure valid similarity score
            if np.isnan(similarity) or np.isinf(similarity):
                continue
                
            results.append({
                'category': category,
                'similarity': similarity,
                'consistency': similarity  # For backward compatibility
            })
        
        if not results:
            raise ValueError("No valid similarity scores could be computed")
        
        # Create DataFrame
        df = pd.DataFrame(results).sort_values('similarity', ascending=False)
        
        # Create bar plot
        fig = px.bar(
            df,
            x='category',
            y='similarity',
            title=f'Cross-Split Consistency: {split1_name} vs {split2_name}',
            labels={'similarity': 'Cosine Similarity', 'category': 'Category'},
            color='similarity',
            color_continuous_scale='Viridis',
            range_color=[0, 1]  # Similarity range from 0 to 1
        )
        
        # Update layout
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis=dict(range=[0, 1.1]),
            width=1000,
            height=600,
            margin=dict(l=50, r=50, t=80, b=150),
            coloraxis_colorbar=dict(title='Similarity')
        )
        
        # Add horizontal line at 1.0
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.5)
        
        # Save the figure
        html_file = self.output_dir / 'cross_split_consistency.html'
        png_file = self.output_dir / 'cross_split_consistency.png'
        
        fig.write_html(str(html_file))
        fig.update_layout(
            width=2000,
            height=1200,
            font=dict(size=18)
        )
        fig.write_image(str(png_file), scale=4)
        
        logger.info(f"Saved cross-split consistency visualization to {html_file} and {png_file}")
        
        return df, fig


if __name__ == "__main__":
    """
    When run directly, this script will generate example visualizations using synthetic data.
    """
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate some example data - two sets of data points for comparison
    # Using fewer categories for cleaner visualization
    categories = [f"Category {i+1}" for i in range(3)]  # Reduced from 5 to 3 categories
    n_points_per_category = 50
    embedding_dim = 384
    
    # First set of data points (train split)
    train_data = {}
    base_patterns = np.eye(len(categories), embedding_dim)
    
    for i, category in enumerate(categories):
        # Create base pattern with some noise for each point
        points = []
        for _ in range(n_points_per_category):
            point = base_patterns[i] + np.random.normal(0, 0.15, embedding_dim)
            # Add some structure
            if i > 0:
                point += 0.1 * base_patterns[i-1]
            # Normalize
            point = point / np.linalg.norm(point)
            points.append(point)
        train_data[category] = np.array(points)
    
    # Second set of data points (test split) - slightly shifted from the first
    test_data = {}
    for cat in categories:
        points = []
        for point in train_data[cat]:
            # Add some shift to each point
            shift = np.random.normal(0, 0.2, embedding_dim)
            shifted = point + 0.3 * shift
            # Normalize
            shifted = shifted / np.linalg.norm(shifted)
            points.append(shifted)
        test_data[cat] = np.array(points)
    
    # Create visualizations
    print("Generating consistency visualizations...")
    visualizer = ConsistencyVisualizer()
    
    # 1. Cross-split consistency (compute centroids first)
    print("Creating cross-split consistency visualization...")
    train_centroids = {cat: np.mean(points, axis=0) for cat, points in train_data.items()}
    test_centroids = {cat: np.mean(points, axis=0) for cat, points in test_data.items()}
    
    consistency_df, consistency_fig = visualizer.compute_cross_split_consistency(
        train_centroids,
        test_centroids,
        split1_name="Training",
        split2_name="Test"
    )
    
    print("\nCross-split consistency scores:")
    print(consistency_df.round(3))
    
    # 2. Data points and centroid shifts
    print("\nCreating data points and centroid shifts visualization...")
    fig = visualizer.visualize_centroid_shifts(
        train_data,
        test_data,
        method='pca',
        split1_name="Training",
        split2_name="Test",
        max_points_per_category=30  # Limit points for better visualization
    )
    
    print("\nVisualizations have been saved to the 'output' directory.")
    print("Look for files starting with 'cross_split_consistency' and 'centroid_shifts_'")
