"""
Distinctiveness Visualization Module

This module provides visualization tools for analyzing the distinctiveness between different categories
in a semantic space. It helps in understanding how well-separated different categories are from each other.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistinctivenessVisualizer:
    """
    A class for visualizing the distinctiveness between categories in a semantic space.
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the DistinctivenessVisualizer.
        
        Args:
            output_dir: Directory to save output visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_pairwise_distinctiveness(
        self,
        centroids: Dict[str, np.ndarray],
        metric: str = 'cosine'
    ) -> Tuple[pd.DataFrame, go.Figure]:
        """
        Compute and visualize pairwise distinctiveness between category centroids.
        
        Args:
            centroids: Dictionary mapping category names to their centroid embeddings
            metric: Similarity metric to use ('cosine', 'euclidean')
            
        Returns:
            Tuple containing:
                - DataFrame with pairwise distinctiveness scores
                - Plotly Figure object with the heatmap visualization
        """
        if not centroids or len(centroids) < 2:
            raise ValueError("At least two categories are required for distinctiveness analysis")
        
        # Convert centroids to a matrix
        labels = list(centroids.keys())
        centroid_matrix = np.array([centroids[label] for label in labels])
        
        # Compute similarity matrix
        if metric == 'cosine':
            similarity_matrix = cosine_similarity(centroid_matrix)
        elif metric == 'euclidean':
            distances = np.linalg.norm(centroid_matrix[:, np.newaxis] - centroid_matrix, axis=2)
            similarity_matrix = 1 / (1 + distances)  # Convert distance to similarity
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Create DataFrame
        df = pd.DataFrame(
            similarity_matrix,
            index=labels,
            columns=labels
        )
        
        # Create heatmap
        fig = px.imshow(
            similarity_matrix,
            x=labels,
            y=labels,
            labels=dict(x="Category", y="Category", color="Similarity"),
            color_continuous_scale='Viridis',
            aspect="auto",
            title=f"Inter-Category Similarity ({metric.capitalize()})",
            zmin=0,  # Similarity ranges from 0 to 1
            zmax=1
        )
        
        # Add annotations
        annotations = []
        for i, row in enumerate(similarity_matrix):
            for j, val in enumerate(row):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{val:.2f}",
                        font=dict(color="white" if val < 0.5 else "black"),
                        showarrow=False
                    )
                )
        
        # Update layout
        fig.update_layout(
            width=800,
            height=700,
            xaxis=dict(side="bottom"),
            coloraxis_colorbar=dict(title="Similarity"),
            annotations=annotations,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Save the figure
        html_file = self.output_dir / 'distinctiveness_heatmap.html'
        png_file = self.output_dir / 'distinctiveness_heatmap.png'
        
        fig.write_html(str(html_file))
        fig.update_layout(
            width=2000,
            height=1800,
            font=dict(size=18)
        )
        fig.write_image(str(png_file), scale=4)
        
        logger.info(f"Saved distinctiveness visualization to {html_file} and {png_file}")
        
        return df, fig
    
    def plot_centroid_distances(
        self,
        centroids: Dict[str, np.ndarray],
        method: str = 'mds',
        show_similarity_threshold: float = 0.0,
        min_line_width: float = 1.0,
        max_line_width: float = 5.0
    ) -> go.Figure:
        """
        Plot the distances between category centroids in 2D space with connections
        showing their pairwise similarities.
        
        Args:
            centroids: Dictionary mapping category names to their centroid embeddings
            method: Dimensionality reduction method ('mds', 'tsne', 'pca')
            show_similarity_threshold: Only show connections with similarity >= this value
            min_line_width: Minimum width for the connection lines
            max_line_width: Maximum width for the connection lines
            
        Returns:
            Plotly Figure object with the centroid distance visualization
        """
        from sklearn.manifold import MDS, TSNE
        from sklearn.decomposition import PCA
        
        if not centroids or len(centroids) < 2:
            raise ValueError("At least two categories are required for centroid distance visualization")
        
        # Get centroids and labels
        centroid_labels = list(centroids.keys())
        centroid_vectors = np.array([centroids[label] for label in centroid_labels])
        
        # Compute pairwise cosine similarities
        similarity_matrix = cosine_similarity(centroid_vectors)
        
        # Apply dimensionality reduction to centroids only
        if method == 'mds':
            # For MDS, we'll use precomputed cosine distances
            distances = 1 - similarity_matrix  # Convert similarity to distance
            reducer = MDS(
                n_components=2,
                dissimilarity='precomputed',
                random_state=42,
                normalized_stress='auto'
            )
            coords = reducer.fit_transform(distances)
        elif method == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(centroid_vectors)-1),
                metric='cosine'
            )
            coords = reducer.fit_transform(centroid_vectors)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(centroid_vectors)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        # Create figure
        fig = go.Figure()
        
        # Add connections between centroids based on similarity
        for i in range(len(centroid_labels)):
            for j in range(i + 1, len(centroid_labels)):
                similarity = similarity_matrix[i, j]
                if similarity >= show_similarity_threshold:
                    # Scale line width based on similarity
                    line_width = min_line_width + (max_line_width - min_line_width) * similarity
                    
                    fig.add_trace(go.Scatter(
                        x=[coords[i, 0], coords[j, 0]],
                        y=[coords[i, 1], coords[j, 1]],
                        mode='lines',
                        line=dict(
                            width=line_width,
                            color=f'rgba(100, 100, 100, {0.2 + 0.6 * similarity})'  # More opaque for higher similarity
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"{centroid_labels[i]} ↔ {centroid_labels[j]}: {similarity:.3f}"
                    ))
        
        # Add centroids with labels
        for i, label in enumerate(centroid_labels):
            # Get a consistent color for this category
            color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            
            # Add centroid as a star
            fig.add_trace(go.Scatter(
                x=[coords[i, 0]],
                y=[coords[i, 1]],
                mode='markers+text',
                name=label,
                text=label,
                textposition="top center",
                marker=dict(
                    size=25,  # Larger size for better visibility
                    color=color,
                    symbol='star',
                    line=dict(width=2, color='black'),
                    opacity=0.9
                ),
                hoverinfo='text',
                hovertext=f"{label} (Centroid)",
                textfont=dict(
                    size=14,
                    color='black'
                )
            ))
            
            # Add similarity values as annotations
            for j in range(i + 1, len(centroid_labels)):
                similarity = similarity_matrix[i, j]
                if similarity >= show_similarity_threshold:
                    # Position the text in the middle of the line
                    x = (coords[i, 0] + coords[j, 0]) / 2
                    y = (coords[i, 1] + coords[j, 1]) / 2
                    
                    # Adjust position to avoid overlap
                    x_offset = (coords[j, 0] - coords[i, 0]) * 0.1
                    y_offset = (coords[j, 1] - coords[i, 1]) * 0.1
                    
                    fig.add_annotation(
                        x=x + x_offset,
                        y=y + y_offset,
                        text=f"{similarity:.2f}",
                        showarrow=False,
                        font=dict(
                            size=12,
                            color='black'
                        ),
                        bgcolor="white",
                        opacity=0.8,
                        borderpad=2,
                        borderwidth=1,
                        bordercolor='gray'
                    )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>Category Distinctiveness ({method.upper()})</b><br>"
                     f"<sup>Line thickness and opacity indicate similarity between centroids</sup>",
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, family="Arial, sans-serif"),
                pad=dict(b=20, t=100)
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title='',
                showline=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title='',
                showline=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=1000,
            margin=dict(l=60, r=60, t=120, b=60),
            hovermode='closest',
            showlegend=False,
            annotations=[
                dict(
                    x=0.5,
                    y=-0.1,
                    showarrow=False,
                    text=f"Similarity threshold: {show_similarity_threshold}",
                    xref="paper",
                    yref="paper",
                    font=dict(size=12, color="gray")
                )
            ]
        )
        
        # Add colorbar for similarity
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale='Viridis',
                showscale=True,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(
                        text='Similarity',
                        side='right'
                    ),
                    thicknessmode='pixels',
                    thickness=20,
                    lenmode='pixels',
                    len=300,
                    yanchor='top',
                    y=1,
                    xanchor='right',
                    x=1.1
                )
            ),
            hoverinfo='none'
        ))
        
        # Save the figure
        html_file = self.output_dir / f'centroid_similarity_network_{method}.html'
        png_file = self.output_dir / f'centroid_similarity_network_{method}.png'
        
        fig.write_html(str(html_file))
        
        # Save high-res version
        fig.update_layout(
            width=2000,
            height=1800,
            font=dict(size=18),
            margin=dict(l=80, r=80, t=160, b=80)
        )
        fig.write_image(str(png_file), scale=4)
        
        logger.info(f"Saved centroid similarity network to {html_file} and {png_file}")
        
        # Also save the similarity matrix as a heatmap for reference
        if len(centroid_labels) > 1:
            self._save_similarity_heatmap(
                similarity_matrix,
                centroid_labels,
                method
            )
        
        return fig
    
    def plot_centroid_vectors(
        self,
        embeddings: Dict[str, np.ndarray],
        points_per_category: int = 30,
        method: str = 'mds',
        show_similarity: bool = True,
        point_size: int = 10,
        centroid_size: int = 25,
        vector_width: float = 2.0,
        fig_size: tuple = (2000, 1600)
    ) -> go.Figure:
        """
        Plot points, their centroids, and vectors between centroids to show similarity.
        
        Args:
            embeddings: Dictionary mapping category names to their centroid embeddings
            points_per_category: Number of points to generate per category
            method: Dimensionality reduction method ('mds', 'tsne', 'pca')
            show_similarity: Whether to show similarity values on vectors
            point_size: Size of the individual points
            centroid_size: Size of the centroid markers
            vector_width: Base width of the vectors
            fig_size: Figure size in pixels (width, height)
            
        Returns:
            Plotly Figure object with the visualization
        """
        from sklearn.manifold import MDS, TSNE
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        from sklearn.metrics.pairwise import cosine_distances
        
        if not embeddings or len(embeddings) < 2:
            raise ValueError("At least two categories are required for visualization")
        
        # Extract category names and prepare data
        categories = list(embeddings.keys())
        
        # Generate points around each centroid with some noise
        all_points = []
        point_labels = []
        
        # First collect all points (centroids + generated points)
        for category, centroid in embeddings.items():
            # Add the actual centroid point
            all_points.append(centroid)
            point_labels.append(f"{category} (Centroid)")
            
            # Generate points around the centroid
            for _ in range(points_per_category):
                noise = np.random.normal(0, 0.1, len(centroid))
                point = centroid + noise
                point = point / np.linalg.norm(point)  # Keep on unit sphere
                all_points.append(point)
                point_labels.append(category)
        
        all_points = np.array(all_points)
        
        # First, reduce the centroids to 2D
        if method == 'mds':
            # For MDS, we'll use cosine distance matrix of just the centroids
            centroid_embeddings = np.array([embeddings[cat] for cat in categories])
            distances = cosine_distances(centroid_embeddings)
            reducer = MDS(
                n_components=2,
                random_state=42,
                dissimilarity='precomputed',
                normalized_stress='auto'
            )
            centroid_coords = reducer.fit_transform(distances)
            
            # Project all points using the same MDS transformation
            # by finding their positions relative to centroids
            point_embeddings = all_points[n_categories:]
            point_coords = []
            
            for point in point_embeddings:
                # Find distances from point to all centroids
                point_dists = np.array([cosine_distances([point], [c])[0][0] for c in centroid_embeddings])
                
                # Find the closest centroid
                closest_idx = np.argmin(point_dists)
                closest_centroid = centroid_coords[closest_idx]
                
                # Add some noise to spread out the points
                noise = np.random.normal(0, 0.1, 2)
                point_coords.append(closest_centroid + noise)
                
            point_coords = np.array(point_coords)
            
        elif method == 'tsne':
            # For t-SNE, we'll first reduce the centroids
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(5, len(categories) - 1),  # Lower perplexity for small number of centroids
                metric='cosine',
                init='pca',
                learning_rate='auto',
                n_iter=1000
            )
            centroid_coords = reducer.fit_transform(np.array([embeddings[cat] for cat in categories]))
            
            # For points, use the same approach as MDS
            point_embeddings = all_points[n_categories:]
            point_coords = []
            centroid_embeddings = np.array([embeddings[cat] for cat in categories])
            
            for point in point_embeddings:
                # Find distances from point to all centroids
                point_dists = np.array([cosine_distances([point], [c])[0][0] for c in centroid_embeddings])
                
                # Find the closest centroid
                closest_idx = np.argmin(point_dists)
                closest_centroid = centroid_coords[closest_idx]
                
                # Add some noise to spread out the points
                noise = np.random.normal(0, 0.1, 2)
                point_coords.append(closest_centroid + noise)
                
            point_coords = np.array(point_coords)
            
        elif method == 'pca':
            # For PCA, we'll first reduce the centroids
            reducer = PCA(n_components=2, random_state=42)
            centroid_coords = reducer.fit_transform(np.array([embeddings[cat] for cat in categories]))
            
            # Transform points using the same PCA transformation
            point_coords = reducer.transform(all_points[n_categories:])
            
            # Add some noise to spread out the points
            point_coords += np.random.normal(0, 0.1, point_coords.shape)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        # Ensure we have the right number of points
        n_points_expected = len(all_points) - n_categories
        assert len(point_coords) == n_points_expected, "Mismatch in number of points"
        
        # Calculate pairwise cosine similarity between centroids using original embeddings
        centroid_embeddings = np.array([embeddings[cat] for cat in categories])
        similarity_matrix = cosine_similarity(centroid_embeddings)
        
        # Create figure with high DPI for better quality
        fig = go.Figure()
        
        # Use a consistent color scheme
        colors = px.colors.qualitative.Plotly
        
        # First, plot all the points with proper grouping
        for i, category in enumerate(categories):
            # Get indices of points for this category
            indices = [j for j, label in enumerate(point_labels[n_categories:]) if label == category]
            if not indices:
                continue
                
            # Add points with consistent color
            fig.add_trace(go.Scatter(
                x=point_coords[indices, 0],
                y=point_coords[indices, 1],
                mode='markers',
                name=f"{category} points",
                marker=dict(
                    size=point_size,
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                ),
                hoverinfo='text',
                hovertext=[f"{category} point {j+1}" for j in range(len(indices))],
                showlegend=False
            ))
        
        # Add connecting lines between all centroid pairs
        for i in range(n_categories):
            for j in range(i + 1, n_categories):
                similarity = similarity_matrix[i, j]
                # Get coordinates
                x0, y0 = centroid_coords[i]
                x1, y1 = centroid_coords[j]
                
                # Calculate line properties based on similarity
                line_width = vector_width * (0.5 + 1.5 * similarity)
                line_opacity = 0.3 + 0.7 * similarity
                
                # Add connecting line
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(
                        width=line_width,
                        color=f'rgba(100, 100, 100, {line_opacity})',
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=(
                        f"<b>{categories[i]}</b> ↔ <b>{categories[j]}</b><br>"
                        f"Similarity: {similarity:.3f}"
                    ),
                    hoverlabel=dict(
                        bgcolor='white',
                        font_size=12,
                        font_family="Arial, sans-serif"
                    )
                ))
                
                # Add similarity value in the middle of the line
                if show_similarity:
                    # Calculate midpoint with a slight offset for better visibility
                    mid_x = x0 + 0.5 * (x1 - x0)
                    mid_y = y0 + 0.5 * (y1 - y0)
                    
                    # Calculate angle for text rotation (in degrees)
                    angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
                    
                    # Add text annotation with background
                    fig.add_annotation(
                        x=mid_x,
                        y=mid_y,
                        text=f"{similarity:.2f}",
                        showarrow=False,
                        font=dict(
                            size=12,
                            color="black",
                            family="Arial, sans-serif"
                        ),
                        bgcolor="white",
                        opacity=0.9,
                        borderpad=4,
                        borderwidth=1,
                        bordercolor='gray',
                        textangle=angle,
                        xanchor='center',
                        yanchor='middle',
                        xshift=0,
                        yshift=0
                    )
        
        # Add centroids (as stars) on top of everything
        for i, (category, coords) in enumerate(zip(categories, centroid_coords)):
            fig.add_trace(go.Scatter(
                x=[coords[0]],
                y=[coords[1]],
                mode='markers+text',
                name=category,
                text=category,
                textposition="top center",
                marker=dict(
                    size=centroid_size,
                    color=colors[i % len(colors)],
                    symbol='star',
                    line=dict(width=2, color='black'),
                    opacity=1.0
                ),
                hoverinfo='text',
                hovertext=f"{category} (Centroid)",
                textfont=dict(
                    size=14,
                    color='black',
                    family='Arial, sans-serif'
                )
            ))
        
        # Calculate axis ranges with some padding
        all_x = np.concatenate([point_coords[:, 0], centroid_coords[:, 0]])
        all_y = np.concatenate([point_coords[:, 1], centroid_coords[:, 1]])
        
        x_range = [np.min(all_x), np.max(all_x)]
        y_range = [np.min(all_y), np.max(all_y)]
        
        # Add 10% padding
        x_pad = (x_range[1] - x_range[0]) * 0.1
        y_pad = (y_range[1] - y_range[0]) * 0.1
        
        x_range = [x_range[0] - x_pad, x_range[1] + x_pad]
        y_range = [y_range[0] - y_pad, y_range[1] + y_pad]
        
        # Update layout with proper scaling and aspect ratio
        fig.update_layout(
            title=dict(
                text=f"<b>Semantic Space Visualization ({method.upper()})</b>",
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=28, family="Arial, sans-serif"),
                pad=dict(b=20, t=120)
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title='',
                showline=False,
                range=x_range,
                constrain='domain',
                scaleanchor='y',
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title='',
                showline=False,
                range=y_range,
                scaleanchor='x',
                scaleratio=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=fig_size[0],
            height=fig_size[1],
            margin=dict(l=100, r=100, t=180, b=100),
            hovermode='closest',
            showlegend=False,
            autosize=False,
            dragmode='pan'
        )
        
        # Save the figure with high resolution
        os.makedirs(self.output_dir, exist_ok=True)
        html_file = self.output_dir / f'semantic_space_{method}.html'
        png_file = self.output_dir / f'semantic_space_{method}.png'
        
        # Save interactive HTML
        fig.write_html(
            str(html_file),
            include_plotlyjs='cdn',
            full_html=False,
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'responsive': True
            }
        )
        
        # Save high-res PNG
        fig.write_image(
            str(png_file),
            scale=4,  # Higher scale for better resolution
            width=fig_size[0],
            height=fig_size[1],
            engine='kaleido'
        )
        
        logger.info(f"Saved semantic space visualization to {html_file} and {png_file}")
        
        return fig
    
    def _save_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        labels: List[str],
        method: str
    ) -> None:
        """Save a heatmap of the similarity matrix."""
        import plotly.figure_factory as ff
        
        # Create annotated heatmap
        fig = ff.create_annotated_heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            zmin=0,
            zmax=1,
            showscale=True,
            annotation_text=np.around(similarity_matrix, 2),
            hoverinfo='z'
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'<b>Centroid Similarity Matrix ({method.upper()})</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis=dict(title='Category', tickangle=45),
            yaxis=dict(title='Category', autorange='reversed'),
            width=900,
            height=800,
            margin=dict(l=100, r=50, t=100, b=150),
            coloraxis_colorbar=dict(
                title='Similarity',
                thicknessmode='pixels',
                thickness=20,
                lenmode='pixels',
                len=300,
                yanchor='top',
                y=1,
                xanchor='right',
                x=1.1
            )
        )
        
        # Save the heatmap
        html_file = self.output_dir / f'centroid_similarity_matrix_{method}.html'
        png_file = self.output_dir / f'centroid_similarity_matrix_{method}.png'
        
        fig.write_html(str(html_file))
        fig.update_layout(
            width=1200,
            height=1000,
            font=dict(size=14)
        )
        fig.write_image(str(png_file), scale=4)
        
        logger.info(f"Saved centroid similarity matrix to {html_file} and {png_file}")


if __name__ == "__main__":
    """
    When run directly, this script will generate example visualizations using synthetic data.
    """
    import numpy as np
    
    # Create sample data for demonstration
    np.random.seed(42)
    
    # Generate random embeddings for 5 categories
    n_categories = 5
    embedding_dim = 50  # Reduced for better visualization
    
    # Create base vectors with some overlap
    base_vectors = np.eye(embedding_dim)[:n_categories]
    
    # Add some structure to make the example more interesting
    for i in range(1, n_categories):
        base_vectors[i] += 0.3 * base_vectors[i-1]
    
    # Normalize base vectors
    base_vectors = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)
    
    # Create categories with varying degrees of similarity
    embeddings = {}
    
    # First three categories are based on the base vectors with some noise
    for i in range(n_categories):
        # Create multiple points around each centroid
        category_points = []
        centroid = base_vectors[i]
        
        # Add some points around the centroid
        for _ in range(20):  # 20 points per category
            noise = np.random.normal(0, 0.1, embedding_dim)
            point = centroid + noise
            point = point / np.linalg.norm(point)  # Keep on unit sphere
            category_points.append(point)
        
        # Add to embeddings dictionary
        embeddings[f"Category {i+1}"] = np.array(category_points)
    
    # Calculate centroids for visualization
    centroids = {k: np.mean(v, axis=0) for k, v in embeddings.items()}
    
    # Create visualizer instance
    print("Generating vector visualizations...")
    visualizer = DistinctivenessVisualizer()
    
    # Generate visualizations for each method
    for method in ['mds', 'tsne', 'pca']:
        print(f"\nCreating vector visualization using {method.upper()}...")
        visualizer.plot_centroid_vectors(
            embeddings=centroids,
            method=method,
            points_per_category=30,  # Points to generate around each centroid
            show_similarity=True,
            point_size=10,          # Size of individual points
            centroid_size=25,        # Size of centroid markers
            vector_width=2.0,        # Base width of vectors
            fig_size=(1800, 1500)    # High resolution figure
        )
    
    # Also create the similarity matrix for reference
    print("\nCreating similarity matrix...")
    similarity_df, _ = visualizer.compute_pairwise_distinctiveness(
        centroids,
        metric='cosine'
    )
    
    print("\nPairwise similarity matrix:")
    print(similarity_df.round(2))
    
    print("\nVisualizations have been saved to the 'output' directory.")
    print("Look for files starting with 'vector_visualization_'")
