import os
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px
from pathlib import Path

# Ensure the plots directory exists
PLOTS_DIR = Path(__file__).parent.parent.parent / 'plots' / 'semantic_similarity'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class SemanticCohesionVisualizer:
    def __init__(self, dimension_reducer=None, random_state=42):
        """
        Initialize the visualizer with a dimension reduction technique.
        
        Args:
            dimension_reducer: Dimensionality reduction model. If None, will use t-SNE with automatic perplexity.
                             If 'auto', will automatically choose between PCA and t-SNE based on data size.
            random_state: Random state for reproducibility
        """
        self.dimension_reducer_type = dimension_reducer
        self.random_state = random_state
        
    def _reduce_dimensions(self, combined_embeddings, all_embeddings, centroids):
        """
        Reduce dimensions of the embeddings using an appropriate method.
        
        Args:
            combined_embeddings: Combined embeddings of points and centroids
            all_embeddings: List of all point embeddings
            centroids: List of centroid embeddings
            
        Returns:
            Tuple of (points_2d, centroids_2d) - reduced dimension arrays
        """
        n_samples = len(combined_embeddings)
        
        # If a specific reducer was provided, use it
        if self.dimension_reducer_type is not None and self.dimension_reducer_type != 'auto':
            reduced_embeddings = self.dimension_reducer_type.fit_transform(combined_embeddings)
        # For very small datasets, use PCA
        elif n_samples < 5:
            from sklearn.decomposition import PCA
            print(f"Very few samples ({n_samples}), using PCA")
            reduced_embeddings = PCA(n_components=2).fit_transform(combined_embeddings)
        # For larger datasets, use t-SNE
        else:
            # Ensure perplexity is at least 1, at most 30, and less than n_samples
            perplexity = min(30, max(1, min(n_samples - 1, n_samples // 3 or 1)))
            
            print(f"Using t-SNE with perplexity={perplexity} for {n_samples} samples")
            
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=self.random_state,
                n_iter=500,
                learning_rate=200.0,
                init='pca',
                method='barnes_hut'
            )
            reduced_embeddings = reducer.fit_transform(combined_embeddings)
        
        # Split back into points and centroids
        points_2d = reduced_embeddings[:-len(centroids)]
        centroids_2d = reduced_embeddings[-len(centroids):]
        
        return points_2d, centroids_2d
    
    def plot_cohesion_3d(
        self,
        texts_dict: Dict[str, List[str]],
        embeddings_dict: Dict[str, np.ndarray],
        title: str = "Semantic Cohesion Visualization"
    ) -> go.Figure:
        """
        Create a 3D visualization of semantic cohesion.
        
        Args:
            texts_dict: Dictionary where keys are labels and values are lists of texts
            embeddings_dict: Dictionary mapping labels to their embeddings
            title: Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive 3D plot
        """
        # Prepare data for visualization
        all_embeddings = []
        all_labels = []
        all_texts = []
        centroids = []
        centroid_labels = []
        
        # Calculate centroids and prepare data
        for label, texts in texts_dict.items():
            if label not in embeddings_dict or len(embeddings_dict[label]) == 0:
                continue
                
            embeddings = embeddings_dict[label]
            centroid = np.mean(embeddings, axis=0, keepdims=True)
            
            all_embeddings.append(embeddings)
            all_labels.extend([label] * len(embeddings))
            all_texts.extend(texts)
            centroids.append(centroid[0])
            centroid_labels.append(f"{label} Centroid")
        
        if not all_embeddings:
            raise ValueError("No valid embeddings found for visualization")
            
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings + centroids)
        
        # Always create a fresh dimension reducer
        n_samples = len(combined_embeddings)
        
        # If a specific reducer was provided, use it
        if self.dimension_reducer_type is not None and self.dimension_reducer_type != 'auto':
            return self.dimension_reducer_type.fit_transform(combined_embeddings)
            
        # For very small datasets, use PCA
        if n_samples < 5:
            from sklearn.decomposition import PCA
            print(f"Very few samples ({n_samples}), using PCA")
            reducer = PCA(n_components=2)
        # For larger datasets, use t-SNE
        else:
            # Ensure perplexity is at least 1, at most 30, and less than n_samples
            perplexity = min(30, max(1, min(n_samples - 1, n_samples // 3 or 1)))
            
            # Fixed parameters for stability
            n_iter = 500
            learning_rate = 200.0
            
            print(f"Using t-SNE with perplexity={perplexity}, n_iter={n_iter} for {n_samples} samples")
            
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=self.random_state,
                n_iter=n_iter,
                learning_rate=learning_rate,
                init='pca',
                method='barnes_hut'
            )
        
        # Reduce dimensions
        points_2d, centroids_2d = self._reduce_dimensions(combined_embeddings, all_embeddings, centroids)
        
        # Create figure
        fig = go.Figure()
        
        # Add points
        for label in set(all_labels):
            mask = np.array(all_labels) == label
            fig.add_trace(go.Scatter3d(
                x=points_2d[mask, 0],
                y=points_2d[mask, 1],
                z=[0] * sum(mask),  # Flat 2D plot in 3D space
                mode='markers',
                name=label,
                text=[f"{label}<br>{text[:100]}{'...' if len(text) > 100 else ''}" 
                     for text, is_label in zip(all_texts, mask) if is_label],
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey')
                )
            ))
        
        # Add centroids
        for (x, y), label in zip(centroids_2d, centroid_labels):
            fig.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[0],
                mode='markers',
                name=label,
                marker=dict(
                    size=15,
                    symbol='diamond',
                    color='black',
                    line=dict(width=2, color='white')
                )
            ))
        
        # Add lines from points to centroids
        point_idx = 0
        for i, (label, texts) in enumerate(texts_dict.items()):
            if label not in embeddings_dict:
                continue
                
            n_points = len(texts)
            centroid_x, centroid_y = centroids_2d[i]
            
            for j in range(n_points):
                x = points_2d[point_idx + j, 0]
                y = points_2d[point_idx + j, 1]
                
                fig.add_trace(go.Scatter3d(
                    x=[x, centroid_x],
                    y=[y, centroid_y],
                    z=[0, 0],
                    mode='lines',
                    line=dict(width=1, color='gray', dash='dot'),
                    showlegend=False,
                    hoverinfo='none',
                    opacity=0.3
                ))
            
            point_idx += n_points
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.1)
                )
            ),
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save the figure
        output_file = PLOTS_DIR / f"cohesion_3d_{title.lower().replace(' ', '_')}.html"
        fig.write_html(str(output_file))
        print(f"Saved 3D cohesion plot to {output_file}")
        
        return fig
        
    def plot_centroid_contribution(
            self,
            texts_dict: Dict[str, List[str]],
            embeddings_dict: Dict[str, np.ndarray],
            title: str = "Label Cohesion Analysis"
    ) -> go.Figure:
        """
        Create a 2D visualization showing the cohesion of data points within each label.
        
        Args:
            texts_dict: Dictionary where keys are labels and values are lists of texts
            embeddings_dict: Dictionary mapping labels to their embeddings
            title: Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive 2D plot showing point cohesion within labels
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Prepare data
        all_embeddings = []
        all_labels = []
        all_texts = []
        
        # Calculate distances and prepare data
        for label, texts in texts_dict.items():
            if label not in embeddings_dict:
                continue
                
            embeddings = embeddings_dict[label]
            if len(embeddings) == 0:
                continue
                
            # Store points and their metadata
            all_embeddings.extend(embeddings)
            all_labels.extend([label] * len(embeddings))
            all_texts.extend(texts)
        
        if not all_embeddings:
            raise ValueError("No valid embeddings found for the provided data")
        
        # Convert to numpy arrays
        all_embeddings = np.array(all_embeddings)
        
        # Always use PCA for small datasets as t-SNE can be unstable
        print(f"Reducing dimensions for {len(all_embeddings)} embeddings")
        from sklearn.decomposition import PCA
        print("Using PCA for dimension reduction")
        
        # Ensure we have at least 2 dimensions
        n_components = min(2, all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else 2)
        pca = PCA(n_components=n_components)
        points_2d = pca.fit_transform(all_embeddings)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Reduced to {points_2d.shape} dimensions")
        
        # Ensure we have valid points
        if len(points_2d) == 0 or points_2d.shape[1] < 2:
            print("Warning: PCA failed to produce 2D points, using random projection")
            points_2d = np.random.rand(len(all_embeddings), 2)
        
        # Calculate average distance to centroid for each point
        avg_distances = []
        label_colors = {}
        unique_labels = list(set(all_labels))
        colors = px.colors.qualitative.Plotly
        
        for i, label in enumerate(unique_labels):
            label_indices = [j for j, l in enumerate(all_labels) if l == label]
            if not label_indices:
                continue
                
            # Get embeddings for this label
            label_embeddings = all_embeddings[label_indices]
            
            # Calculate centroid
            centroid = np.mean(label_embeddings, axis=0, keepdims=True)
            
            # Calculate cosine similarities to centroid
            similarities = cosine_similarity(label_embeddings, centroid).flatten()
            
            # Store distances for this label
            for j, idx in enumerate(label_indices):
                avg_distances.append(similarities[j])
            
            # Assign color to this label
            label_colors[label] = colors[i % len(colors)]
        
        # Normalize distances for better visualization
        if len(set(avg_distances)) > 1:  # Only normalize if there's variation
            scaler = MinMaxScaler()
            avg_distances = scaler.fit_transform(np.array(avg_distances).reshape(-1, 1)).flatten()
        
        # Create figure
        fig = go.Figure()
        
        # Calculate centroids for each label
        label_centroids = {}
        
        # First pass: calculate centroids
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(all_labels) if l == label and i < len(points_2d)]
            if not label_indices:
                continue
            label_centroids[label] = np.mean(points_2d[label_indices], axis=0)
        
        # Second pass: plot points and connections to centroids
        print(f"\nProcessing {len(unique_labels)} unique labels")
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(all_labels) if l == label and i < len(points_2d)]
            if not label_indices:
                print(f"No valid points for label '{label}', skipping")
                continue
                
            print(f"\nLabel '{label}' has {len(label_indices)} points")
            label_points = points_2d[label_indices]
            label_distances = [avg_distances[i] for i in label_indices]
            centroid = label_centroids[label]
            
            print(f"  Point range - x: [{min(label_points[:,0]):.2f}, {max(label_points[:,0]):.2f}], "
                  f"y: [{min(label_points[:,1]):.2f}, {max(label_points[:,1]):.2f}]")
            print(f"  Centroid at: {centroid}")
            
            # Add lines from points to centroid
            for point in label_points:
                fig.add_trace(go.Scatter(
                    x=[point[0], centroid[0]],
                    y=[point[1], centroid[1]],
                    mode='lines',
                    line=dict(
                    color=label_colors[label],
                    width=1.5,  # Increased from 0.5
                    dash='solid'  # Changed from 'dot' to solid lines
                ),
                hoverinfo='none',
                showlegend=False,
                opacity=0.6  # Increased from 0.3
                ))
            
            # Add the points
            hover_texts = [
                f"<b>{label}</b><br>"
                f"Text: {all_texts[i]}<br>"
                f"Cohesion: {d:.3f}<br>"
                f"Distance to centroid: {np.linalg.norm(points_2d[i] - centroid):.2f}"
                for i, d in zip(label_indices, label_distances)
            ]
            
            fig.add_trace(go.Scatter(
                x=label_points[:, 0],
                y=label_points[:, 1],
                mode='markers',
                name=label,
                text=hover_texts,
                hoverinfo='text',
                marker=dict(
                    color=label_colors[label],
                    size=12,
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                showlegend=True
            ))
            
            # Add the centroid with a star marker
            fig.add_trace(go.Scatter(
                x=[centroid[0]],
                y=[centroid[1]],
                mode='markers+text',
                name=f"{label} Centroid",
                text=[f"{label} Centroid"],
                textposition="top center",
                textfont=dict(size=12, color='black'),
                marker=dict(
                    symbol='star',
                    size=20,
                    color=label_colors[label],
                    line=dict(width=2, color='black')
                ),
                hoverinfo='text',
                hovertext=f"<b>{label} Centroid</b><br>"
                         f"Number of points: {len(label_points)}<br>"
                         f"Average cohesion: {np.mean(label_distances):.3f}",
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            hovermode='closest',
            height=700,
            width=1000,
            plot_bgcolor='white'
        )
        
        # Update layout for better visualization
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=24)  # Increased title font size
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                showticklabels=False,
                title='',
                range=[points_2d[:, 0].min() - 0.2, points_2d[:, 0].max() + 0.2]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                showticklabels=False,
                title='',
                range=[points_2d[:, 1].min() - 0.2, points_2d[:, 1].max() + 0.2]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=40, r=40, t=80, b=40),  # Increased margins
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                font=dict(size=14)  # Increased legend font size
            )
        )
        
        # Save the figure in HTML format
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        html_file = output_dir / 'label_cohesion_2d.html'
        fig.write_html(str(html_file))
        
        # Save as high-resolution PNG
        png_file = output_dir / 'label_cohesion_2d.png'
        fig.update_layout(
            width=2000,  # Increased width for higher resolution
            height=1400,  # Increased height for higher resolution
            font=dict(size=18)  # Slightly larger font for PNG
        )
        
        # Use plotly's to_image to get high-res PNG
        import plotly.io as pio
        pio.write_image(fig, str(png_file), scale=4)  # scale=4 for higher DPI
        
        print(f"\nSaved 2D cohesion plot to {html_file}")
        print(f"Saved high-resolution PNG to {png_file}")
        print(f"Figure has {len(fig.data)} traces")
        
        return fig
                
    def plot_cohesion_radar(
        self,
        cohesion_metrics: Dict[str, Dict[str, float]],
        title: str = "Semantic Cohesion Metrics"
    ) -> go.Figure:
        """
        Create a radar chart comparing cohesion metrics across different labels.
        
        Args:
            cohesion_metrics: Dictionary mapping labels to their cohesion metrics
                             (should contain 'mean_distance', 'std_distance', etc. keys)
            title: Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive radar chart
        """
        categories = ['Mean Cohesion', 'Median Cohesion', 'Cohesion STD']
        
        fig = go.Figure()
        
        for label, metrics in cohesion_metrics.items():
            fig.add_trace(go.Scatterpolar(
                r=[metrics['mean'], metrics['median'], metrics['std']],
                theta=categories,
                fill='toself',
                name=label
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=title,
            height=600
        )
        
        # Save the figure
        output_file = PLOTS_DIR / f"cohesion_radar_{title.lower().replace(' ', '_')}.html"
        fig.write_html(str(output_file))
        print(f"Saved radar plot to {output_file}")
        
        return fig

def compute_cohesion_metrics(
    texts_dict: Dict[str, List[str]],
    get_embeddings_func: callable
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Compute cohesion metrics for multiple sets of texts.
    
    Args:
        texts_dict: Dictionary where keys are labels and values are lists of texts
        get_embeddings_func: Function that takes a list of texts and returns their embeddings
        
    Returns:
        Tuple of (metrics_dict, embeddings_dict)
    """
    metrics = {}
    embeddings_dict = {}
    
    for label, texts in texts_dict.items():
        if not texts:
            continue
            
        # Get embeddings
        embeddings = get_embeddings_func(texts)
        if len(embeddings) == 0:
            continue
            
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        
        # Calculate cosine similarities to centroid
        similarities = cosine_similarity(embeddings, centroid).flatten()
        
        # Calculate statistics
        metrics[label] = {
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities))
        }
        
        # Store embeddings for visualization
        embeddings_dict[label] = embeddings
    
    return metrics, embeddings_dict



# Example usage
if __name__ == "__main__":
    # Example data with more points and variety
    texts_dict = {
        "Topic A": [
            "Introduction to machine learning concepts",
            "Deep learning models and neural networks",
            "Supervised vs unsupervised learning",
            "Training and testing machine learning models",
            "Feature engineering for ML"
        ],
        "Topic B": [
            "Data visualization techniques and tools",
            "Creating interactive plots with Python",
            "Best practices for data visualization",
            "Choosing the right chart type",
            "Color theory in data visualization"
        ],
        "Topic C": [
            "Python programming basics",
            "Object-oriented programming in Python",
            "Working with Python libraries",
            "Python for data analysis"
        ]
    }
    
    # Simple embedding function that creates more meaningful embeddings
    def get_embeddings(texts):
        # Create embeddings with some structure based on topic
        embeddings = []
        for text in texts:
            # Base embedding with some random variation
            if any(word in text.lower() for word in ['machine', 'learning', 'model']):
                base = [0.8, 0.1, 0.1]  # Topic A
            elif any(word in text.lower() for word in ['visualization', 'plot', 'chart']):
                base = [0.1, 0.8, 0.1]  # Topic B
            else:
                base = [0.1, 0.1, 0.8]  # Topic C
            
            # Add some random variation
            embedding = [x + np.random.normal(0, 0.1) for x in base]
            # Ensure positive values (simple normalization)
            embedding = [max(0.01, min(0.99, x)) for x in embedding]
            embeddings.append(embedding)
        
        # Convert to numpy array and expand to 100D with random values
        embeddings = np.array(embeddings)
        # Add some random dimensions to make it 100D
        if embeddings.shape[1] < 100:
            extra_dims = 100 - embeddings.shape[1]
            random_part = np.random.rand(len(texts), extra_dims) * 0.1
            embeddings = np.hstack([embeddings, random_part])
        
        return embeddings
    
    # Compute metrics and get embeddings
    metrics, embeddings_dict = compute_cohesion_metrics(texts_dict, get_embeddings)
    
    # Create visualizer
    visualizer = SemanticCohesionVisualizer()
    
    # Create 3D visualization
    fig_3d = visualizer.plot_cohesion_3d(texts_dict, embeddings_dict)
    
    # Create radar chart
    fig_radar = visualizer.plot_cohesion_radar(metrics)
    
    # Create 2D cohesion visualization
    fig_2d = visualizer.plot_centroid_contribution(texts_dict, embeddings_dict)
    
    # Save figures
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    fig_3d.write_html(str(output_dir / 'cohesion_3d.html'))
    fig_radar.write_html(str(output_dir / 'cohesion_radar.html'))
    fig_2d.write_html(str(output_dir / 'cohesion_2d.html'))
    
    print("Visualizations saved to 'output' directory")