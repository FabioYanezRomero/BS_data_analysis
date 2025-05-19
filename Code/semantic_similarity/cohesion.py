import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px

class SemanticCohesionVisualizer:
    def __init__(self, dimension_reducer=TSNE(n_components=2, random_state=42)):
        """
        Initialize the visualizer with a dimension reduction technique.
        
        Args:
            dimension_reducer: Dimensionality reduction model (default: TSNE)
        """
        self.dimension_reducer = dimension_reducer
    
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
            
        # Combine all embeddings and reduce dimensions
        combined_embeddings = np.vstack(all_embeddings + centroids)
        reduced_embeddings = self.dimension_reducer.fit_transform(combined_embeddings)
        
        # Split back into points and centroids
        n_points = len(all_embeddings)
        points_2d = reduced_embeddings[:-len(centroids)]
        centroids_2d = reduced_embeddings[-len(centroids):]
        
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
                             (should contain 'mean', 'median', and 'std' keys)
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
    # Example data
    texts_dict = {
        "label1": ["text 1 about topic A", "another text about topic A", "more text about A"],
        "label2": ["text about topic B", "another B topic text", "more B content"],
    }
    
    # You'll need to implement or provide an embedding function
    def get_embeddings(texts):
        # This should return a numpy array of shape (n_texts, embedding_dim)
        # Replace this with your actual embedding function
        return np.random.rand(len(texts), 100)
    
    # Compute metrics and get embeddings
    metrics, embeddings_dict = compute_cohesion_metrics(texts_dict, get_embeddings)
    
    # Create visualizer
    visualizer = SemanticCohesionVisualizer()
    
    # Create 3D visualization
    fig_3d = visualizer.plot_cohesion_3d(texts_dict, embeddings_dict)
    fig_3d.show()
    
    # Create radar chart
    fig_radar = visualizer.plot_cohesion_radar(metrics)
    fig_radar.show()