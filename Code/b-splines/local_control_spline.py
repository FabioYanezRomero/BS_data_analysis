import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import BSpline

def plot_local_control_spline():
    """
    Create a high-quality plot demonstrating local control splines using B-splines
    """
    # Define control points
    control_points = np.array([
        [0.0, 2.0],
        [1.0, 5.0],
        [3.0, 2.0],
        [5.0, 5.5],
        [7.0, 3.0],
        [9.0, 5.0],
        [10.0, 1.0]
    ])
    
    # B-spline parameters
    degree = 3
    
    # Create knot vector with appropriate multiplicity at endpoints
    # For a B-spline of degree k, we need k+1 repeated knots at endpoints
    n_control_points = len(control_points)
    knots = np.concatenate([
        [0] * (degree + 1),
        np.linspace(0, 1, n_control_points - degree + 1)[1:-1],
        [1] * (degree + 1)
    ])
    
    # Create B-spline object
    x_control = control_points[:, 0]
    y_control = control_points[:, 1]
    
    # Create the B-spline representation
    spline_x = BSpline(knots, x_control, degree)
    spline_y = BSpline(knots, y_control, degree)
    
    # Generate points on the B-spline curve
    t_fine = np.linspace(0, 1, 500)
    points_on_curve_x = spline_x(t_fine)
    points_on_curve_y = spline_y(t_fine)
    
    # Create figure
    fig = go.Figure()
    
    # Add the spline curve
    fig.add_trace(go.Scatter(
        x=points_on_curve_x,
        y=points_on_curve_y,
        mode='lines',
        name='B-Spline Curve',
        line=dict(color='royalblue', width=3),
    ))
    
    # Add control points
    fig.add_trace(go.Scatter(
        x=x_control,
        y=y_control,
        mode='markers+lines',
        name='Control Points',
        marker=dict(size=12, color='firebrick', symbol='circle'),
        line=dict(color='firebrick', width=1, dash='dash'),
    ))
    
    # Add control point labels
    for i, (x, y) in enumerate(zip(x_control, y_control)):
        fig.add_annotation(
            x=x,
            y=y,
            text=f"P{i}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            ax=20,
            ay=-30,
            font=dict(size=14),
        )
    
    # Customize layout for high-quality plotting
    fig.update_layout(
        title={
            'text': "Local Control Spline (B-Spline) Demonstration",
            'font': {'size': 24, 'family': 'Arial', 'color': '#333333'},
            'y': 0.95,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            font=dict(family="Arial", size=14),
            bordercolor="#333333",
            borderwidth=1,
        ),
        width=1200,
        height=800,
        template="plotly_white",
        xaxis=dict(
            title=dict(
                text="X-axis",
                font=dict(family="Arial", size=18, color="#333333"),
            ),
            showline=True,
            linecolor="#333333",
            linewidth=2,
            showgrid=True,
            gridcolor="#DDDDDD",
            zeroline=True,
            zerolinecolor="#333333",
            zerolinewidth=1,
            tickfont=dict(family="Arial", size=14),
        ),
        yaxis=dict(
            title=dict(
                text="Y-axis",
                font=dict(family="Arial", size=18, color="#333333"),
            ),
            showline=True,
            linecolor="#333333",
            linewidth=2,
            showgrid=True,
            gridcolor="#DDDDDD",
            zeroline=True,
            zerolinecolor="#333333",
            zerolinewidth=1,
            tickfont=dict(family="Arial", size=14),
        ),
    )
    
    # Add explanation text
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        text=(
            "Local Control Splines (B-splines):<br>"
            "Moving a control point mainly affects<br>"
            "the curve in the vicinity of that point,<br>"
            "demonstrating the 'local control' property."
        ),
        showarrow=False,
        font=dict(
            family="Arial",
            size=14,
            color="#333333"
        ),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#333333",
        borderwidth=1,
        borderpad=4
    )
    
    # Save the figure as a high-quality image
    fig.write_image("local_control_spline.png", scale=3)  # Higher scale for better resolution
    fig.write_html("local_control_spline.html")
    
    # Show the figure
    fig.show()
    
    print("Local control spline plot saved as 'local_control_spline.png' and 'local_control_spline.html'")
    
    # Add another plot to demonstrate local control by modifying a control point
    demonstrate_local_control(control_points, knots, degree)

def demonstrate_local_control(original_control_points, knots, degree):
    """Create a plot that demonstrates the local control property by modifying a control point"""
    # Make a copy of control points and modify one point (the 3rd point)
    modified_control_points = original_control_points.copy()
    modified_control_points[3, 1] += 2.0  # Move the 4th point (index 3) up by 2 units
    
    # Extract x and y coordinates
    x_original = original_control_points[:, 0]
    y_original = original_control_points[:, 1]
    x_modified = modified_control_points[:, 0]
    y_modified = modified_control_points[:, 1]
    
    # Create B-spline objects
    spline_x_original = BSpline(knots, x_original, degree)
    spline_y_original = BSpline(knots, y_original, degree)
    spline_x_modified = BSpline(knots, x_modified, degree)
    spline_y_modified = BSpline(knots, y_modified, degree)
    
    # Generate points on the curves
    t_fine = np.linspace(0, 1, 500)
    original_curve_x = spline_x_original(t_fine)
    original_curve_y = spline_y_original(t_fine)
    modified_curve_x = spline_x_modified(t_fine)
    modified_curve_y = spline_y_modified(t_fine)
    
    # Create figure
    fig = go.Figure()
    
    # Add the original spline curve
    fig.add_trace(go.Scatter(
        x=original_curve_x,
        y=original_curve_y,
        mode='lines',
        name='Original Curve',
        line=dict(color='royalblue', width=2.5),
    ))
    
    # Add the modified spline curve
    fig.add_trace(go.Scatter(
        x=modified_curve_x,
        y=modified_curve_y,
        mode='lines',
        name='Modified Curve',
        line=dict(color='green', width=2.5),
    ))
    
    # Add original control points
    fig.add_trace(go.Scatter(
        x=x_original,
        y=y_original,
        mode='markers+lines',
        name='Original Control Points',
        marker=dict(size=10, color='royalblue', symbol='circle'),
        line=dict(color='royalblue', width=1, dash='dash'),
    ))
    
    # Add modified control points
    fig.add_trace(go.Scatter(
        x=x_modified,
        y=y_modified,
        mode='markers+lines',
        name='Modified Control Points',
        marker=dict(size=10, color='green', symbol='circle'),
        line=dict(color='green', width=1, dash='dash'),
    ))
    
    # Highlight the modified point
    fig.add_trace(go.Scatter(
        x=[x_modified[3]],
        y=[y_modified[3]],
        mode='markers',
        name='Modified Point',
        marker=dict(size=14, color='red', symbol='circle', line=dict(width=2, color='black')),
        showlegend=False,
    ))
    
    # Add arrow to show the movement of the control point
    fig.add_annotation(
        x=x_modified[3],
        y=y_modified[3],
        ax=x_original[3],
        ay=y_original[3],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="red",
    )
    
    # Customize layout for high-quality plotting
    fig.update_layout(
        title={
            'text': "Local Control Property Demonstration",
            'font': {'size': 24, 'family': 'Arial', 'color': '#333333'},
            'y': 0.95,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            font=dict(family="Arial", size=14),
            bordercolor="#333333",
            borderwidth=1,
        ),
        width=1200,
        height=800,
        template="plotly_white",
        xaxis=dict(
            title=dict(
                text="X-axis",
                font=dict(family="Arial", size=18, color="#333333"),
            ),
            showline=True,
            linecolor="#333333",
            linewidth=2,
            showgrid=True,
            gridcolor="#DDDDDD",
            zeroline=True,
            zerolinecolor="#333333",
            zerolinewidth=1,
            tickfont=dict(family="Arial", size=14),
        ),
        yaxis=dict(
            title=dict(
                text="Y-axis",
                font=dict(family="Arial", size=18, color="#333333"),
            ),
            showline=True,
            linecolor="#333333",
            linewidth=2,
            showgrid=True,
            gridcolor="#DDDDDD",
            zeroline=True,
            zerolinecolor="#333333",
            zerolinewidth=1,
            tickfont=dict(family="Arial", size=14),
        ),
    )
    
    # Add explanation text
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        text=(
            "Local Control Property:<br>"
            "When control point P3 is moved up,<br>"
            "only the nearby portion of the curve<br>"
            "is affected, while distant parts<br>"
            "remain unchanged."
        ),
        showarrow=False,
        font=dict(
            family="Arial",
            size=14,
            color="#333333"
        ),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#333333",
        borderwidth=1,
        borderpad=4
    )
    
    # Save the figure as a high-quality image
    fig.write_image("local_control_demonstration.png", scale=3)  # Higher scale for better resolution
    fig.write_html("local_control_demonstration.html")
    
    # Show the figure
    fig.show()
    
    print("Local control demonstration plot saved as 'local_control_demonstration.png' and 'local_control_demonstration.html'")

if __name__ == "__main__":
    plot_local_control_spline()