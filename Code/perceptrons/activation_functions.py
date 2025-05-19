import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def sigmoid(x):
    """Sigmoid activation function: f(x) = 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function: f(x) = max(0, x)"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function: f(x) = max(alpha*x, x)"""
    return np.maximum(alpha * x, x)

def tanh(x):
    """Hyperbolic tangent activation function: f(x) = tanh(x)"""
    return np.tanh(x)

def elu(x, alpha=1.0):
    """Exponential Linear Unit: f(x) = x if x > 0 else alpha * (exp(x) - 1)"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x, beta=1.0):
    """Swish activation function: f(x) = x * sigmoid(beta * x)"""
    return x * sigmoid(beta * x)

def plot_activation_function(func, x_range=(-5, 5), name="Activation Function", 
                            color="royalblue", derivative=False):
    """Plot an activation function and optionally its derivative"""
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = func(x)
    
    # Calculate derivative using central differences if requested
    if derivative:
        h = 0.001  # small step for differentiation
        x_deriv = x[1:-1]  # remove the endpoints to calculate central differences
        y_deriv = (func(x_deriv + h) - func(x_deriv - h)) / (2 * h)
    
    # Create figure
    fig = go.Figure()
    
    # Add the activation function curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=name,
        line=dict(color=color, width=3),
    ))
    
    # Add the derivative if requested
    if derivative:
        fig.add_trace(go.Scatter(
            x=x_deriv,
            y=y_deriv,
            mode='lines',
            name=f"{name} Derivative",
            line=dict(color='firebrick', width=2, dash='dash'),
        ))
    
    # Customize the layout
    fig.update_layout(
        title={
            'text': f"{name} Activation Function",
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
        width=900,
        height=600,
        template="plotly_white",
        xaxis=dict(
            title=dict(
                text="Input (x)",
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
                text="Output f(x)",
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
            f"{name} Function Properties:<br>"
            f"{'Non-linear, differentiable, output range [0,1]' if name=='Sigmoid' else ''}"
            f"{'Non-linear, introduces sparsity, non-differentiable at x=0' if name=='ReLU' else ''}"
            f"{'Non-linear, differentiable, output range [-1,1]' if name=='Tanh' else ''}"
            f"{'Non-linear, differentiable, no dead neurons' if name=='ELU' else ''}"
            f"{'Addresses vanishing gradient problem' if name=='Leaky ReLU' else ''}"
            f"{'Self-gated activation function' if name=='Swish' else ''}"
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
    
    return fig

def plot_all_activation_functions():
    """Plot all activation functions in a grid for comparison"""
    # Define the activation functions to plot
    activation_functions = [
        (sigmoid, "Sigmoid", "royalblue"),
        (relu, "ReLU", "green"),
        (tanh, "Tanh", "purple"),
        (lambda x: leaky_relu(x, 0.1), "Leaky ReLU", "orange"),
        (lambda x: elu(x), "ELU", "red"),
        (lambda x: swish(x), "Swish", "teal"),
    ]
    
    # Create a 2x3 subplot grid
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[func[1] for func in activation_functions],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    # Plot each activation function
    for i, (func, name, color) in enumerate(activation_functions):
        row = i // 3 + 1
        col = i % 3 + 1
        
        x = np.linspace(-5, 5, 1000)
        y = func(x)
        
        # Add the activation function curve
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=name,
                line=dict(color=color, width=2.5),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add x-axis and y-axis line at 0
        fig.add_shape(
            type="line", line=dict(color="gray", width=1, dash="dot"),
            x0=-5, y0=0, x1=5, y1=0,
            row=row, col=col
        )
        fig.add_shape(
            type="line", line=dict(color="gray", width=1, dash="dot"),
            x0=0, y0=-1.5, x1=0, y1=1.5,
            row=row, col=col
        )
    
    # Update layout for the entire figure
    fig.update_layout(
        title={
            'text': "Neural Network Activation Functions Comparison",
            'font': {'size': 26, 'family': 'Arial', 'color': '#333333'},
            'y': 0.98,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=800,
        template="plotly_white",
    )
    
    # Update all x and y axes
    for i in range(1, 7):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1
        
        # X-axis settings
        fig.update_xaxes(
            title=dict(text="Input (x)", font=dict(family="Arial", size=14)),
            showline=True, linecolor="#333333", linewidth=1,
            showgrid=True, gridcolor="#DDDDDD",
            tickfont=dict(family="Arial", size=12),
            range=[-5, 5],
            row=row, col=col
        )
        
        # Y-axis settings
        fig.update_yaxes(
            title=dict(text="Output f(x)", font=dict(family="Arial", size=14)),
            showline=True, linecolor="#333333", linewidth=1,
            showgrid=True, gridcolor="#DDDDDD",
            tickfont=dict(family="Arial", size=12),
            range=[-1.5, 1.5],
            row=row, col=col
        )
    
    # Save the figure
    fig.write_image("activation_functions_comparison.png", scale=3)
    fig.write_html("activation_functions_comparison.html")
    
    # Show the figure
    fig.show()
    
    print("Activation functions comparison plot saved as 'activation_functions_comparison.png' and 'activation_functions_comparison.html'")

def plot_sigmoid_vs_tanh():
    """Create a detailed comparison of sigmoid and tanh activation functions"""
    # Generate input values
    x = np.linspace(-6, 6, 1000)
    
    # Calculate function values
    sigmoid_y = sigmoid(x)
    tanh_y = tanh(x)
    
    # Calculate derivatives
    h = 0.001
    x_deriv = x[1:-1]
    sigmoid_deriv = (sigmoid(x_deriv + h) - sigmoid(x_deriv - h)) / (2 * h)
    tanh_deriv = (tanh(x_deriv + h) - tanh(x_deriv - h)) / (2 * h)
    
    # Create figure
    fig = make_subplots(rows=1, cols=2, 
                         subplot_titles=["Sigmoid vs Tanh", "Derivatives"],
                         horizontal_spacing=0.1)
    
    # Plot activation functions in first subplot
    fig.add_trace(
        go.Scatter(
            x=x, y=sigmoid_y,
            mode='lines',
            name='Sigmoid',
            line=dict(color='royalblue', width=3)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x, y=tanh_y,
            mode='lines',
            name='Tanh',
            line=dict(color='green', width=3)
        ),
        row=1, col=1
    )
    
    # Plot derivatives in second subplot
    fig.add_trace(
        go.Scatter(
            x=x_deriv, y=sigmoid_deriv,
            mode='lines',
            name='Sigmoid Derivative',
            line=dict(color='royalblue', width=2, dash='dash')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_deriv, y=tanh_deriv,
            mode='lines',
            name='Tanh Derivative',
            line=dict(color='green', width=2, dash='dash')
        ),
        row=1, col=2
    )
    
    # Add horizontal and vertical lines at zero
    for col in [1, 2]:
        fig.add_shape(
            type="line", line=dict(color="gray", width=1, dash="dot"),
            x0=-6, y0=0, x1=6, y1=0,
            row=1, col=col
        )
        fig.add_shape(
            type="line", line=dict(color="gray", width=1, dash="dot"),
            x0=0, y0=-1.5, x1=0, y1=1.5,
            row=1, col=col
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Sigmoid vs Tanh Activation Functions",
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
        height=600,
        template="plotly_white"
    )
    
    # Update x and y axes for both subplots
    for col in [1, 2]:
        # X-axis settings
        fig.update_xaxes(
            title=dict(
                text="Input (x)",
                font=dict(family="Arial", size=16, color="#333333")
            ),
            showline=True, linecolor="#333333", linewidth=2,
            showgrid=True, gridcolor="#DDDDDD",
            zeroline=True, zerolinecolor="#333333", zerolinewidth=1,
            tickfont=dict(family="Arial", size=14),
            range=[-6, 6],
            row=1, col=col
        )
        
        # Y-axis settings
        fig.update_yaxes(
            title=dict(
                text="Output" if col == 1 else "Derivative",
                font=dict(family="Arial", size=16, color="#333333")
            ),
            showline=True, linecolor="#333333", linewidth=2,
            showgrid=True, gridcolor="#DDDDDD",
            zeroline=True, zerolinecolor="#333333", zerolinewidth=1,
            tickfont=dict(family="Arial", size=14),
            range=[-1.5, 1.5] if col == 1 else [-0.5, 0.5],
            row=1, col=col
        )
    
    # Add annotation for sigmoid
    fig.add_annotation(
        x=-4, y=0.7,
        text="Sigmoid:<br>Range: [0, 1]<br>Saturates at extremes",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="royalblue",
        borderwidth=1,
        font=dict(family="Arial", size=14, color="royalblue"),
        row=1, col=1
    )
    
    # Add annotation for tanh
    fig.add_annotation(
        x=4, y=-0.7,
        text="Tanh:<br>Range: [-1, 1]<br>Zero-centered",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="green",
        borderwidth=1,
        font=dict(family="Arial", size=14, color="green"),
        row=1, col=1
    )
    
    # Save the figure
    fig.write_image("sigmoid_vs_tanh.png", scale=3)
    fig.write_html("sigmoid_vs_tanh.html")
    
    # Show the figure
    fig.show()
    
    print("Sigmoid vs Tanh comparison plot saved as 'sigmoid_vs_tanh.png' and 'sigmoid_vs_tanh.html'")

def plot_relu_variants():
    """Create a comparison of ReLU and its variants"""
    # Generate input values
    x = np.linspace(-3, 3, 1000)
    
    # Calculate function values
    relu_y = relu(x)
    leaky_relu_y = leaky_relu(x, alpha=0.1)
    elu_y = elu(x, alpha=1.0)
    
    # Create figure
    fig = go.Figure()
    
    # Add ReLU
    fig.add_trace(go.Scatter(
        x=x, y=relu_y,
        mode='lines',
        name='ReLU',
        line=dict(color='royalblue', width=3),
    ))
    
    # Add Leaky ReLU
    fig.add_trace(go.Scatter(
        x=x, y=leaky_relu_y,
        mode='lines',
        name='Leaky ReLU (α=0.1)',
        line=dict(color='green', width=3),
    ))
    
    # Add ELU
    fig.add_trace(go.Scatter(
        x=x, y=elu_y,
        mode='lines',
        name='ELU (α=1.0)',
        line=dict(color='red', width=3),
    ))
    
    # Add horizontal and vertical lines at zero
    fig.add_shape(
        type="line", line=dict(color="gray", width=1, dash="dot"),
        x0=-3, y0=0, x1=3, y1=0
    )
    fig.add_shape(
        type="line", line=dict(color="gray", width=1, dash="dot"),
        x0=0, y0=-1, x1=0, y1=3
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "ReLU and Its Variants",
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
        width=900,
        height=600,
        template="plotly_white",
        xaxis=dict(
            title=dict(
                text="Input (x)",
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
            range=[-3, 3],
        ),
        yaxis=dict(
            title=dict(
                text="Output f(x)",
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
            range=[-1, 3],
        ),
    )
    
    # Add explanation box
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        text=(
            "ReLU Variants Comparison:<br>"
            "• ReLU: f(x) = max(0, x)<br>"
            "• Leaky ReLU: f(x) = max(αx, x), α=0.1<br>"
            "• ELU: f(x) = x if x>0 else α(e^x-1), α=1.0<br><br>"
            "Leaky ReLU and ELU address the<br>"
            "'dying ReLU' problem by allowing<br>"
            "negative values with non-zero gradients."
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
    
    # Save the figure
    fig.write_image("relu_variants.png", scale=3)
    fig.write_html("relu_variants.html")
    
    # Show the figure
    fig.show()
    
    print("ReLU variants plot saved as 'relu_variants.png' and 'relu_variants.html'")

def main():
    """Create all activation function visualizations"""
    # Visualize individual activation functions
    sigmoid_fig = plot_activation_function(sigmoid, name="Sigmoid", color="royalblue", derivative=True)
    sigmoid_fig.write_image("sigmoid.png", scale=3)
    sigmoid_fig.write_html("sigmoid.html")
    print("Sigmoid plot saved as 'sigmoid.png' and 'sigmoid.html'")
    
    relu_fig = plot_activation_function(relu, name="ReLU", color="green", derivative=True)
    relu_fig.write_image("relu.png", scale=3)
    relu_fig.write_html("relu.html")
    print("ReLU plot saved as 'relu.png' and 'relu.html'")
    
    # Create comparison plots
    plot_all_activation_functions()
    plot_sigmoid_vs_tanh()
    plot_relu_variants()

if __name__ == "__main__":
    main()