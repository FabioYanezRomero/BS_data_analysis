import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

def linear_transformation(x, w, b):
    """Compute a linear transformation with weight and bias
    
    Args:
        x: Input values (can be scalar or array)
        w: Weight
        b: Bias
        
    Returns:
        The linearly transformed value w*x + b
    """
    return w * x + b

def step_function(x):
    """Simple step activation function for perceptrons
    
    Args:
        x: Input value(s)
        
    Returns:
        1 if x >= 0, otherwise 0
    """
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    """Sigmoid activation function
    
    Args:
        x: Input value(s)
        
    Returns:
        1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Hyperbolic tangent activation function
    
    Args:
        x: Input value(s)
        
    Returns:
        tanh(x)
    """
    return np.tanh(x)

def relu(x):
    """ReLU (Rectified Linear Unit) activation function
    
    Args:
        x: Input value(s)
        
    Returns:
        max(0, x)
    """
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    """Leaky ReLU activation function
    
    Args:
        x: Input value(s)
        alpha: Slope for negative values (default 0.1)
        
    Returns:
        x if x > 0, alpha*x otherwise
    """
    return np.where(x > 0, x, alpha * x)

def get_activation_function(name):
    """Get activation function by name
    
    Args:
        name: Name of the activation function
        
    Returns:
        The corresponding activation function
    """
    activation_functions = {
        'step': step_function,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu,
        'leaky_relu': leaky_relu
    }
    
    if name not in activation_functions:
        raise ValueError(f"Unknown activation function: {name}. Available options: {', '.join(activation_functions.keys())}")
    
    return activation_functions[name]

def plot_perceptron_with_parameters(weight=1.0, bias=0.0, activation='sigmoid'):
    """
    Create a visualization showing the components of a perceptron with specific parameters:
    1. Linear transformation with user-specified weight and bias
    2. User-specified activation function
    3. Complete perceptron transformation (combination of both)
    
    Args:
        weight: Weight for the linear transformation (default 1.0)
        bias: Bias for the linear transformation (default 0.0)
        activation: Name of the activation function (default 'sigmoid')
    """
    # Get the specified activation function
    activation_func = get_activation_function(activation)
    
    # Create a subplot with 3 rows, 1 column
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f"Linear Transformation: y = {weight}x + {bias}",
            f"Non-Linear Activation: {activation}",
            f"Complete Transformation: {activation}({weight}x + {bias})"
        ],
        vertical_spacing=0.15,
    )
    
    # Define domain for x values
    x = np.linspace(-5, 5, 500)
    
    # Plot 1: Linear Transformation
    y_linear = linear_transformation(x, weight, bias)
    fig.add_trace(
        go.Scatter(
            x=x, y=y_linear,
            mode='lines',
            name=f"w={weight}, b={bias}",
            line=dict(color='royalblue', width=3),
        ),
        row=1, col=1
    )
    
    # Plot 2: Activation Function
    x_activation = np.linspace(-5, 5, 1000)
    y_activation = activation_func(x_activation)
    
    fig.add_trace(
        go.Scatter(
            x=x_activation, y=y_activation,
            mode='lines',
            name=activation,
            line=dict(color='green', width=3),
        ),
        row=2, col=1
    )
    
    # Plot 3: Complete Perceptron (Linear + Activation)
    perceptron_output = activation_func(linear_transformation(x, weight, bias))
    
    fig.add_trace(
        go.Scatter(
            x=x, y=perceptron_output,
            mode='lines',
            name=f"{activation}({weight}x + {bias})",
            line=dict(color='red', width=3),
        ),
        row=3, col=1
    )
    
    # Update layout for the entire figure
    fig.update_layout(
        title={
            'text': f"Perceptron with {activation.capitalize()} Activation",
            'font': {'size': 26, 'family': 'Arial', 'color': '#333333'},
            'y': 0.98,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=1200,
        template="plotly_white",
        legend=dict(
            font=dict(family="Arial", size=14),
            bordercolor="#333333",
            borderwidth=1,
        ),
    )
    
    # Update axes for all subplots
    for i in range(1, 4):
        # Add horizontal line at y=0
        fig.add_shape(
            type="line", line=dict(dash="dot", color="gray", width=1),
            x0=-5, y0=0, x1=5, y1=0,
            row=i, col=1
        )
        
        # Add vertical line at x=0
        y_max = 5 if i==1 else (1.5 if activation in ['sigmoid', 'tanh', 'step'] else 5)
        y_min = -5 if i==1 else (-1.5 if activation == 'tanh' else (-0.5 if activation == 'leaky_relu' else 0))
        
        fig.add_shape(
            type="line", line=dict(dash="dot", color="gray", width=1),
            x0=0, y0=y_min, x1=0, y1=y_max,
            row=i, col=1
        )
        
        # X-axis settings
        fig.update_xaxes(
            title=dict(
                text="Input (x)",
                font=dict(family="Arial", size=16, color="#333333")
            ),
            showline=True, linecolor="#333333", linewidth=2,
            showgrid=True, gridcolor="#DDDDDD",
            zeroline=False,
            tickfont=dict(family="Arial", size=14),
            range=[-5, 5],
            row=i, col=1
        )
        
        # Y-axis settings (different for each plot)
        y_titles = ["Linear Output", f"{activation.capitalize()} Output", "Perceptron Output"]
        
        # Set y-range based on activation type
        if i == 1:  # Linear transformation
            y_range = [-5, 5]
        elif i == 2:  # Activation function
            if activation == 'relu':
                y_range = [-0.5, 5]
            elif activation == 'leaky_relu':
                y_range = [-2, 5]
            elif activation == 'tanh':
                y_range = [-1.2, 1.2]
            elif activation in ['sigmoid', 'step']:
                y_range = [-0.2, 1.2]
            else:
                y_range = [-2, 2]
        else:  # Combined output
            if activation == 'relu':
                y_range = [-0.5, 5]
            elif activation == 'leaky_relu':
                y_range = [-2, 5]
            elif activation == 'tanh':
                y_range = [-1.2, 1.2]
            elif activation in ['sigmoid', 'step']:
                y_range = [-0.2, 1.2]
            else:
                y_range = [-2, 2]
        
        fig.update_yaxes(
            title=dict(
                text=y_titles[i-1],
                font=dict(family="Arial", size=16, color="#333333")
            ),
            showline=True, linecolor="#333333", linewidth=2,
            showgrid=True, gridcolor="#DDDDDD",
            zeroline=False,
            tickfont=dict(family="Arial", size=14),
            range=y_range,
            row=i, col=1
        )
    
    # Add explanatory annotations
    # Row 1: Linear transformation
    fig.add_annotation(
        xref="x", yref="y",
        x=3.5, y=-4,
        text=(
            "Linear Transformation:<br>"
            f"y = {weight}x + {bias}<br><br>"
            "• w controls the slope<br>"
            "• b shifts the line up/down"
        ),
        showarrow=False,
        font=dict(family="Arial", size=14, color="#333333"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#333333",
        borderwidth=1,
        borderpad=4,
        row=1, col=1
    )
    
    # Row 2: Activation function
    activation_explanations = {
        'step': "Step Function:<br>σ(x) = 1 if x ≥ 0, else 0<br><br>• Binary threshold activation<br>• Not differentiable at x=0",
        'sigmoid': "Sigmoid Function:<br>σ(x) = 1/(1+e^(-x))<br><br>• Output range: (0, 1)<br>• Smooth, differentiable",
        'tanh': "Hyperbolic Tangent:<br>tanh(x)<br><br>• Output range: (-1, 1)<br>• Zero-centered",
        'relu': "ReLU Function:<br>f(x) = max(0, x)<br><br>• Linear for x > 0<br>• Helps with vanishing gradient",
        'leaky_relu': "Leaky ReLU:<br>f(x) = x if x > 0, else 0.1x<br><br>• Small slope for negative inputs<br>• Prevents 'dead neurons'"
    }
    
    fig.add_annotation(
        xref="x", yref="y",
        x=3.5, y=0.5 if activation in ['sigmoid', 'step'] else 0,
        text=activation_explanations.get(activation, f"{activation.capitalize()} Activation"),
        showarrow=False,
        font=dict(family="Arial", size=14, color="#333333"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#333333",
        borderwidth=1,
        borderpad=4,
        row=2, col=1
    )
    
    # Row 3: Complete perceptron
    fig.add_annotation(
        xref="x", yref="y",
        x=3.5, y=0.5 if activation in ['sigmoid', 'step'] else 0,
        text=(
            "Complete Transformation:<br>"
            f"y = {activation}({weight}x + {bias})<br><br>"
            "• Combines linear and non-linear<br>&nbsp;&nbsp;components"
        ),
        showarrow=False,
        font=dict(family="Arial", size=14, color="#333333"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#333333",
        borderwidth=1,
        borderpad=4,
        row=3, col=1
    )
    
    # Generate filenames based on parameters
    base_filename = f"perceptron_{activation}_w{weight}_b{bias}".replace(".", "p")
    png_filename = f"{base_filename}.png"
    html_filename = f"{base_filename}.html"
    
    # Create the plots directory if it doesn't exist
    import os
    os.makedirs("plots/perceptrons", exist_ok=True)
    
    # Save the figure
    png_path = os.path.join("plots/perceptrons", png_filename)
    html_path = os.path.join("plots/perceptrons", html_filename)
    
    fig.write_image(png_path, scale=3)
    fig.write_html(html_path)
    
    # Show the figure
    fig.show()
    
    print(f"Perceptron visualization saved as '{png_path}' and '{html_path}'")
    
    return fig

def main():
    """
    Create perceptron visualization based on command-line parameters
    """
    parser = argparse.ArgumentParser(description='Visualize perceptron with specific parameters')
    parser.add_argument('--weight', type=float, default=1.0, help='Weight for linear transformation (default: 1.0)')
    parser.add_argument('--bias', type=float, default=0.0, help='Bias for linear transformation (default: 0.0)')
    parser.add_argument('--activation', type=str, default='sigmoid', 
                        choices=['step', 'sigmoid', 'tanh', 'relu', 'leaky_relu'],
                        help='Activation function (default: sigmoid)')
    
    args = parser.parse_args()
    
    # Plot perceptron with specified parameters
    plot_perceptron_with_parameters(
        weight=args.weight,
        bias=args.bias,
        activation=args.activation
    )

if __name__ == "__main__":
    main()