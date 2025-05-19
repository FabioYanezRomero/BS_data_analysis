import numpy as np
import plotly.graph_objs as go
from scipy.special import legendre

# Define the domain
x = np.linspace(-1, 1, 400)

# Define a polynomial as a linear combination of Legendre polynomials
# Original coefficients for degrees 0 to 5
coeffs = [0.3, -0.7, 1.0, 0.2, -0.2, 0.5]

# Evaluate the original polynomial
poly = sum(c * legendre(i)(x) for i, c in enumerate(coeffs))

# Modify one coefficient (e.g., degree 3)
coeffs_mod = coeffs.copy()
coeffs_mod[3] += 1.0  # Increase degree 3 coefficient
poly_mod = sum(c * legendre(i)(x) for i, c in enumerate(coeffs_mod))

# Plot with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=poly, mode='lines', name='Original Polynomial (all coeffs: [0.3, -0.7, 1.0, 0.2, -0.2, 0.5])', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x, y=poly_mod, mode='lines', name='Modified Polynomial (degree 3 coeff = 1.2)', line=dict(color='red', dash='dash')))

fig.update_layout(
    title="Global Control Property of Orthogonal Polynomials (Legendre Basis)",
    xaxis_title="x",
    yaxis_title="Polynomial Value",
    legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
)

fig.write_html("/usrvol/plots/orthogonal_polynomials/global_control_orthogonal_polynomials.html")
fig.write_image(
    "/usrvol/plots/orthogonal_polynomials/global_control_orthogonal_polynomials.png",
    format="png",
    width=1920,
    height=1080,
    scale=3
)
fig.show()
