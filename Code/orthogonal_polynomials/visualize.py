import numpy as np
import plotly.graph_objects as go
from scipy.special import chebyt, gegenbauer, legendre, jacobi


def plot_and_save_polynomial(x, y, name, folder, params=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name))
    fig.update_layout(title=f"{name} Polynomial", xaxis_title="x", yaxis_title="P(x)")
    
    if params:
        param_str = "_".join(f"{k}{v}" for k, v in params.items())
        filename = f"{name}_{param_str}"
    else:
        filename = name
    
    fig.write_image(f"{folder}/{filename}.png", scale=3)  # High quality
    fig.write_html(f"{folder}/{filename}.html")
    print(f"Saved {filename}.png and {filename}.html in {folder}")


def plot_chebyshev(degree, folder="."):
    x = np.linspace(-1, 1, 500)
    y = chebyt(degree)(x)
    plot_and_save_polynomial(x, y, f"Chebyshev_T{degree}", folder)

def plot_chebyshev_multi(degrees, folder="."):
    x = np.linspace(-1, 1, 500)
    fig = go.Figure()
    for n in degrees:
        y = chebyt(n)(x)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"T{n}"))
    fig.update_layout(title=f"Chebyshev Polynomials: {degrees}", xaxis_title="x", yaxis_title="T_n(x)")
    filename = f"Chebyshev_multi_{'_'.join(map(str, degrees))}"
    fig.write_image(f"{folder}/{filename}.png", scale=3)
    fig.write_html(f"{folder}/{filename}.html")
    print(f"Saved {filename}.png and {filename}.html in {folder}")

def plot_gegenbauer(degree, alpha, folder="."):
    x = np.linspace(-1, 1, 500)
    y = gegenbauer(degree, alpha)(x)
    plot_and_save_polynomial(x, y, f"Gegenbauer_{degree}_a{alpha}", folder, params={"n": degree, "a": alpha})

def plot_gegenbauer_multi(degrees, alpha, folder="."):
    x = np.linspace(-1, 1, 500)
    fig = go.Figure()
    for n in degrees:
        y = gegenbauer(n, alpha)(x)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"n={n}"))
    fig.update_layout(title=f"Gegenbauer Polynomials (α={alpha}): {degrees}", xaxis_title="x", yaxis_title="C_n(x)")
    filename = f"Gegenbauer_multi_a{alpha}_{'_'.join(map(str, degrees))}"
    fig.write_image(f"{folder}/{filename}.png", scale=3)
    fig.write_html(f"{folder}/{filename}.html")
    print(f"Saved {filename}.png and {filename}.html in {folder}")

def plot_legendre(degree, folder="."):
    x = np.linspace(-1, 1, 500)
    y = legendre(degree)(x)
    plot_and_save_polynomial(x, y, f"Legendre_{degree}", folder)

def plot_legendre_multi(degrees, folder="."):
    x = np.linspace(-1, 1, 500)
    fig = go.Figure()
    for n in degrees:
        y = legendre(n)(x)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"n={n}"))
    fig.update_layout(title=f"Legendre Polynomials: {degrees}", xaxis_title="x", yaxis_title="P_n(x)")
    filename = f"Legendre_multi_{'_'.join(map(str, degrees))}"
    fig.write_image(f"{folder}/{filename}.png", scale=3)
    fig.write_html(f"{folder}/{filename}.html")
    print(f"Saved {filename}.png and {filename}.html in {folder}")

def plot_jacobi(degree, alpha, beta, folder="."):
    x = np.linspace(-1, 1, 500)
    y = jacobi(degree, alpha, beta)(x)
    plot_and_save_polynomial(x, y, f"Jacobi_{degree}_a{alpha}_b{beta}", folder, params={"n": degree, "a": alpha, "b": beta})

def plot_jacobi_multi(degrees, alpha, beta, folder="."):
    x = np.linspace(-1, 1, 500)
    fig = go.Figure()
    for n in degrees:
        y = jacobi(n, alpha, beta)(x)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"n={n}"))
    fig.update_layout(title=f"Jacobi Polynomials (α={alpha}, β={beta}): {degrees}", xaxis_title="x", yaxis_title="P_n(x)")
    filename = f"Jacobi_multi_a{alpha}_b{beta}_{'_'.join(map(str, degrees))}"
    fig.write_image(f"{folder}/{filename}.png", scale=3)
    fig.write_html(f"{folder}/{filename}.html")
    print(f"Saved {filename}.png and {filename}.html in {folder}")

def plot_jacobi(degree, alpha, beta, folder="."):
    x = np.linspace(-1, 1, 500)
    y = jacobi(degree, alpha, beta)(x)
    plot_and_save_polynomial(x, y, f"Jacobi_{degree}_a{alpha}_b{beta}", folder, params={"n": degree, "a": alpha, "b": beta})


def main():
    out_folder = "."
    # Chebyshev polynomials T0-T4
    for n in range(5):
        plot_chebyshev(n, out_folder)
    # Gegenbauer polynomials n=0-4, alpha=0.5
    for n in range(5):
        plot_gegenbauer(n, alpha=0.5, folder=out_folder)
    # Legendre polynomials n=0-4
    for n in range(5):
        plot_legendre(n, out_folder)
    # Jacobi polynomials n=0-4, alpha=0.5, beta=0.5
    for n in range(5):
        plot_jacobi(n, alpha=0.5, beta=0.5, folder=out_folder)
    # Multi-plot for Chebyshev
    plot_chebyshev_multi([0, 1, 2, 3, 4], out_folder)
    # Multi-plot for Gegenbauer
    plot_gegenbauer_multi([0, 1, 2, 3, 4, 5], alpha=0.5, folder=out_folder)
    # Multi-plot for Legendre
    plot_legendre_multi([0, 1, 2, 3, 4, 5], out_folder)
    # Multi-plot for Jacobi
    plot_jacobi_multi([0, 1, 2, 3, 4, 5], alpha=0.5, beta=0.5, folder=out_folder)

if __name__ == "__main__":
    main()
