import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")

def generate_data(n=100, noise=0.0, outliers=False):
    np.random.seed(42)
    X = np.linspace(0, 10, n)
    y = 2 * X + 5

    y = y + np.random.normal(0, noise, size=n)

    if outliers:
        X[:5] = np.random.uniform(0, 10, 5)
        y[:5] = np.random.uniform(20, 40, 5)

    return X, y


def compute_mse(X, y, m, b):
    y_pred = m * X + b
    return np.mean((y - y_pred) ** 2)


def gradient_descent(X, y, m, b, lr, epochs):
    m_vals, b_vals, loss_vals = [], [], []

    n = len(X)

    for _ in range(epochs):
        y_pred = m * X + b

        dm = (-2 / n) * np.sum(X * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)

        m -= lr * dm
        b -= lr * db

        m_vals.append(m)
        b_vals.append(b)
        loss_vals.append(compute_mse(X, y, m, b))

    return m_vals, b_vals, loss_vals


st.sidebar.title("Controls")

noise = st.sidebar.slider("Noise Level", 0.0, 10.0, 1.0)
m = st.sidebar.slider("Slope (m)", -5.0, 5.0, 1.0)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0)

lr = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.01)
epochs = st.sidebar.slider("Epochs", 10, 500, 100)

add_outliers = st.sidebar.checkbox("Add Outliers")

# Generate data
X, y = generate_data(noise=noise, outliers=add_outliers)

# Tabs
tabs = st.tabs([
    "Data & Fit",
    "Error (MSE)",
    "Loss Surface",
    "Gradient Descent",
    "Learning Rate",
    "Noise & Outliers"
])

with tabs[0]:
    st.title("Data & Line Fit")

    fig, ax = plt.subplots()
    ax.scatter(X, y)

    y_pred = m * X + b
    ax.plot(X, y_pred)

    st.pyplot(fig)

    st.write(f"Current Equation: y = {m:.2f}x + {b:.2f}")



with tabs[1]:
    st.title("Error Visualization (MSE)")

    y_pred = m * X + b
    mse = compute_mse(X, y, m, b)

    fig, ax = plt.subplots()

    ax.scatter(X, y)
    ax.plot(X, y_pred)

    for i in range(len(X)):
        ax.plot([X[i], X[i]], [y[i], y_pred[i]])

    st.pyplot(fig)
    st.write(f"MSE: {mse:.4f}")



with tabs[2]:
    st.title("Loss Surface")

    m_range = np.linspace(-3, 5, 50)
    b_range = np.linspace(-5, 10, 50)

    M, B = np.meshgrid(m_range, b_range)
    Z = np.zeros_like(M)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Z[i, j] = compute_mse(X, y, M[i, j], B[i, j])

    fig = go.Figure(data=[go.Surface(z=Z, x=M, y=B)])
    fig.update_layout(title="Loss Surface")

    st.plotly_chart(fig)



with tabs[3]:
    st.title("Gradient Descent")

    m_vals, b_vals, loss_vals = gradient_descent(X, y, m, b, lr, epochs)

    fig, ax = plt.subplots()
    ax.plot(loss_vals)

    st.pyplot(fig)
    st.write("Loss decreasing over iterations")



with tabs[4]:
    st.title("Learning Rate Comparison")

    lrs = [0.001, 0.01, 0.1]

    fig, ax = plt.subplots()

    for lr_val in lrs:
        _, _, losses = gradient_descent(X, y, m, b, lr_val, epochs)
        ax.plot(losses, label=f"lr={lr_val}")

    ax.legend()
    st.pyplot(fig)


# -----------------------------
# TAB 6: NOISE & OUTLIERS
# -----------------------------
with tabs[5]:
    st.title("Noise & Outliers Effect")

    X_clean, y_clean = generate_data(noise=0.0, outliers=False)
    X_noisy, y_noisy = generate_data(noise=noise, outliers=add_outliers)

    fig, ax = plt.subplots()

    ax.scatter(X_clean, y_clean, label="Clean")
    ax.scatter(X_noisy, y_noisy, label="Noisy/Outliers")

    ax.legend()
    st.pyplot(fig)