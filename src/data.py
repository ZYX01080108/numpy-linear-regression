import numpy as np


def make_linear_data(n_samples=100, true_w=5.0, true_b=-1.0, noise_std=1.0, seed=42):
    np.random.seed(seed)
    x = np.linspace(0, 10, n_samples)
    noise = np.random.normal(0, noise_std, size=n_samples)
    y = true_w * x + true_b + noise
    return x, y
