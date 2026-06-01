import numpy as np


def predict(x, w, b):
    return w * x + b


def mse_loss(y_pred, y):
    return np.mean((y_pred - y) ** 2)


def compute_gradients(x, y, y_pred):
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))
    return dw, db


def train_model(x, y, learning_rate, epochs):
    w = 0.0
    b = 0.0
    losses = []

    for _ in range(epochs):
        y_pred = predict(x, w, b)
        loss = mse_loss(y_pred, y)
        losses.append(loss)

        dw, db = compute_gradients(x, y, y_pred)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if not np.isfinite(loss):
            break

    return w, b, losses
