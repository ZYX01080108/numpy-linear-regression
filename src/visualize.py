import matplotlib.pyplot as plt

from src.model import predict


def save_loss_curve(losses, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def save_learning_rate_comparison(all_losses, output_path):
    plt.figure(figsize=(8, 5))

    for learning_rate, losses in all_losses.items():
        plt.plot(losses, label=f"lr = {learning_rate}")

    plt.title("Learning Rate Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def save_prediction_plot(x, y, w, b, output_path):
    y_pred = predict(x, w, b)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Training data")
    plt.plot(x, y_pred, color="red", label=f"Prediction: y = {w:.2f}x + {b:.2f}")
    plt.title("Prediction Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
