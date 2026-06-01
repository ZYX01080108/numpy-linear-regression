from pathlib import Path

from src.data import make_linear_data
from src.model import train_model
from src.visualize import save_loss_curve, save_prediction_plot


def main():
    # 1. Create simple training data.
    true_w = 5.0
    true_b = -1.0
    x, y = make_linear_data(true_w=true_w, true_b=true_b)

    # 2. Choose training settings.
    learning_rate = 0.001
    epochs = 10000

    # 3. Train the model with gradient descent.
    w, b, losses = train_model(x, y, learning_rate, epochs)
    print(f"learning_rate = {learning_rate}")
    print(f"final loss = {losses[-1]:.4f}")
    print(f"Learned rule: y = {w:.4f}x + {b:.4f}")

    # 4. Show the true rule.
    print("\nTraining finished.")
    print(f"True rule:    y = {true_w:.4f}x + {true_b:.4f}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    save_loss_curve(losses, output_dir / "loss_curve.png")
    save_prediction_plot(x, y, w, b, output_dir / "prediction_plot.png")
    print("\nSaved plots:")
    print(f"- {output_dir / 'loss_curve.png'}")
    print(f"- {output_dir / 'prediction_plot.png'}")


if __name__ == "__main__":
    main()
