from sklearn.linear_model import LinearRegression

from src.data import make_linear_data
from src.model import mse_loss, predict, train_model


def main():
    true_w = 5.0
    true_b = -1.0
    x, y = make_linear_data(true_w=true_w, true_b=true_b)

    learning_rate = 0.001
    epochs = 10000

    manual_w, manual_b, losses = train_model(x, y, learning_rate, epochs)
    manual_pred = predict(x, manual_w, manual_b)
    manual_loss = mse_loss(manual_pred, y)

    sklearn_model = LinearRegression()
    sklearn_model.fit(x.reshape(-1, 1), y)
    sklearn_w = sklearn_model.coef_[0]
    sklearn_b = sklearn_model.intercept_
    sklearn_pred = sklearn_model.predict(x.reshape(-1, 1))
    sklearn_loss = mse_loss(sklearn_pred, y)

    print("True rule:")
    print(f"  y = {true_w:.4f}x + {true_b:.4f}")

    print("\nManual NumPy model:")
    print(f"  y = {manual_w:.4f}x + {manual_b:.4f}")
    print(f"  final loss = {manual_loss:.4f}")
    print(f"  training steps = {len(losses)}")

    print("\nsklearn LinearRegression:")
    print(f"  y = {sklearn_w:.4f}x + {sklearn_b:.4f}")
    print(f"  final loss = {sklearn_loss:.4f}")

    print("\nDifference between manual model and sklearn:")
    print(f"  w difference = {abs(manual_w - sklearn_w):.6f}")
    print(f"  b difference = {abs(manual_b - sklearn_b):.6f}")


if __name__ == "__main__":
    main()
