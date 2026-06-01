# NumPy Linear Regression

This project implements simple linear regression from scratch with NumPy. It uses gradient descent to learn the parameters of a line:

```text
y = wx + b
```

The goal is to understand the core training loop behind machine learning:

```text
predict -> compute loss -> compute gradients -> update parameters
```

## Motivation

This is my first AI learning project. Instead of calling a machine learning library such as scikit-learn, I manually implement the main parts of linear regression:

- data generation
- prediction
- mean squared error loss
- gradient computation
- gradient descent
- visualization with Matplotlib

## Method

The synthetic data is generated from:

```text
y = 5x - 1 + noise
```

The model starts with:

```text
w = 0
b = 0
```

For each training step, the model computes:

```text
y_pred = wx + b
loss = mean((y_pred - y)^2)
dw = mean(2 * (y_pred - y) * x)
db = mean(2 * (y_pred - y))
```

Then it updates the parameters:

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

## Project Structure

```text
.
├── train.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── model.py
│   └── visualize.py
└── outputs/
    ├── loss_curve.png
    └── prediction_plot.png
```

## How to Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run training:

```powershell
python train.py
```

## Results

With:

```text
learning_rate = 0.001
epochs = 10000
```

The model learns approximately:

```text
y = 5.0118x - 1.1597
```

The true rule is:

```text
y = 5.0000x - 1.0000
```

The result is not exactly the same because the training data includes random noise.

## Visualizations

Loss curve:

![Loss Curve](outputs/loss_curve.png)

Prediction plot:

![Prediction Plot](outputs/prediction_plot.png)

## What I Learned

- Linear regression can be trained with gradient descent.
- Mean squared error measures the average prediction error.
- Gradients tell the model how to update `w` and `b`.
- A learning rate that is too small makes training slow.
- A learning rate that is too large can make training unstable.
- NumPy makes it possible to compute over many data points at once.

## Next Steps

- Compare this implementation with scikit-learn's `LinearRegression`.
- Extend the project to multiple linear regression.
- Add command-line arguments for learning rate and epochs.
- Write unit tests for loss and gradient computation.
