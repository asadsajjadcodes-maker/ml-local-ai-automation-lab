import numpy as np

# ---------------------------------------------------------
# Training Dataset: Hours studied vs. Pass/Fail
# ---------------------------------------------------------
# Feature: Hours studied
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
], dtype=float)

# Target labels: 0 = Fail, 1 = Pass
y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
], dtype=float)


# ---------------------------------------------------------
# Activation Function: Sigmoid
# Squashes any real-valued number into a probability [0, 1]
# ---------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------
# Model Parameters & Hyperparameters
# ---------------------------------------------------------
weight = np.zeros(1)     # Start with weight = 0
bias = 0.0               # Start with bias = 0.0
learning_rate = 0.01     # Step size for parameter updates
steps = 2000             # Total training iterations (epochs)
number_of_examples = len(X)


# ---------------------------------------------------------
# Training Loop: Gradient Descent Optimization
# ---------------------------------------------------------
for step in range(steps):
    # 1. Forward Pass: compute linear combination z = Xw + b
    z = X @ weight + bias
    # Pass z through sigmoid to get predicted probabilities
    probabilities = sigmoid(z)

    # 2. Compute prediction error (predicted probability - actual label)
    error = probabilities - y

    # 3. Calculate gradients (derivatives of loss with respect to weight & bias)
    weight_gradient = (X.T @ error) / number_of_examples
    bias_gradient = np.mean(error)

    # 4. Parameter Update: move opposite to the gradient slope
    weight = weight - learning_rate * weight_gradient
    bias = bias - learning_rate * bias_gradient

    # 5. Compute Binary Cross-Entropy (Log Loss)
    # Clip probabilities slightly away from 0 and 1 to prevent log(0) numerical error
    clipped = np.clip(probabilities, 1e-10, 1 - 1e-10)
    loss = -np.mean(
        y * np.log(clipped) + (1 - y) * np.log(1 - clipped)
    )

    # 6. Progress Logging every 200 steps
    if step % 200 == 0:
        print(f"Step: {step:4d}, Loss: {loss:.4f}")


