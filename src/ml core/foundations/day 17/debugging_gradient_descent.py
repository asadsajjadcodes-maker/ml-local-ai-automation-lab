import numpy as np 

#========================================================
#     Day 17: Debugging Gradient Descent 
#========================================================

# --- DATA PREPARATION ---

# Features: Column 1 = house size, Column 2 = bedrooms, Column 3 = house age
X = np.array([
    [800, 1, 20],
    [900, 2, 18],
    [1000, 2, 15],
    [1100, 2, 14],
    [1200, 3, 12],
    [1300, 3, 10],
    [1400, 3, 9],
    [1500, 3, 8],
    [1600, 4, 7],
    [1700, 4, 6],
    [1800, 4, 5],
    [1900, 4, 4],
    [2000, 4, 3],
    [2100, 5, 2],
    [2200, 5, 1],
    [2300, 5, 0],
    [2400, 5, 8],
    [2500, 5, 6],
    [2600, 6, 4],
    [2700, 6, 2]
], dtype=float)


# Target: Actual house prices
y = np.array([
    80000,
    90000,
    100000,
    110000,
    120000,
    130000,
    140000,
    150000,
    160000,
    170000,
    180000,
    190000,
    200000,
    210000,
    220000,
    230000,
    240000,
    250000,
    260000,
    270000
], dtype=float)

# Shuffle the data randomly to mix up the samples
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]

# Define data splitting points (70% Train, 15% Validation, 15% Test)
n = len(X)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

# Split features (X) and targets (y) into Train, Validation, and Test sets
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]


# --- SCALING FUNCTIONS ---

# Scale feature arrays to a 0-1 range using min-max scaling
def scale_input(X_train, X_values):
    minimum = X_train.min(axis=0)
    range_X = (X_train.max(axis=0)) - minimum
    return (X_values - minimum) / (range_X)

# Scale target arrays to a 0-1 range using min-max scaling
def scale_output(y_train, y_values):
    minimum = y_train.min(axis=0)
    range_y = (y_train.max(axis=0)) - minimum
    return (y_values - minimum) / (range_y)

# Reverse the scaling to turn 0-1 outputs back into actual dollar prices
def unscale_output(y_train, y_scaled_values):
    minimum = y_train.min(axis=0)
    range_y = y_train.max(axis=0) - minimum
    return (y_scaled_values * range_y) + minimum


# --- APPLY SCALING ---

# Scale features based on training baseline values
X_train_scaled = scale_input(X_train, X_train)
X_val_scaled = scale_input(X_train, X_val)
X_test_scaled = scale_input(X_train, X_test)

# Scale targets based on training baseline values
y_train_scaled = scale_output(y_train, y_train)
y_val_scaled = scale_output(y_train, y_val)
y_test_scaled = scale_output(y_train, y_test)


# --- MODEL TRAINING ENGINE ---

# Optimize weights and bias using Gradient Descent optimization
def train_model(X_data, y_data):
    learning_rate = 0.01
    steps = 1000

    weights = np.zeros(X_data.shape[1])
    bias = 0.0
    n_samples = len(X_data)

    for step in range(steps):
        # Calculate current predictions and prediction errors
        predictions = X_data @ weights + bias
        error = y_data - predictions

        # Calculate slope/gradients for weights and bias
        weight_gradient = (-2 / n_samples) * (X_data.T @ error)
        bias_gradient = (-2 / n_samples) * np.sum(error)

        # Update weights and bias opposite to the gradient slope
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    return weights, bias

# Use current weights and bias to generate predictions
def predict(X_data, weights, bias):
    return X_data @ weights + bias

# Calculate Mean Squared Error to measure model accuracy
def mse(actual, prediction):
    return np.mean((actual - prediction) ** 2)


# --- EXPERIMENTATION & MODEL SELECTION ---

# Model A: Train and evaluate using ONLY the first column (house size)
weights_a, bias_a = train_model(X_train_scaled[:, [0]], y_train_scaled)
val_prediction_a = predict(X_val_scaled[:, [0]], weights_a, bias_a)
val_mse_a = mse(y_val_scaled, val_prediction_a)

# Model B: Train and evaluate using ALL THREE features
weights_b, bias_b = train_model(X_train_scaled, y_train_scaled)
val_prediction_b = predict(X_val_scaled, weights_b, bias_b)
val_mse_b = mse(y_val_scaled, val_prediction_b)

# Select the winner with the lowest validation error
print(f"Model A validation mse : {val_mse_a}")
print(f"Model B validation mse: {val_mse_b}")

if val_mse_a < val_mse_b:
    best_name = "Model A"
    best_weights = weights_a
    best_bias = bias_a
    use_all_features = False
else:
    best_name = "Model B"
    best_weights = weights_b
    best_bias = bias_b
    use_all_features = True

print("-----------------------------")
print(f"Selected Model : {best_name}")


# --- FINAL EVALUATION ---

# Isolate the correct features for the test set based on the winning model
if use_all_features:
    final_test_X = X_test_scaled
else:
    final_test_X = X_test_scaled[:, [0]]

# Generate test predictions and calculate scaled performance
test_predictions = predict(final_test_X, best_weights, best_bias)
test_mse = mse(y_test_scaled, test_predictions)

# Invert target values back to standard house pricing range
real_test_prices = unscale_output(y_train, y_test_scaled)
real_predicted_prices = unscale_output(y_train, test_predictions)

# Display final performance and raw output predictions
print(f"Final test MSE (scaled): {test_mse}")
print(f"Actual test prices ($): {real_test_prices}")
print(f"Predicted test prices ($): {real_predicted_prices}")
