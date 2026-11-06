# Training data: input features (e.g., hours)
hours = [1, 2, 3, 4, 5]
# Target values: actual scores corresponding to each hour
actual_scores = [15, 25, 35, 45, 55]

# Initial model parameters (starting guesses for weight and bias)
weight = 1
bias = 0

# Hyperparameter: learning rate controlling the step size of parameter updates
learning_rate = 0.01

# Main training loop running for 3500 iterations (epochs)
for step in range(3500):

    # Initialize accumulators for loss and gradients across the dataset
    total_loss = 0
    total_weight_gradient = 0
    total_bias_gradient = 0

    # Loop through each data point to compute errors and gradients
    for i in range(len(hours)):

        # Linear model prediction: y = mx + b
        prediction = hours[i] * weight + bias

        # Difference between the actual score and our model's prediction
        error = actual_scores[i] - prediction

        # Squared error loss for the current data point
        loss = error ** 2

        # Partial derivative of the loss with respect to weight
        weight_gradient = -2 * hours[i] * error
        # Partial derivative of the loss with respect to bias
        bias_gradient = -2 * error

        # Accumulate totals to average later
        total_loss += loss
        total_weight_gradient += weight_gradient
        total_bias_gradient += bias_gradient

    # Compute mean squared error (MSE) across all data points
    average_loss = total_loss / len(hours)

    # Compute average gradient for the weight parameter
    average_weight_gradient = (
        total_weight_gradient / len(hours)
    )

    # Compute average gradient for the bias parameter
    average_bias_gradient = (
        total_bias_gradient / len(hours)
    )

    # Update weight using gradient descent
    weight = weight - learning_rate * average_weight_gradient

    # Update bias using gradient descent
    bias = bias - learning_rate * average_bias_gradient

    # Print training progress and metrics for the current step
    print(f"Step: {step}")
    print(f"Loss: {average_loss:.4f}")
    print(f"weight: {weight:.4f}")
    print(f"bias: {bias:.4f}")
    print("--------------------------")

# Make a prediction for a new input value (6.2 hours) using trained parameters
hour = 6.2
prediction = hour * weight + bias 
print(f"hour: {hour}")
print(f"prediction: {prediction:.4f}")

# Make a prediction for a new input value (6 hours)
hour = 6
prediction = hour * weight + bias 
print(f"hour: {hour}")
print(f"prediction: {prediction:.4f}")

# Make a prediction for a new input value (9.5 hours)
hour = 9.5
prediction = hour * weight + bias 
print(f"hour: {hour}")
print(f"prediction: {prediction:.4f}")