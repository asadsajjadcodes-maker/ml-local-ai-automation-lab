# Training data: hours studied and actual exam scores.
hours = [1, 2, 3, 4, 5, 6]
actual_scores = [10, 20, 30, 40, 50, 60]

# Start with a simple weight.
weight = 1

# Controls how big each weight update will be.
learning_rate = 0.01 


for step in range(100):

    # Reset totals for this training step.
    total_loss = 0
    total_gradient = 0

    # Go through every example in our dataset.
    for i in range(len(hours)):

        # Make prediction using the current weight.
        prediction = hours[i] * weight

        # Difference b/w actual and prediction.
        error = actual_scores[i] - prediction

        # Squared error tells us how large the mistake is.
        loss = error ** 2

        # Calculate how the weight should change.
        gradient = -2 * hours[i] * error

        # Add this example's loss and gradient to the totals.
        total_loss += loss
        total_gradient += gradient

    # Average the results for the all training examples.
    average_gradient = total_gradient / len(hours)
    average_loss = total_loss / len(hours)

    # Update the weight using gradient descent.
    weight = weight - learning_rate * average_gradient


    print(
        f"Step: {step}",
        f"Loss: {average_loss:.4f}",
        f"Weight: {weight:.4f}"
    )
