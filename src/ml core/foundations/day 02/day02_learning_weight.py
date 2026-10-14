# Input data: How many hours studied
hours = 6
# Target: The score we want to predict
actual_score = 70 

# Starting with an initial guess for weight
weight = 1
# How fast we want the model to learn (step size)
learning_rate = 0.01

# Training loop: Repeat the process to get closer to the target
for i in range(70):
    # Calculate current prediction based on weight
    prediction = hours * weight
    # Find the difference between prediction and actual score
    error = actual_score - prediction

    # Update weight: adjust it based on error and learning rate
    weight = weight + learning_rate * error

# Display results
print(f"Prediction: {prediction}")
print(f"Actual: {actual_score}")
print(f"Error: {error}")
print(f"Weight: {weight}")

