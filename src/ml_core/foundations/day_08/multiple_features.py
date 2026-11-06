import numpy as np  # Import NumPy library

# Input features for each student
study_hours = np.array([5, 7, 8])
sleep_hours = np.array([7, 8, 6])
practice_questions = np.array([20, 30, 40])

# Feature weights and bias term
weights = np.array([5, 2, 1])
bias = 10

# Compute predictions manually using element-wise operations
predictions = (
    study_hours * weights[0]
    + sleep_hours * weights[1]
    + practice_questions * weights[2]
    + bias
)

print(f"Predictions : {predictions}")

# Combine all features into a single 2D array (rows = samples, columns = features)
features = np.array([
    [5, 7, 20],
    [7, 8, 30],
    [8, 6, 40]
])

weights = np.array([5, 2, 1])
bias = 10 

# Compute predictions efficiently using matrix dot product
predictions = np.dot(features, weights) + bias

# Display results
print("-----------------------")
print("Features:")
print(features)
print()
print("Weights:")
print(weights)
print()
print("Predictions:")
print(predictions)

