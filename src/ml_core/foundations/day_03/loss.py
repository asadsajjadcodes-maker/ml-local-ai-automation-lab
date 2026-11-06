# Define the actual and predicted values as lists
actual = [45, 60, 100, 55, 99, 76]
prediction = [44, 57, 95, 57, 98, 76]

# Initialize a variable to keep track of the total squared error
total_squared_error = 0

# Loop through each index in the lists
for i in range(len(actual)):
    # Calculate the error between actual and predicted values at index i
    error = actual[i] - prediction[i]
    
    # Square the error
    squared_error = error ** 2
    
    # Add the squared error to the total_squared_error variable
    total_squared_error += squared_error

# Calculate the Mean Squared Error (MSE) by dividing the total squared error by the number of elements
mse = total_squared_error / len(actual)

# Print the MSE, actual values, and predicted values for verification
print(f"Mse: {mse}")
print(f"Actual: {actual}")
print(f"Prediction: {prediction}")


