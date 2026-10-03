import numpy as np 

# 1. Prepare the Data
# Input feature: number of hours studied/worked (independent variable)
hours = np.array([1, 2, 3, 4, 5])
# Actual target values: the true observed outcomes (dependent variable)
actual = np.array([15, 25, 35, 45, 55])

# 2. Initialize Model Parameters
# Weight (slope) controls the impact of the 'hours' feature on the prediction
weight = 8 
# Bias (intercept) represents the base prediction when hours = 0
bias = 3 

# 3. Forward Pass / Prediction
# Linear equation (y = mx + b) applied across arrays using NumPy broadcasting
predictions = hours * weight + bias

# 4. Calculate Loss & Error Metrics
# Error is the residual (difference between the true value and the model's prediction)
error = actual - predictions

# Loss calculated per data point using Squared Error (penalizes larger errors heavily)
loss = error ** 2 

# Mean Squared Error (MSE): The average of all the squared losses (overall model performance)
mse = np.mean(loss)

# 5. Output Results
print(f"Predictions : {predictions}")
print(f"Error : {error}")
print(f"loss: {loss}")
print(f"MSE : {mse}")
