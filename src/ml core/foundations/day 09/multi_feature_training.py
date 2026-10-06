#       Day 09 : Training a multi-feature feature model 


import numpy as np 


# Features 

# column 1 = feature 1 = study hours 
# column 2 = feature 2 = marks in class test 

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
], dtype=float)


# Actual answers 
# scores in final exam

y = np.array([
    5,
    8,
    11,
    14,
    17
], dtype=float)


# Initial weights 

# We have 2 features so we need 2 weights 

weights = np.array([1.0, 1.0])

bias = 0.0

# Learning rate 
learning_rate = 0.01


# Number of training steps 
steps = 5000


# Number of examples 
n = len(X)



for step in range(steps):
    predictions = np.dot(X, weights) +bias

    error = y - predictions

    loss = np.mean(error ** 2)

    weight_gradient = (-2 / n) * np.dot(X.T, error)

    bias_gradient = (-2 / n) * np.sum(error)


    # Update the weights 

    weights = weights - learning_rate * weight_gradient

    bias = bias - learning_rate * bias_gradient


    # Show progress

    if step % 100 == 0:
        print(f"Steps : {step}")
        print(f"Loss : {loss:.4f}")
        print(f"Weights : {weights}")
        print(f"Bias : {bias}")






# Final results 

print("----------------------")
print("Final Model")
print(f"Weights : {weights}")
print(f"Bias : {bias}")



print("----------------------------")

predictions = np.dot(X, weights) + bias


print(f"Actual : {y}")
print(f"Predictions : {predictions}")
print(f"Final Mse: {np.mean((y - predictions)** 2)}")



