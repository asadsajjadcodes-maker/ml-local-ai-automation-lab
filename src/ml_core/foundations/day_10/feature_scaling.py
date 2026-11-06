import numpy as np 


# Day 10 - Feature Scaling 



# Original data 
# Column 1 = house size 
# Column 2 = bedrooms 
# Column 3 = age



X = np.array([
    [1000, 2, 10],
    [1500, 3, 8],
    [2000, 4, 5],
    [2500, 5, 2]
], dtype= float)


minimum = X.min(axis = 0)
maximum = X.max(axis = 0)

X_scaled = (X - minimum) / (maximum - minimum)


y = np.array([
    10000,
    12000,
    14000,
    16000
], dtype= float)


weights = np.array([1.0, 1.0, 1.0])
bias = 0.0
learning_rate = 0.1
n = len(X)
steps = 100000

for step in range(steps):
    predictions = np.dot(X_scaled, weights) + bias

    error = y - predictions
    loss = np.mean(error ** 2)

    weight_gradint = (-2 / n) * np.dot(X_scaled.T, error)
    bias_gradient = (-2 / n) * np.sum(error)

    weights = weights - learning_rate * weight_gradint
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

predictions = np.dot(X_scaled, weights) + bias


print(f"Actual : {y}")
print(f"Predictions : {predictions}")
print(f"Final Mse: {np.mean((y - predictions)** 2)}")





print("Original data:")
print(X)
print("---------")
print("Shape:")
print(X.shape)

print("--------")
print(f"Minimum values : {minimum}")
print(f"Maximum values : {maximum}")
print("Scaled data :")
print(X_scaled)