# day 11 : Train and  test split 


import numpy as np 


# our data 
# X = hours studied 
# y = exam score

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
], dtype= float)

y = np.array([
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100
], dtype= float)



# Shuffle the data 

#-------------------------


np.random.seed(42) # for reproducibility
indices = np.random.permutation(len(X)) # shuffle the indices of the data
X = X[indices] # shuffle the data using the shuffled indices
y = y[indices] # shuffle the labels using the shuffled indices


# Split the data into training and testing sets
#------------------------------------------


split_index = int(0.8 * len(X))

X_train = X[:split_index]
y_train = y[:split_index]


X_test = X[split_index:]
y_test = y[split_index:]

print ("training data:")
print(X_train)

print("Testing data:")
print(X_test)





weight = np.array([1.0])
bias = 0.0
learning_rate = 0.01
n = len(X_train)

steps = 1000

#========================================================
#        Training 
#========================================================

for step in range(steps):
    predictions = np.dot(X_train, weight) + bias

    error = y_train - predictions
    loss = np.mean(error ** 2)

    weight_gradient = (-2 / n) * np.dot(X_train.T, error)
    bias_gradient = (-2 / n) * np.sum(error)

    weight = weight - learning_rate * weight_gradient
    bias = bias - learning_rate * bias_gradient


#=======================================
# Test the unseen data 
#=======================================


test_predictions = np.dot(X_test, weight) + bias
test_error = y_test - test_predictions
test_loss = np.mean(test_error ** 2)

print("----------------------")
print(f"Learned weight : {weight}")
print(f"learned bias : {bias}")

print("------------------------------")
print("Actual test values :")
print(y_test)
print()
print("predicted test values:")
print(test_predictions)

print()
print(f"test mse: {test_loss}")

    