import numpy as np 

#========================================================
#     Day 15: Complete ML Workflow 
#========================================================





# Features
# Column 1 = house size 
# Column 2 = bedrooms
# Column 3 = house age

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


# actual price 
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




# Shuffle examples 

rng = np.random.default_rng(42)
indices = rng.permutation(len(X))


X = X[indices]
y = y[indices]


# split

n = len(X)

train_end = int(0.70 * n)
val_end = int(0.85 * n)


X_train = X[:train_end]
y_train = y[:train_end]


X_val = X[train_end:val_end]
y_val = y[train_end:val_end]


X_test = X[val_end:]
y_test = y[val_end:]


# Fit the scaler using the training data only.
minimum = X_train.min(axis = 0)
feature_range = X_train.max(axis = 0) - minimum

# Avoid division by zero for any constant feature 
feature_range[feature_range == 0] = 1

def scale(data):
    return (data - minimum) / feature_range


X_train_scaled = scale(X_train)
X_val_scaled = scale(X_val)
X_test_scaled = scale(X_test)



# Train a linear model using gradient descent.

def train_model(X_data, y_data):
    learning_rate = 0.01
    steps = 5000

    weights = np.zeros(X_data.shape[1])
    bias = 0.0
    n_samples = len(X_data)

    for step in range (steps):
        predictions = X_data @ weights + bias

        error = y_data - predictions

        weight_gradient = (-2 / n_samples) * (X_data.T @ error)
        bias_gradient = (-2 / n_samples) * np.sum(error)



        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient


    return weights , bias

def predict(X_data, weights , bias):
    return X_data @ weights + bias

def mse(actual, prediction):
    return  np.mean((actual - prediction) ** 2)
#======================================================================================

# Model A use house size only.
weights_a, bias_a = train_model(X_train_scaled[:, [0]], y_train)

val_prediction_a = predict(X_val_scaled[:, [0]], weights_a, bias_a)


val_mse_a = mse(y_val , val_prediction_a)

#==================================================================================
# Model B using all three features.


weights_b, bias_b = train_model(X_train_scaled, y_train)

val_prediction_b = predict(X_val_scaled, weights_b, bias_b)

val_mse_b = mse(y_val, val_prediction_b)


#=========================================================================

# Select the model using validation MSE

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

#=========================================================================================
# Evaluate the selected model once on the test data.
if use_all_features:
    final_test_X = X_test_scaled
else:
    final_test_X = X_test_scaled[:, [0]]


test_predictions = predict(final_test_X, best_weights, best_bias)

test_mse = mse(y_test, test_predictions)

print(f"Final test MSE : {test_mse}")
print(f"Actual test prices : {y_test}")
print(f"Predicted test prices: {test_predictions}")






