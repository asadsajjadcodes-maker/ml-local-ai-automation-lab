import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# 1. Create input data (X)
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15],
    [16], [17], [18], [19], [20]
], dtype=float)

# 2. Create target values (y) 
# Underlying quadratic relationship: y = 2*x^2 + 3*x + 5
y = np.array([
    10, 19, 32, 49, 70, 
    95, 124, 157, 194, 235, 
    280, 329, 382, 439, 500, 
    565, 634, 707, 784, 865
], dtype=float)

# 3 Split the data before the training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create polynomial features.
# include_bias=False because LinearRegression fits its own intercept/bias term
poly = PolynomialFeatures(degree=3, include_bias=False)

# Fit strictly on training data to prevent data leakage, then transform test data
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 5. Train linear regression on the new features.
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 6. Predict test values 
predictions = model.predict(X_test_poly)

# 7. Inspect what the model learned.
print(f"Original input shape: {X_train.shape}")
print(f"Polynomial input shape: {X_train_poly.shape}")

print(f"Feature names : {poly.get_feature_names_out(['x'])}")

print(f"Learned weights ; {model.coef_}")
print(f"Learned bias: {model.intercept_}")

# 8. Evaluate.
print(f"Actual values : {y_test}")
print(f"Predicted values : {np.round(predictions, 2)}")

print(f"MAE : {mean_absolute_error(y_test, predictions)}")
print(f"MSE ; {mean_squared_error(y_test, predictions)}")

# 9. Predict a new input.
new_x = np.array([[12.5]],dtype=float)
new_x_poly = poly.transform(new_x)
new_prediction = model.predict(new_x_poly)
print(f"Prediction for new value x = 12.5 : \n{new_prediction[0]}")