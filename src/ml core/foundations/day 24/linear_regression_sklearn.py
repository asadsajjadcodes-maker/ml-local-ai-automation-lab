import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Input data : house sizes in square feet.
X = np.array([
    [800],
    [1000],
    [1200],
    [1500],
    [1800],
    [2000],
    [2200],
    [2500],
    [2800],
    [3000],
], dtype= float)


# Target data: house prices in dollers.
y = np.array([
    80000,
    100000,
    120000,
    150000,
    180000,
    200000,
    220000,
    250000,
    280000,
    300000,
], dtype= float)


# Split data before training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )


# Create a model.
model = LinearRegression()


# Train the model.
model.fit(X_train, y_train)


# Predict prices for unseen test data.
predictions = model.predict(X_test)


# Inspect what the model learned.
print(f"Learned weights: {model.coef_[0]}")
print(f"learned bias: {model.intercept_}")


# Evaluate its predictions.
print(f"Actual prices: {y_test}")
print(f"Predicted prices ; {np.round(predictions, 2)}")
print(f"Test MAE: {mean_absolute_error(y_test, predictions)}")
print(f"Test MSE: {mean_squared_error(y_test, predictions)}")


# Predict the price of a new house.
new_house = np.array([
    [1750]
],dtype=float)

new_prediction = model.predict(new_house)


print(f"Prediction for a 1750 sq ft house: {new_prediction}")
