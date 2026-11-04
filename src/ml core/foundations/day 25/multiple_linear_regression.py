import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Each row: [size in sq ft, bedrooms, age in years]
X = np.array([
    [800,  2, 20],
    [900,  2, 15],
    [1000, 2, 10],
    [1100, 3, 12],
    [1200, 3, 8],
    [1300, 3, 5],
    [1400, 3, 15],
    [1500, 4, 10],
    [1600, 4, 8],
    [1700, 4, 5],
    [1800, 4, 12],
    [1900, 5, 10],
    [2000, 5, 7],
    [2100, 5, 5],
    [2200, 5, 3],
    [2300, 5, 8],
], dtype=float)

# Target: house price
# Underlying relationship:
# price = 100*size + 10000*bedrooms - 500*age + 20000
y = (
    100 * X[:, 0]
    + 10000 * X[:, 1]
    - 500 * X[:, 2]
    + 20000
)

# 1. Split before training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.25, random_state=42
)

# 2. Create and train the model.
model = LinearRegression()
model.fit(X_train, y_train)

# 3. See what the model learned.
print(f"Learned weights: {model.coef_}")
print(f"Learned bias: {model.intercept_}")


# 4. Predict test prices.
predictions = model.predict(X_test)


print(f"Actual prices: {y_test}")
print(f"Predicted prices: {np.round(predictions, 2)}")


# 5. Evaluate.
print(f"Test MAE; {mean_absolute_error(y_test, predictions)}")
print(f"Test MSE: {mean_squared_error(y_test, predictions)}")


# 6. Predict a new house.
new_house = np.array([
    [1500, 3, 10]
], dtype=float)
new_prediction = model.predict(new_house)

print(f"New house features:{new_house[0]}")
print(f"Predicted price for new house: {round(new_prediction[0], 2)}")

