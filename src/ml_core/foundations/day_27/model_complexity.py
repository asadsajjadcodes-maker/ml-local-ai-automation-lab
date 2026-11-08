import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Generate data 
rng = np.random.default_rng(42) # Seed so we will  get exact same numbers every single time 
X = np.linspace(-30, 30, 100).reshape(-1, 1) # Start : -30 , end : 30 number of values : 100


# Target values ; a quadratic curve
y = 2 * X[:, 0] ** 2 +3 * X[:, 0] + 5


# Add a little random noise
y = y + rng.normal(0, 2, size=len(y))

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state= 42
)


# 3. Compare different model complexities
degrees = [1, 2, 3, 5, 10, 15, 20]

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_predictions = model.predict(X_train_poly)
    test_predictions = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    print(f"Degree : {degree}")
    print(f"Training MSE ; {train_mse:.2f}")
    print(f"Test MSE : {test_mse:.2f}")
    print()

