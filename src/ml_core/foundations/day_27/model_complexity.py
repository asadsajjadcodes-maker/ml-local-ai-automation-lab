import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 1. Generate data 
rng = np.random.default_rng(42) # Seed so we will get exact same numbers every single time 
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
results = []
trained_model = {}

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

    results.append((degree, train_mse, test_mse))
    trained_model[degree] = model
    print(f"Degree : {degree}")
    print(f"Training MSE ; {train_mse:.2f}")
    print(f"Test MSE : {test_mse:.2f}")
    print()

#=====================================================
#  Prepare Graph
#=====================================================
# Generate 300 evenly spaced numbers
# These are used to draw smooth prediction curves
X_curve = np.linspace(-30, 30, 300).reshape(-1, 1)


# Create an empty graph
plt.figure(figsize=(12, 7))

# Draw original data as dots
plt.scatter(X[:, 0], y, label="Original data", color="black", alpha=0.6)

# Draw model predictions
for degree in degrees:
    # Get the already trained model
    model = trained_model[degree]

    # FIX: Re-create the transformer to match the model's expected shape
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_curve_poly = poly.fit_transform(X_curve) # Or just poly.transform if fitted, fit_transform works fine here too

    # Predict values for the graph using the transformed features
    curve_predictions = model.predict(X_curve_poly) 

    # Draw prediction curve
    plt.plot(X_curve[:, 0], curve_predictions, label= f"Degree {degree}")



# Finish the graph
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Underfitting vs Overfitting")

# Optional bonus: Limit y-view so overfitting oscillations don't ruin the chart scale
plt.ylim(-100, 2000)

plt.legend()
plt.grid(True)

plt.show()


# Print final results 
print("Final comparison")
for degree, train_mse, test_mse in results:
    print(
        f"Degree : {degree:2} | "
        f"Train mse: {train_mse:10.2f} | "
        f"Test mse: {test_mse:10.2f}"
    )
