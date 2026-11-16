import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import mean_squared_error


# ==========================================
# 1. GENERATE DATASET
# ==========================================
# 20 total features, but only 5 drive the target (n_informative=5).
# This tests how well regularization handles the 15 non-informative noise features.
X, y = make_regression(
    n_samples=200,
    n_features=20,
    n_informative=5,
    noise=20,
    random_state=42
)

print("Dataset shape:", X.shape)


# ==========================================
# 2. SPLIT DATA
# ==========================================
# Split first to prevent data leakage from test distributions into scalers
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)




# ==========================================
# 3. CREATE THREE MODELS (WITH PIPELINES)
# ==========================================
# Crucial: Feature scaling (StandardScaler) is mandatory for regularized models
# so weights are penalized on an equal scale regardless of feature magnitudes.
# - Linear Regression: No penalty; fits all features (can overfit noise)
# - Ridge (L2 penalty: alpha * sum(w^2)): Shrinks weights smoothly toward zero
# - Lasso (L1 penalty: alpha * sum(|w|)): Forces uninformative weights to exact 0 (sparse feature selection)
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10)
    ),
    "Lasso Regression": make_pipeline(
        StandardScaler(),
        Lasso(alpha=1, max_iter=10000)
    )
}


# ==========================================
# 4. TRAIN AND EVALUATE MODELS
# ==========================================
for name, model in models.items():
    # Fit pipeline: fits scaler and estimator strictly on training data
    model.fit(X_train, y_train)

    # Make predictions (pipeline automatically transforms test data via fitted scaler)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    print()
    print(name)
    print(f"Training MSE: {round(train_mse, 2)}")
    print(f"Test MSE: {round(test_mse, 2)}")

# ==========================================
# 5. INSPECT MODEL WEIGHTS & SPARSITY
# ==========================================
# Compare weight magnitudes: Ridge shrinks weights; Lasso sets irrelevant weights to 0.
for name, model in models.items():
    # Get the final estimator from the pipeline
    trained_model = model.named_steps[
        list(model.named_steps.keys())[-1]
    ]
    weights = trained_model.coef_
    print()
    print(name)
    print(f"Weights: {np.round(weights, 2)}")
    # Count weights shrunk to zero (sparsity check)
    zero_weights = np.sum(np.isclose(weights, 0, atol=1e-5))
    print(f"Weights close to zero : {zero_weights}")
