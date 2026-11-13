import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error


#=====================================================================
# Create dataset
#======================================================================

X, y = make_regression(
    n_samples=300,
    n_features=15,
    n_informative=7,
    noise=25,
    random_state=42
)


print(f"Dataset shape : {X.shape}")


#==================================================================================
# Split training and test data
#==================================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")



#========================================================================================
# Define alpha values to test
#========================================================================================
# Test a wide range of regularization strengths (small to large)
alpha_values = [
    0.001,
    0.01,
    0.1,
    0.5,
    1,
    3,
    5,
    10,
    50,
    100,
    500,
    1000,
]

#==========================================================================================
# Build Ridge pipeline
#===========================================================================================
# Scale features inside the pipeline so each CV fold scales strictly on its own training data
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])


#============================================================================================
# Build ridge gridsearchcv
#============================================================================================
# "model__alpha" targets the "alpha" parameter of the "model" step in the pipeline
ridge_search = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid={
        "model__alpha": alpha_values
    },
    cv=10,                              # 10-fold cross-validation
    scoring="neg_mean_squared_error",   # Scikit-learn maximizes scores, so MSE is negated
    n_jobs=-1                           # Use all available CPU cores
)

#=======================================================================
# Train and search Ridge
#====================================================================
ridge_search.fit(X_train, y_train)

print()
print("Ridge Results")
print(f"Best alpha: {ridge_search.best_params_}")
print(f"Best CV MSE: {round(-ridge_search.best_score_, 2)}")

#===================================================================
# Build Lasso pipeline
#===================================================================
# Higher max_iter ensures Lasso converges for small alpha values
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=20000))
])


#==========================================================
# Build Lasso GridSearchCV
#=========================================================
lasso_search = GridSearchCV(
    estimator=lasso_pipeline,
    param_grid={
        "model__alpha": alpha_values
    },
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

#===========================================================
# Train and search Lasso
#=========================================================
lasso_search.fit(X_train, y_train)

print()
print("Lasso Results")
print(f"Best alpha: {lasso_search.best_params_}")
print(f"Best CV MSE: {round(-lasso_search.best_score_, 2)}")


#=====================================================
# Compare best model
#=====================================================
# GridSearchCV automatically refits best_estimator_ on 100% of X_train
best_ridge = ridge_search.best_estimator_
best_lasso = lasso_search.best_estimator_

ridge_predictions = ridge_search.predict(X_test)
lasso_predictions = lasso_search.predict(X_test)

ridge_test_mse = mean_squared_error(
    y_test,
    ridge_predictions
)

lasso_test_mse = mean_squared_error(
    y_test,
    lasso_predictions
)

print()
print("Final Test Results")
print(f"Ridge test MSE: {round(ridge_test_mse, 2)}")
print(f"Lasso test MSE: {round(lasso_test_mse, 2)}")


#========================================================
# Select best model using CV score
#=========================================================
# Always pick the final model using CV score (not test MSE) to avoid leaking test data into selection
if ridge_search.best_score_ > lasso_search.best_score_:
    print("Best model: Ridge")

else:
    print("Best model: Lasso")
