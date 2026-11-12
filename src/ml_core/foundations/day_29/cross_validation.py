import numpy as np
from sklearn.datasets import make_regression  # for generating an example dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


#=====================================================================
# 1. CREATE AN EXAMPLE DATASET
#=====================================================================
# 200 samples, 5 features (4 informative, 1 noise)
X, y = make_regression(
    n_samples=200,
    n_features=5,
    n_informative=4,
    noise=20,
    random_state=42
)
print(f"Shape of dataset: {X.shape}")


#=======================================================================
# 2. SPLIT DATA INTO TRAINING AND HELD-OUT TEST SET
#=======================================================================
# Reserve 20% strictly as an unseen test set for final verification.
# All model comparison / hyperparameter tuning is done via CV on X_train.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.20, 
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


#====================================================================
# 3. DEFINE CANDIDATE MODELS IN PIPELINES
#====================================================================
# Pipelines prevent data leakage: StandardScaler will be fit strictly
# on each fold's training slice during cross-validation.
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Ridge alpha=1": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1)
    ),
    "Ridge alpha=10": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10)
    )
}


#======================================================================
# 4. PERFORM 5-FOLD CROSS-VALIDATION
#======================================================================
# K-Fold CV partitions training data into 5 folds, training on 4 and
# validating on 1 iteratively. This measures performance stability across folds.
results = {}
for name, model in models.items():
    # Scikit-learn scoring maximizes utility, so loss metrics are negative
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    # Invert negative MSE back to positive MSE
    fold_mse = -scores
    average_mse = np.mean(fold_mse)
    std_mse = np.std(fold_mse)  # Measures variance/stability across folds

    results[name] = average_mse

    print()
    print(name)
    print(f"Fold MSEs: {np.round(fold_mse, 2)}")
    print(f"Average CV MSE: {round(average_mse, 2)}")
    print(f"Standard Deviation: {round(std_mse, 2)}")


#===========================================================================
# 5. SELECT THE BEST MODEL VIA LOWEST CV MSE
#===========================================================================
best_name = min(results, key=results.get)
best_model = models[best_name]

print()
print(f"Best model: {best_name}")
print(f"Best CV MSE: {round(results[best_name], 2)}")


#=====================================================================================
# 6. RETRAIN SELECTED MODEL ON ALL TRAINING DATA
#======================================================================================
# After selecting the optimal hyperparameter via CV, retrain the model on the full
# training set (100% of X_train) so it learns from all available training observations.
best_model.fit(X_train, y_train)


#==================================================================================
# 7. FINAL TEST EVALUATION
#==================================================================================
# Unbiased evaluation on the held-out test data (never seen during CV or training)
test_predictions = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)

print()
print(f"Final test MSE: {round(test_mse, 2)}")
