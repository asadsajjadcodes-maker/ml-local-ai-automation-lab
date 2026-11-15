import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error
)

#===============================================================
# 1. Create a synthetic house dataset
#===============================================================
# Use a fixed seed for reproducible synthetic data
rng = np.random.default_rng(42)
n_samples = 500

size = rng.integers(600, 3500, size=n_samples)
bedrooms = rng.integers(1, 7, size=n_samples)
age = rng.integers(0, 50, size=n_samples)

cities = rng.choice(
    ["Muscat", "Lahore", "Islamabad", "Karachi", "Multan"],
    size=n_samples
)       # Randomly picks one of the cities for each of the 500 samples

# Location premium added to base house price
city_effect = {
    "Muscat": 70000,
    "Lahore": 25000,
    "Islamabad": 40000,
    "Karachi": 35000,
    "Multan": 15000
}

# True price formula + random Gaussian noise
price = (
    size * 100
    + bedrooms * 12000
    - age * 700
    + np.array([city_effect[city] for city in cities])
    + rng.normal(0, 20000, size=n_samples)
)
df = pd.DataFrame({
    "size": size,
    "bedrooms": bedrooms,
    "age": age,
    "city": cities,
    "price": price
})

#=================================================================
# 2. Introduce some missing values
#=================================================================
# Randomly set 15 values per feature column to NaN to simulate messy real-world data
for column in ["size", "bedrooms", "age", "city"]:
    missing_indices = rng.choice(
        df.index,
        size=15,
        replace=False
    )
    df.loc[missing_indices, column] = np.nan

print("Dataset preview:")
print(df.head())
print()
print("Missing values:")
print(df.isnull().sum())
print(f"Dataset shape: {df.shape}")

#==========================================================
# 3. Separate features and target
#==========================================================
X = df.drop(columns=["price"])
y = df["price"]


#================================================================
# 4. Train / test split
#=================================================================
# Split before any imputation or scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print()
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


#================================================================
# 5. Define numerical and categorical features
#================================================================
numeric_features = [
    "size",
    "bedrooms",
    "age"
]

categorical_features = ["city"]


#================================================================
# 6. Numerical preprocessing
#================================================================
# Fill missing numbers with the median, then standardize to zero mean and unit variance
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


#=====================================================
# 7. Categorical preprocessing
#=====================================================
# Fill missing categories with the most frequent city, then one-hot encode
# handle_unknown="ignore" prevents crashes if a new city appears at test/inference time
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


#=======================================================
# 8. Combine preprocessing
#========================================================
# Apply numeric and categorical pipelines to their respective columns
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


#====================================================
# 9. Create complete model pipeline
#====================================================
# Chain preprocessing and Ridge regression into a single leak-free pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge())
])

#===========================================================================
# 10. Hyperparameter tuning
#===========================================================================
# Tune Ridge regularization strength (alpha) using 5-fold cross-validation
alpha_values = [
    0.01,
    0.1,
    1,
    5,
    10,
    100,
    500
]
search = GridSearchCV(
    estimator=model_pipeline,
    param_grid={
        "model__alpha": alpha_values
    },
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

#===================================================
# 11. Create a baseline model
#=====================================================
# Simple baseline that always predicts the median training price
baseline = DummyRegressor(strategy="median")

baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)
baseline_mse = mean_squared_error(y_test, baseline_predictions)

#=============================================================================
# 12. Train and search
#==============================================================================
search.fit(X_train, y_train)
best_model = search.best_estimator_
print()
print("Best model results")
print("Best parameters:")
print(search.best_params_)
print("Best CV MSE:")
print(round(-search.best_score_, 2))

#=========================================================================
# 13. Final test evaluation
#========================================================================
# Evaluate the best pipeline on the held-out test set and compare against baseline
predictions = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print()
print("Final model evaluation")
print(f"Baseline test MSE: {round(baseline_mse, 2)}")
print(f"Model test MAE : {round(mae, 2)}")
print(f"Model test MSE : {round(mse, 2)}")
print(f"Model test RMSE: {round(rmse, 2)}")

#======================================================================
# 14. Residual analysis
#=====================================================================
# Residuals = actual price - predicted price (near-zero mean indicates low bias)
residuals = y_test.to_numpy() - predictions
print()
print("Residual analysis")
print(f"Mean residual: {round(np.mean(residuals), 2)}")
print(f"Largest positive residual: {round(np.max(residuals), 2)}")
print(f"Largest negative residual: {round(np.min(residuals), 2)}")


# ==========================================
# 15. PREDICT A NEW HOUSE
# ==========================================
# Pass raw DataFrame; pipeline handles imputation, scaling, and encoding automatically
new_house = pd.DataFrame({
    "size": [1800],
    "bedrooms": [3],
    "age": [8],
    "city": ["Muscat"]
})

new_prediction = best_model.predict(new_house)

print("\nNEW HOUSE PREDICTION")

print("Predicted house price:", round(new_prediction[0], 2))

# ==========================================
# 16. TEST AN UNSEEN CITY
# ==========================================
# Verify OneHotEncoder(handle_unknown="ignore") safely handles an unseen city
unseen_house = pd.DataFrame({
    "size": [1500],
    "bedrooms": [3],
    "age": [5],
    "city": ["Unknown City"]
})

unseen_prediction = best_model.predict(unseen_house)

print("\nUNSEEN CITY TEST")

print("Predicted price:", round(unseen_prediction[0], 2))



