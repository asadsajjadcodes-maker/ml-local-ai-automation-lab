import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Split dataset into training data and test data.
from sklearn.compose import ColumnTransformer # Applies different processing to different columns.
from sklearn.pipeline import Pipeline  # Connects multiple processing steps in correct order.
from sklearn.impute import SimpleImputer # Replaces the missing values.
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Standardize numeric features, Converts categories into numeric columns.
from sklearn.linear_model import LinearRegression # Learns a relationship between input and its numerical target.
from sklearn.metrics import mean_absolute_error # Measures the average size of prediction errors.


# 1. Dataset.
data = pd.DataFrame({
    "size":[800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
    "bedrooms": [2, 2, 3, 3, np.nan, 3, 4, 4, 4, 5, 5, 5],
    "age": [10, 8, 5, np.nan, 7, 4, 12, 6, 3, 9, 2, 5],
    "city": ["A", "A", "B", "A", "B", "C", "C", "B", "A", "C", "B", "A"],
    "price": [90, 100, 120, 125, 140, 150, 155, 175, 180, 200, 210, 220]
})  # .DataFrame creates a table.


# 2. Saparate input features from the target.
X = data.drop(columns= ["price"]) # Create a version of data without the price column.
y = data["price"] # Version of the data with only price column.


# 3. Split before preprocessing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42) # Unpacking 


# 4. Decide which columns are numeric and categorical.
numeric_features = ["size", "bedrooms", "age"]
categorical_features = ["city"]


# 5. Numeric preprocessing.
# Fill the missing values with the median, then scale.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 6. Categorical preprocessing.
# Fill missing categories , then one hot encode.
categorical_piprline = Pipeline([
    ("impure", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 7. Apply the correct preprocessing to the each column group.
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_piprline, categorical_features)
])


# 8. Connect the preprocessing to the model.
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# 9. Fit everything using training data only.
model.fit(X_train, y_train)


# 10. Predict and evaluate on unseen test data
predictions = model.predict(X_test)


print("Actual prices:", y_test.to_numpy()) # .to_numpy converts it to numpy array.
print("Predictions:", np.round(predictions, 2))
print("Test MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
