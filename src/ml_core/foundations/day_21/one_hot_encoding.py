import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Each row: [size, city, house_type]
X = np.array([
    [1000, "Lahore", "Apartment"],
    [1500, "Multan", "House"],
    [1200, "Lahore", "House"],
    [2000, "Layyah", "Apartment"],
], dtype=object)

# Separate numeric and categorical columns
sizes = X[:, 0].astype(float).reshape(-1, 1)
categories = X[:, 1:]  # 2D slice: grabs both 'city' and 'house_type'
# (If you only wanted 'city', use: categories = X[:, 1].reshape(-1, 1))

# Create an encoder (use sparse_output=False in modern scikit-learn)
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Learn categories and encode them
encoded_categories = encoder.fit_transform(categories)

X_final = np.column_stack([sizes, encoded_categories])

print("Categories learned:")
print(encoder.categories_)

print("Encoded categorical features:")
print(encoded_categories)

print("Final feature matrix:")
print(X_final)

print(f"Final matrix shape:  {X_final.shape}")