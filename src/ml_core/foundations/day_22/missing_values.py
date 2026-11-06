import numpy as np
from sklearn.impute import SimpleImputer

# Each row : [house size, bedrooms, age]

X_train = np.array([
    [1000, 2, 10],
    [1500, 3, np.nan],
    [1200, np.nan, 5],
    [2000, 4, 15],
], dtype= float)


X_test = np.array([
    [1400, np.nan, 8],
    [1800, 4, np.nan],
], dtype= float)




# Count missing values in eash column
print("Missing values per column:")
print(np.isnan(X_train).sum(axis = 0))


# Create an imputer 
imputer = SimpleImputer(strategy= "median")

# Learn replacement values from training data only
X_train_filled = imputer.fit_transform(X_train)


# Resuse the same values on the test data
X_test_filled = imputer.transform(X_test)


print("Learned medians:")
print(imputer.statistics_)

print("Training data after imputation:")
print(X_train_filled)

print("Test data after imputation:")
print(X_test_filled)

print("Any missing training values left ?,")
print(np.isnan(X_train_filled).any())

print("Any MIssig test values left ?")
print(np.isnan(X_test_filled).any())