import numpy as np


#========================================================================
#         Day 14 : Data Leakage and Scaling
#========================================================================\

# Features
# Column 1 = house size 
# Column 2 = bedrooms
# Column 3 = house age

X = np.array([
    [800, 1, 20],
    [900, 2, 18],
    [1000, 2, 15],
    [1100, 2, 14],
    [1200, 3, 12],
    [1300, 3, 10],
    [1400, 3, 9],
    [1500, 3, 8],
    [1600, 4, 7],
    [1700, 4, 6],
    [1800, 4, 5],
    [1900, 4, 4],
    [2000, 4, 3],
    [2100, 5, 2],
    [2200, 5, 1],
    [2300, 5, 0],
    [2400, 5, 8],
    [2500, 5, 6],
    [2600, 6, 4],
    [2700, 6, 2]
], dtype=float)


# actual price 
y = np.array([
    80000,
    90000,
    100000,
    110000,
    120000,
    130000,
    140000,
    150000,
    160000,
    170000,
    180000,
    190000,
    200000,
    210000,
    220000,
    230000,
    240000,
    250000,
    260000,
    270000
], dtype=float)


# 1.  Shuffle X and y togather.
np.random.seed(42)  # Create the same shuffle everytime to prevent mismatch in dataset.
indices = np.random.permutation(len(X))  # Shuffle the indices once and then use on both X and y.

X = X[indices]  # Shuffle the X accourding to the indices.
y = y[indices]  # Shuffle the y same as X accourding to the indices.


# 2.   Split before preprocessing.
n = len(X)  # Number of examples.

train_end = int(0.70 * n)  # Training set will and at 70%.
val_end = int(0.85 * n)   # Validation set will end at 85 % (15% total).


X_train = X[:train_end]
y_train = y[:train_end]


X_val = X[train_end:val_end]
y_val = y[train_end:val_end]


X_test = X[val_end:]
y_test = y[val_end:]


# 3.   Learn scaling data from training data only.
minimum = X_train.min(axis = 0)
maximum = X_train.max(axis = 0)


# 4.  Apply the same scaling to all three sets 

X_train_scaled = (X_train  - minimum) / (maximum - minimum)

X_val_scaled = (X_val - minimum) / (maximum - minimum)

X_test_scaled = (X_test - minimum) / (maximum - minimum)




# 5.  Display and varify the pipeline.
print(f"Number of total examples : {n}")
print(f"Number of training examples : {len(X_train)}")
print(f"Number of validation examples : {len(X_val)}")
print(f"Number of test examples : {len(X_test)}")

print("------------------------------------------------")

print(f"Training minimums : {minimum}")
print(f"Training maximums : {maximum}")

print("------------------------------------------------")

print(f"Scaled training data : {X_train_scaled}")
print(f"Scaled validation data : {X_val_scaled}")
print(f"Scaled test data : {X_test_scaled}")



# Varify the split sizes.
assert len(X_train) == 14
assert len(X_val) == 3
assert len(X_test) == 3

print()
print("Pipeline checks passed!")