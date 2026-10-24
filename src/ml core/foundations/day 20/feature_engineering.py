#==================================================================================================
#             Day 20 : Feature Engineering
#==================================================================================================
import numpy as np 

# Each row is a house.
# house size , bedrooms, age of house.

X = np.array([
    [1000, 2, 10],
    [1500, 3, 5],
    [2000, 4, 15],
    [2500, 5, 20],
], dtype= float)


size = X[:, 0]
bedrooms = X[:, 1]
age = X[:, 2]


print("Original features :")
print(X)


# Feature 1 size per bedroom.
size_per_bedroom = size / bedrooms

# Feature 2: Squared age.
squared_age = age ** 2

# Feature 3: interaction between size and age.
size_age_interaction = size * age



# Combine original and engineered features.
# np.column_stack joins the original columns and engineered columns into one matrix.
X_engineered = np.column_stack([
    X,
    size_per_bedroom,
    squared_age,
    size_age_interaction,
])



print("--------------------------")
print("Engineered features:")
print(X_engineered)

print("----------------------------")
print("Feature names :")
print([
    "size",
    "bedrooms",
    "age",
    "size_per_bedroom",
    "squared_age",
    "size_age_interaction",
])


