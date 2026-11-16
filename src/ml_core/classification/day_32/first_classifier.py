from sklearn.tree import DecisionTreeClassifier

# Training data
# [hours studied, hours slept]

X = [
    [1, 3],
    [2, 4],
    [3, 5],
    [5, 7],
    [6, 8],
    [7, 8]
]

# 0 = Fail
# 1 = Pass

y = [0, 0, 0, 1, 1, 1]

# Create a model 
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X, y)

# Make a Prediction
prediction = model.predict([[6, 7]])


print(f"Prediction : {prediction}")
