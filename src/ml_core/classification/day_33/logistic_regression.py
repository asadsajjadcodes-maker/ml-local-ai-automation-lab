from sklearn.linear_model import LogisticRegression
import numpy as np

# Training data
# Hours studied
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
],dtype= float)

# 0 = fail
# 1 = pass
y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])


# Create Logistic Regression model (learns a sigmoid probability boundary)
model = LogisticRegression()

# Train the model
model.fit(X, y)

# New student studied 3.5 hours
new_student = np.array([[3.5]])

# Predict class probabilities: [P(class 0 / Fail), P(class 1 / Pass)]
probability = model.predict_proba(new_student)

# Predict final class label (uses 0.50 probability threshold by default)
prediction = model.predict(new_student)

print(f"Probabilities [Fail, Pass]: {probability}")
print(f"Prediction: {prediction}")

# Extract probability of class 1 (Pass)
print(f"Probability of passing: {probability[0][1]:.4f}")
print("Final result:")
print("Pass" if prediction[0] == 1 else "Fail")