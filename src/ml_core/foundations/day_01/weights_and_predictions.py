# Day one firt machine learning practice 
# Data -> model -> prediction -> loss


# Input data (X)
hours = [1, 2, 3, 4, 5]

# Target labels (y)
scores = [20, 40, 60, 80, 100]

# Model parameter
weight = 20

# Calculate predictions using the model
predictions = []
for hour in hours:
    prediction = weight * hour
    predictions.append(prediction)


print("Predictions:", predictions)


# Calculate the total error (loss)
loss = 0
for prediction, real_score in zip(predictions, scores):
    # Absolute error
    error = prediction - real_score
    loss += abs(error)

print(f"total loss is : {loss}")
