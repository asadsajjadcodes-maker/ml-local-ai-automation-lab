import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------------------------------
# Ground Truth & Model Predictions
# ---------------------------------------------------------
# Actual class labels (0 = Negative, 1 = Positive)
y_true = np.array([
    1, 1, 1, 0, 0, 0, 0, 1
])

# Predicted class labels from a binary classifier
predictions = np.array([
    1, 0, 1, 0, 1, 0, 0, 1
])


# ---------------------------------------------------------
# Confusion Matrix
# Rows = Actual [0, 1], Columns = Predicted [0, 1]
# [[TN, FP],
#  [FN, TP]]
# ---------------------------------------------------------
conf_matrix = confusion_matrix(y_true, predictions)
print("Confusion Matrix:")
print(conf_matrix)
print()


# ---------------------------------------------------------
# Evaluation Metrics (zero_division=0 safely returns 0 if
# no positive predictions or positive labels exist)
# ---------------------------------------------------------
accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions, zero_division=0)
recall = recall_score(y_true, predictions, zero_division=0)
f1 = f1_score(y_true, predictions, zero_division=0)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


