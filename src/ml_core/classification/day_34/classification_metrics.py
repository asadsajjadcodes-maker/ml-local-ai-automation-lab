from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Ground truth (actual target labels: 0 or 1)
y_true = [
    1, 0, 1, 1, 0,
    0, 1, 0, 1, 0
]

# Model predictions
y_pred = [
    1, 0, 0, 1, 0,
    1, 1, 0, 1, 0
]


# Confusion Matrix:
# Rows = Actual classes [0, 1]
# Columns = Predicted classes [0, 1]
# [[True Negative (TN),  False Positive (FP)],
#  [False Negative (FN), True Positive (TP)]]
matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("Confusion Matrix:")
print(matrix)
print()


# Evaluation Metrics:
# Accuracy : Overall percentage of correct predictions (TP + TN) / Total
accuracy = accuracy_score(y_true, y_pred)

# Precision: When model predicts positive (1), how often is it right? TP / (TP + FP)
precision = precision_score(y_true, y_pred)

# Recall   : Out of all actual positives (1), how many were captured? TP / (TP + FN)
recall = recall_score(y_true, y_pred)

# F1 Score : Harmonic mean of precision and recall: 2 * (P * R) / (P + R)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print()


# Detailed Classification Report (Per-class precision, recall, f1-score & support)
print("Classification Report:")
print(classification_report(y_true, y_pred))

