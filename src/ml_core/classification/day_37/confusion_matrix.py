import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# ---------------------------------------------------------
# Ground Truth & Model Predictions
# ---------------------------------------------------------
# Actual class labels (0 = Negative, 1 = Positive)
y_true = np.array([
    0, 0, 0, 0, 1, 1, 1, 1
])

# Predicted class labels from a binary classifier
predictions = np.array([
    1, 0, 0, 0, 0, 0, 0, 1
])


# ---------------------------------------------------------
# Confusion Matrix
# Rows = Actual class labels [0, 1]
# Columns = Predicted class labels [0, 1]
#
# Layout:
# [[True Negative (TN),  False Positive (FP)],
#  [False Negative (FN), True Positive (TP)]]
# ---------------------------------------------------------
conf_matrix = confusion_matrix(y_true, predictions)
print("Confusion Matrix:")
print(conf_matrix)
print()


# ---------------------------------------------------------
# Matrix Decomposition / Quadrant Extraction
# ---------------------------------------------------------
# TN: Actual is 0 and Predicted is 0 (Correct rejection)
true_negative = conf_matrix[0, 0]

# FP: Actual is 0 but Predicted is 1 (Type I error / False alarm)
false_positive = conf_matrix[0, 1]

# FN: Actual is 1 but Predicted is 0 (Type II error / Missed detection)
false_negative = conf_matrix[1, 0]

# TP: Actual is 1 and Predicted is 1 (Hit / Successful detection)
true_positive = conf_matrix[1, 1]

print(f"True Negative (TN) : {true_negative}")
print(f"False Positive (FP): {false_positive}")
print(f"False Negative (FN): {false_negative}")
print(f"True Positive (TP) : {true_positive}")
print()


# ---------------------------------------------------------
# Manual Metrics Derivation from Confusion Matrix Elements
# ---------------------------------------------------------

# Accuracy: Proportion of all predictions that were correct
# Formula: (TP + TN) / (TN + FP + FN + TP)
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)
print(f"Accuracy : {accuracy:.4f}")

# Precision: Out of all positive predictions, how many were actually positive?
# Formula: TP / (TP + FP)
precision = true_positive / (true_positive + false_positive)
print(f"Precision: {precision:.4f}")

# Recall (Sensitivity): Out of all actual positives, how many did the model capture?
# Formula: TP / (TP + FN)
recall = true_positive / (true_positive + false_negative)
print(f"Recall   : {recall:.4f}")

# F1 Score: Harmonic mean of precision and recall
# Formula: 2 * (Precision * Recall) / (Precision + Recall)
manual_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
f1 = f1_score(y_true, predictions)

print(f"F1 Score (Manual)  : {manual_f1:.4f}")
print(f"F1 Score (Sklearn) : {f1:.4f}")
