import numpy as np

# ========================================================
# Day 16: Model Evaluation — MSE, MAE, and RMSE
# ========================================================

y_true = np.array([100000, 150000, 200000, 250000, 300000], dtype=float)
y_pred_normal = np.array([105000, 145000, 205000, 245000, 295000], dtype=float)
y_pred_outlier = np.array([105000, 145000, 205000, 245000, 380000], dtype=float)

def mae(actual, predicted):
    # TODO: Return Mean Absolute Error using np.mean and np.abs
    return np.mean(np.abs(actual - predicted))
    
def mse(actual, predicted):
    # TODO: Return Mean Squared Error
    return np.mean((actual - predicted) ** 2)

def rmse(actual, predicted):
    # TODO: Return Root Mean Squared Error using np.sqrt
    return np.sqrt(mse(actual , predicted))

print("=== Normal Predictions ===")
print(f"MAE  : {mae(y_true, y_pred_normal):.2f}")
print(f"MSE  : {mse(y_true, y_pred_normal):.2f}")
print(f"RMSE : {rmse(y_true, y_pred_normal):.2f}")

print("\n=== With One Outlier Prediction ===")
print(f"MAE  : {mae(y_true, y_pred_outlier):.2f}")
print(f"MSE  : {mse(y_true, y_pred_outlier):.2f}")
print(f"RMSE : {rmse(y_true, y_pred_outlier):.2f}")
