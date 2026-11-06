import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)

print("Hi")
#========================================================================
#            Day 19 : Residual and Error Analysis
#=========================================================================


actual = np.array([
    100,
    120,
    140,
    160,
    180,
    200
], dtype= float)

predicted = np.array([
    105,
    118,
    135,
    155,
    172,
    185
])


# 1 . Calculate individual residuals 

residuals = actual - predicted

print(f"Actual values: {actual}")
print(f"Predicted values: {predicted}")
print(f"Residuals: {residuals}")


# Calculate sumaary matrics 
print(f"Mean residual: {np.mean(residuals)}")
print(f"MAE: {mean_absolute_error(actual, predicted)}")
print(f"MSE: {mean_squared_error(actual,  predicted)}")
print(f"RMSE: {root_mean_squared_error(actual, predicted)}")


# Plot residuals against presictions 
plt.scatter(predicted, residuals)
plt.axhline(y=0, linestyle = "--")
plt.xlabel("Predicted value")
plt.ylabel("Residual (actual - predicted)")
plt.title("Day 19 : Residual Analysis")
plt.grid(True)
plt.show()