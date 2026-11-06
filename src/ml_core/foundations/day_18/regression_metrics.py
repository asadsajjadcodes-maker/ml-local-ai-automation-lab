#============================================================================
#      Day 18 : Regression Metrics
#===========================================================================




import numpy as np 
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)


actual = np.array([
    100,
    200,
    300
],dtype=float)

predicted = np.array([
    110,
    180,
    510
],dtype=float)


mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = root_mean_squared_error(actual, predicted)


print(f"MAE : {mae}")
print(f"MSE : {mse}")
print(f"RMSE : {rmse}")

