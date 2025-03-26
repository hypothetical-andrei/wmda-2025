# [Compare R², MSE, and MAE on a small dataset of predicted vs. actual house prices]
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Define small datasets for actual vs. predicted house prices
actual_prices = np.array([300000, 450000, 250000, 400000, 320000])
predicted_prices = np.array([280000, 480000, 230000, 420000, 310000])

# 2. Compute evaluation metrics
r2 = r2_score(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)

# 3. Print the results
print("R² Score:", r2)
print("MSE:", mse)
print("MAE:", mae)
