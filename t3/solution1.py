## Exercise 1 (10 minutes): Baseline Linear Regression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a small synthetic dataset
#    For simplicity, let's assume we have only numeric features.
np.random.seed(42)  # For reproducibility
num_samples = 20
X = np.random.rand(num_samples, 2) * 100  # e.g., two numeric features
# True relationship (just as an example):
# price = 3.0*(feature1) + 2.0*(feature2) + some_noise
true_coeffs = np.array([3.0, 2.0])
y = X.dot(true_coeffs) + np.random.normal(0, 10, size=num_samples)  # Add noise

# Convert to a pandas DataFrame for familiarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Price"] = y

# 2. Separate features (X) and target (y)

# 3. Split the dataset into training and test sets

# 4. Create and train the Linear Regression model

# 5. Use the model to predict on the test set

# 6. Evaluate the model
