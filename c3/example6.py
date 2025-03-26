# [Demonstrate how the same dataset can lead to different weight values with Lasso vs. Ridge]
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

# 1. Create a small synthetic dataset
np.random.seed(42)

# Features (X) with 5 columns, some of which might be correlated
X = np.random.rand(100, 5) * 10
# True coefficients (some are zero for demonstration)
true_coefs = np.array([1.5, 0.0, -2.0, 0.0, 3.0])
# Generate target with some noise
y = X.dot(true_coefs) + np.random.normal(0, 2, size=100)

# 2. Split into training and test sets (to mimic a typical ML workflow)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit a Ridge regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_coefs = ridge.coef_
ridge_intercept = ridge.intercept_

# 4. Fit a Lasso regression model
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_coefs = lasso.coef_
lasso_intercept = lasso.intercept_

# 5. Compare the coefficients
print("True coefficients:", true_coefs)
print("\nRidge coefficients:", ridge_coefs)
print("Ridge intercept:", ridge_intercept)
print("\nLasso coefficients:", lasso_coefs)
print("Lasso intercept:", lasso_intercept)

# 6. Evaluate on test data (optional, to see performance)
ridge_score = ridge.score(X_test, y_test)
lasso_score = lasso.score(X_test, y_test)
print(f"\nRidge R^2 on test data: {ridge_score:.3f}")
print(f"Lasso R^2 on test data: {lasso_score:.3f}")
