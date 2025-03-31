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
X = df[["Feature1", "Feature2"]]
y = df["Price"]

# 3. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Use the model to predict on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")
