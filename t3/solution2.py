## Exercise 2 (10 minutes): Polynomial Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic non-linear dataset
np.random.seed(42)
num_samples = 30

# Single feature for clarity (e.g., 'sqft' or just X)
X = np.linspace(0, 10, num_samples).reshape(-1, 1)

# True relationship: y = 2 * X^2 - 3 * X + noise
y_true = 2 * (X**2) - 3 * X
noise = np.random.normal(0, 3, size=num_samples)
y = (y_true + noise).flatten()

# Convert to DataFrame
df = pd.DataFrame({"Feature": X.flatten(), "Target": y})

# 2. Separate features and target
X = df[["Feature"]]
y = df["Target"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Transform features to polynomial (degree=2 or 3 for illustration)
poly_degree = 2  # Try changing to 3 or 4 to see differences
poly = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 5. Create and train a Linear Regression model on the polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 6. Evaluate the model on the test set
y_pred = model.predict(X_test_poly)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Polynomial Degree:", poly_degree)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")

# 7. Optional: Plot to visualize the fit
#    Generate a smooth curve for plotting
X_range = np.linspace(0, 10, 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)

plt.scatter(X, y, label="Data")
plt.plot(X_range, y_range_pred, color="red", label="Polynomial Fit")
plt.title("Polynomial Regression Example")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()
