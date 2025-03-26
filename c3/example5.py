# [Visual of a high-degree polynomial hugging noisy data points but missing the general trend]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Generate a small, noisy dataset (we'll have only 8 points)
np.random.seed(0)  # For reproducibility
X_small = np.linspace(-3, 3, 8)
y_true = 0.5 * X_small**2  # True underlying trend (quadratic)
noise = np.random.normal(loc=0.0, scale=2.0, size=len(X_small))
y_small = y_true + noise  # Observed data has noise

# Reshape for scikit-learn
X_small = X_small.reshape(-1, 1)

# 2. Create high-degree polynomial features (e.g., degree=9)
poly = PolynomialFeatures(degree=9)
X_poly = poly.fit_transform(X_small)

# 3. Fit a linear model to these high-degree polynomial features
model = LinearRegression()
model.fit(X_poly, y_small)

# 4. Predict on a finer grid for plotting
X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot_pred = model.predict(X_plot_poly)

# 5. Plot the original points and the high-degree polynomial fit
plt.scatter(X_small, y_small, label="Noisy Data Points")
plt.plot(X_plot, y_plot_pred, label="Degree=9 Polynomial Fit")
plt.plot(X_plot, 0.5 * X_plot**2, label="True Underlying Trend (Quadratic)")
plt.title("High-Degree Polynomial Overfitting Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
