# [Scatter plot of a curved relationship (e.g., car’s speed vs. braking distance), demonstrating a polynomial fit]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Generate synthetic data
#    Assume a "true" relationship that is quadratic with some noise.
np.random.seed(42)  # For reproducibility
speeds = np.linspace(10, 100, 20)  # 20 different speeds (km/h)
true_braking_distance = 0.02 * speeds**2 - 1.5 * speeds + 50
noise = np.random.normal(loc=0.0, scale=20.0, size=len(speeds))
braking_distance = true_braking_distance + noise

# 2. Convert the speeds array into the correct shape and transform with PolynomialFeatures
X = speeds.reshape(-1, 1)  # Reshape for scikit-learn
poly = PolynomialFeatures(degree=2)  # We'll try a quadratic fit
X_poly = poly.fit_transform(X)

# 3. Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, braking_distance)

# 4. Generate smooth data points for plotting the fitted curve
speeds_plot = np.linspace(min(speeds), max(speeds), 100)
speeds_plot_poly = poly.transform(speeds_plot.reshape(-1, 1))
braking_distance_pred = model.predict(speeds_plot_poly)

# 5. Plot the results
plt.scatter(speeds, braking_distance, label="Data (Observed)")
plt.plot(speeds_plot, braking_distance_pred, label="Polynomial Fit")
plt.xlabel("Car's Speed (km/h)")
plt.ylabel("Braking Distance (m)")
plt.title("Polynomial Regression: Speed vs. Braking Distance")
plt.legend()
plt.show()
