# [An example: Using a perceptron to classify points above or below a line]
import numpy as np

# Generate simple training data
X = np.array([
    [0.2, 0.6],
    [0.5, 0.4],
    [0.8, 0.9],
    [0.3, 0.2],
    [0.1, 0.7],
    [0.6, 0.5]
])

# Labels: 1 if y > x, else 0
y = np.array([
    1, 0, 1, 0, 1, 0
])

# Add bias term (add 1 as third input)
X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

# Initialize weights randomly
weights = np.random.randn(3)

# Learning rate and epochs
lr = 0.1
epochs = 20

# Activation function: step function
def step(x):
    return 1 if x >= 0 else 0

# Training loop
for epoch in range(epochs):
    total_errors = 0
    for xi, target in zip(X_bias, y):
        z = np.dot(xi, weights)
        pred = step(z)
        error = target - pred
        weights += lr * error * xi
        total_errors += abs(error)
    print(f"Epoch {epoch+1}: Errors = {total_errors}")

# Test prediction
def predict(x_point):
    x_with_bias = np.append(x_point, 1)
    return step(np.dot(x_with_bias, weights))

# Try on a new point
test_point = [0.4, 0.7]
print(f"Point {test_point} classified as: {predict(test_point)}")
