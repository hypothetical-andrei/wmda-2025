# [An example showing how loss decreases during training with gradient descent]
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 1, 100)
true_w, true_b = 2, 1
y = true_w * X + true_b + np.random.randn(*X.shape) * 0.1  # Add noise

# Initialize parameters
w = np.random.randn()
b = np.random.randn()

# Learning rate and epochs
lr = 0.1
epochs = 50
loss_history = []

# Training loop
for epoch in range(epochs):
    # Predictions
    y_pred = w * X + b

    # Compute loss (MSE)
    loss = np.mean((y - y_pred) ** 2)
    loss_history.append(loss)

    # Gradients
    dw = -2 * np.mean((y - y_pred) * X)
    db = -2 * np.mean(y - y_pred)

    # Update parameters
    w -= lr * dw
    b -= lr * db

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

# Plot loss over time
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.title('Loss Decreasing Over Time')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
