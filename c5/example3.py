# [An example: Manually implementing an MLP to classify handwritten digits]
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST (70000 samples)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.0  # Normalize
y = y.astype(int)

# One-hot encode targets
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.to_numpy().reshape(-1, 1))

# Use a subset for speed
X_train, _, y_train, _ = train_test_split(X, y_onehot, train_size=10000, random_state=42)

X_train = X_train.to_numpy()
y_train = np.asarray(y_train)

# Set sizes
input_size = 784
hidden_size = 64
output_size = 10
lr = 0.1
epochs = 10
batch_size = 64

# Initialize weights
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

# Loss function
def cross_entropy(predictions, targets):
    return -np.mean(np.sum(targets * np.log(predictions + 1e-8), axis=1))

# Training loop
for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = softmax(z2)

        # Loss (optional for debug)
        if i == 0:
            loss = cross_entropy(a2, y_batch)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # Backward pass
        dz2 = a2 - y_batch
        dW2 = a1.T @ dz2 / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = X_batch.T @ dz1 / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

# Evaluate on training data
z1 = X_train @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
preds = np.argmax(z2, axis=1)
true = np.argmax(y_train, axis=1)

acc = np.mean(preds == true)
print(f"Training accuracy: {acc * 100:.2f}%")
