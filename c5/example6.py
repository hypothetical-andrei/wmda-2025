# [An example visualizing output of each activation on sample data]
import numpy as np
import matplotlib.pyplot as plt

# Input values
x = np.linspace(-10, 10, 100)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()

# Compute outputs
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_softmax = softmax(x)  # softmax is usually over a vector, not element-wise

# Plot
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y_relu)
plt.title("ReLU")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, y_sigmoid)
plt.title("Sigmoid")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, y_tanh)
plt.title("Tanh")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, y_softmax)
plt.title("Softmax (on full vector)")
plt.grid(True)

plt.tight_layout()
plt.show()
