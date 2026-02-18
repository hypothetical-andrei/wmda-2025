# Exercise 2 (15 minutes): Building an MLP Using PyTorch or TensorFlow
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Create dataset: 2D points, label = 1 if y > x else 0
torch.manual_seed(42)
X = torch.rand(200, 2)
y = (X[:, 1] > X[:, 0]).float().unsqueeze(1)  # shape (200, 1)

# 2. Define a simple MLP model

# 3. Loss and optimizer

# 4. Training loop


# 5. Evaluation

# 6. Plot loss over time
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()
