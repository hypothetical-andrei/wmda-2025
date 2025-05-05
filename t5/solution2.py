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
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# 3. Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
losses = []
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# 5. Evaluation
with torch.no_grad():
    preds = (model(X) > 0.5).float()
    acc = (preds == y).float().mean()
    print(f"\nFinal accuracy: {acc.item() * 100:.2f}%")

# 6. Plot loss over time
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()
