# [An example: PyTorch training loop on a simple dataset]
import torch
import torch.nn as nn
import torch.optim as optim

# Sample dataset: [x, y], label = 1 if y > x else 0
X = torch.tensor([
    [0.2, 0.6],
    [0.5, 0.4],
    [0.8, 0.9],
    [0.3, 0.2],
    [0.1, 0.7],
    [0.6, 0.5],
    [0.9, 0.1],
    [0.4, 0.8]
], dtype=torch.float32)

y = torch.tensor([
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [0],
    [1]
], dtype=torch.float32)

# Define a simple MLP
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")

# Test prediction
test_point = torch.tensor([[0.7, 0.9]])
prediction = model(test_point).item()
print(f"\nTest point {test_point.numpy().tolist()[0]} → Probability of y > x: {prediction:.2f}")
