# [An example comparing different optimizers on a classification task]
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create simple 2D dataset
X = torch.tensor([
    [0.2, 0.6], [0.5, 0.4], [0.8, 0.9], [0.3, 0.2],
    [0.1, 0.7], [0.6, 0.5], [0.9, 0.1], [0.4, 0.8]
], dtype=torch.float32)

y = torch.tensor([
    [1], [0], [1], [0], [1], [0], [0], [1]
], dtype=torch.float32)

# Simple binary classifier model
def create_model():
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

# Training loop
def train_model(optimizer_name, model, optimizer, epochs=50):
    criterion = nn.BCELoss()
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history

# Optimizer configs
optimizers = {
    "SGD": lambda model: optim.SGD(model.parameters(), lr=0.1),
    "Adam": lambda model: optim.Adam(model.parameters(), lr=0.01),
    "RMSprop": lambda model: optim.RMSprop(model.parameters(), lr=0.01),
}

# Train models and collect loss histories
loss_results = {}

for name, opt_fn in optimizers.items():
    model = create_model()
    optimizer = opt_fn(model)
    loss_history = train_model(name, model, optimizer)
    loss_results[name] = loss_history

# Plot results
plt.figure(figsize=(8, 6))
for name, loss_history in loss_results.items():
    plt.plot(loss_history, label=name)
plt.title("Loss Comparison of Optimizers")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()
