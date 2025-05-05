# Exercise 5 (10 minutes): Experimenting with Optimizers
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load and prepare the dataset
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)  # Binary labels: 0 = malignant, 1 = benign

# Scale and split
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# 2. Model definition
def build_model():
    return nn.Sequential(
        nn.Linear(30, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )

# 3. Training function
def train_model(optimizer_class, name, epochs=50):
    model = build_model()
    optimizer = optimizer_class(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    losses = []

    for _ in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

# 4. Compare optimizers
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop
}

results = {}
for name, opt_class in optimizers.items():
    print(f"Training with {name}")
    results[name] = train_model(opt_class, name)

# 5. Plot training loss
plt.figure(figsize=(8, 5))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.title("Optimizer Comparison on Breast Cancer Dataset")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()
