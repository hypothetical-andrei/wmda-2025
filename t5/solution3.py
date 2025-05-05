# Exercise 3 (10 minutes): Exploring Activation Functions on the Iris Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load and prepare the Iris dataset (binary classification)
iris = load_iris()
X = iris['data']
y = (iris['target'] == 0).astype(float)  # 1 for Setosa, 0 otherwise

# Train/test split and normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

# 2. Build model factory with pluggable activation
def build_model(activation_fn):
    return nn.Sequential(
        nn.Linear(4, 8),
        activation_fn,
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

# 3. Training function
def train(model):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for _ in range(50):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

# 4. Try different activations
activations = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh()
}

results = {}
for name, act in activations.items():
    print(f"Training with {name} activation")
    model = build_model(act)
    results[name] = train(model)

# 5. Plot loss curves
plt.figure(figsize=(8, 5))
for name, loss_curve in results.items():
    plt.plot(loss_curve, label=name)
plt.title("Activation Function Comparison (Iris - Binary Classification)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
