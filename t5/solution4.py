# Exercise 4 (15 minutes):  Preventing Overfitting with Dropout and Batch Normalization on the Iris Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load and prepare the Iris dataset (binary classification)
iris = load_iris()
X = iris['data']
y = (iris['target'] == 0).astype(float)  # 1 for Setosa, 0 otherwise

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

# 2. Model builder with optional dropout and batch norm

# 3. Training function
def train_model(model, epochs=50):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        pred_train = model(X_train)
        loss_train = criterion(pred_train, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            loss_val = criterion(pred_val, y_val)
            val_losses.append(loss_val.item())

    return train_losses, val_losses

# 4. Train models with different configs
configs = {
    "Baseline": build_model(),
    "With Dropout": build_model(use_dropout=True),
    "With BatchNorm": build_model(use_batchnorm=True)
}

results = {}
for label, model in configs.items():
    print(f"Training: {label}")
    results[label] = train_model(model)

# 5. Plot training vs validation loss
plt.figure(figsize=(10, 6))
for label, (train_l, val_l) in results.items():
    plt.plot(train_l, label=f"{label} - Train", linestyle='--')
    plt.plot(val_l, label=f"{label} - Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (Iris Dataset)")
plt.legend()
plt.grid(True)
plt.show()
