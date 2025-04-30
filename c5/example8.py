# [An example showing training vs validation loss with and without dropout]
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Synthetic classification dataset: label = 1 if y > x else 0
np.random.seed(42)
X_data = np.random.rand(1000, 2)
y_data = (X_data[:, 1] > X_data[:, 0]).astype(np.float32).reshape(-1, 1)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Define model architecture
class MLP(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Training function
def train_model(use_dropout=False, epochs=50):
    model = MLP(use_dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
            val_losses.append(val_loss.item())

    return train_losses, val_losses

# Train both models
loss_no_dropout = train_model(use_dropout=False)
loss_with_dropout = train_model(use_dropout=True)

# Plot
epochs = range(1, 51)
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_no_dropout[0], label='Train (No Dropout)', linestyle='--')
plt.plot(epochs, loss_no_dropout[1], label='Val (No Dropout)', linestyle='--')
plt.plot(epochs, loss_with_dropout[0], label='Train (With Dropout)')
plt.plot(epochs, loss_with_dropout[1], label='Val (With Dropout)')
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training vs Validation Loss: With and Without Dropout")
plt.legend()
plt.grid(True)
plt.show()
