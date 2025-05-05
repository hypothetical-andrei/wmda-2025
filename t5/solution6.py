# Exercise 6 (15 minutes): Tuning Learning Rate
# dataset can be found here: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records or here https://www.kaggle.com/datasets/devvret/congressional-voting-records
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# 1. Load and clean the dataset
df = pd.read_csv("house-votes-84.csv")  # Make sure to download and load locally

# Map class labels (Democrat = 0, Republican = 1)
df['Class Name'] = df['Class Name'].map({'democrat': 0, 'republican': 1})

# Replace missing values with mode of each column
df = df.replace({'n': 0, 'y': 1, '?': pd.NA})
df = df.fillna(df.mode().iloc[0])

# Encode features and labels
X = df.drop(columns=["Class Name"]).astype(int).values
y = df["Class Name"].values.reshape(-1, 1)

# Scale and split
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# 2. Define model
def build_model():
    return nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

# 3. Train model for a given learning rate
def train_model(lr, epochs=50):
    model = build_model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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

# 4. Try different learning rates
learning_rates = [0.1, 0.01, 0.001]
results = {}

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    results[f"lr={lr}"] = train_model(lr)

# 5. Plot training loss
plt.figure(figsize=(8, 5))
for label, loss_curve in results.items():
    plt.plot(loss_curve, label=label)
plt.title("Learning Rate Comparison on Congressional Voting Records")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
