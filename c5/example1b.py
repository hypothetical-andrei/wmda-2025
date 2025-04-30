# [An example showing equivalent code for defining a neural net in both TensorFlow and PyTorch]
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # input to hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)  # hidden to output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate model, loss, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Input and target tensors
X = torch.tensor([[0.1, 0.2], [0.4, 0.3], [0.6, 0.8], [0.9, 0.5]], dtype=torch.float32)
y = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
