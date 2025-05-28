# # [Show how to adapt a ResNet trained on ImageNet to classify medical images]
# curl -L -o ~/Downloads/covid19-radiography-database.zip\
#   https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet model
resnet = models.resnet18(pretrained=True)  # Load ResNet18 with ImageNet weights

# Modify the last fully connected layer for medical image classification
num_classes = 3  # This will be 3
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)  

# Move model to device
resnet = resnet.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Define transforms for medical images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load medical image dataset (replace 'data_path' with actual dataset path)
data_path = 'covid19_prepared'  # e.g., 'data/medical_images/'
train_dataset = datasets.ImageFolder(root=data_path + '/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training loop (simplified for demo purposes)
num_epochs = 2
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0

    count = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = resnet(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        count = count + 1
        if count % 10 == 0:
            print(count)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training Completed!")
