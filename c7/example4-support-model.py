import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- Simple U-Net Model ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1)  # output mask
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

# --- Dataset Loader ---
class LungSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])
        
        image = Image.open(img_path).convert("L").resize((128, 128))
        mask = Image.open(mask_path).convert("L").resize((128, 128))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# --- Training Setup ---
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = LungSegmentationDataset("lung_dataset/images", "lung_dataset/masks", transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Train the Model ---
print("🚀 Training U-Net...")
model.train()
for epoch in range(3):  # small number of epochs for speed
    running_loss = 0.0
    for images, masks in tqdm(dataloader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# --- Save the Model ---
torch.save(model.state_dict(), "unet_lung_segmentation.pth")
print("✅ Saved model as unet_lung_segmentation.pth")
