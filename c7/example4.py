# [Show how U-Net segments a lung CT scan into healthy vs. diseased regions]
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define a simple U-Net model (for demo purposes, usually use a pre-trained model)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


# Load a pre-trained U-Net model (for demo, using a simple model)
model = UNet()
model.load_state_dict(torch.load("unet_lung_segmentation.pth", map_location=torch.device("cpu")))
model.eval()

# Load and preprocess the lung CT scan image
image_path = "lung_ct_scan.png"  # Replace with your actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))  # Resize to match model input
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Normalize

# Perform segmentation
with torch.no_grad():
    output_mask = model(image_tensor)  # Forward pass through U-Net

# Convert output mask to NumPy array
mask = output_mask.squeeze().numpy()
mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask

# Overlay segmentation mask on the original image
overlay = cv2.applyColorMap(mask * 255, cv2.COLORMAP_JET)  # Color the mask
blended = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0)

# Display original and segmented images
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(image, cmap='gray'), plt.title("Original CT Scan")
plt.subplot(1,3,2), plt.imshow(mask, cmap='gray'), plt.title("Segmentation Mask")
plt.subplot(1,3,3), plt.imshow(blended), plt.title("Overlayed Mask")
plt.tight_layout()
plt.show()
