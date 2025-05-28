# Exercise 4: Image Segmentation with U-Net
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleUNet(nn.Module):
# TODO

model = SimpleUNet()
input_image = torch.rand(1, 1, 128, 128)
output_mask = model(input_image).squeeze().detach().numpy()

plt.subplot(1, 2, 1)
plt.imshow(input_image.squeeze().numpy(), cmap='gray')
plt.title("Input")
plt.subplot(1, 2, 2)
plt.imshow(output_mask, cmap='gray')
plt.title("Output Mask")
plt.show()
