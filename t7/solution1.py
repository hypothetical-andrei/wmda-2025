# Exercise 1: Visualizing CNN Filters
import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

img = cv2.imread(cv2.samples.findFile("lena.png"), cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

# Simple edge detection filter
# TODO

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray")
plt.title("Edge Detection")
plt.show()
