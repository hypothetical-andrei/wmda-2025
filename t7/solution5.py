# Exercise 5: Style Transfer (Simplified Blending)
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import requests
from io import BytesIO

style_url = "https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

content = Image.open('eiffel-content.jpg').convert("RGB")
style = Image.open(BytesIO(requests.get(style_url).content)).convert("RGB")

# TODO

def show(img_tensor, title):
    img = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
show(content_tensor, "Content")
plt.subplot(1, 3, 2)
show(style_tensor, "Style")
plt.subplot(1, 3, 3)
show(blended, "Blended")
plt.tight_layout()
plt.show()
