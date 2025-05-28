import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Load a pre-trained CartoonGAN generator
# Available models: 'shinkai', 'hayao', 'hosoda', 'paprika'
model = torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained='paprika')
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load an example image (URL or local path)
image = Image.open('portrait.jpg.jpg').convert("RGB")

# Prepare image
input_tensor = transform(image).unsqueeze(0)

# Generate stylized cartoon image
with torch.no_grad():
    output_tensor = model(input_tensor)[0]

# Post-process and display
output_image = (output_tensor * 0.5 + 0.5).clamp(0, 1)  # Unnormalize
output_pil = transforms.ToPILImage()(output_image)

# Save or show result
output_pil.save("cartoonized_output.png")
output_pil.show()
