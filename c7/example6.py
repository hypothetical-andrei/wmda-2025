# [Apply style from a famous painting to a photograph of a city skyline]
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess images
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')

    # Resize if the image is too large
    size = max_size if max(image.size) > max_size else max(image.size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Load content and style images
content_path = 'city-skyline.jpg'  # Replace with your city image
style_path = 'starry-night.jpg'  # Replace with a famous painting

content_image = load_image(content_path)
style_image = load_image(style_path)

# Load pre-trained VGG-19 model
vgg = models.vgg19(pretrained=True).features.eval()

# Define layers for content and style extraction
content_layer_index = 21  # conv4_2
style_layer_indices = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, ..., conv5_1

# Extract features from the VGG model
def extract_features(image, model):
    features = {}
    x = image
    for i, layer in enumerate(model):
        x = layer(x)
        if i in style_layer_indices:
            features[f'style_{i}'] = x
        if i == content_layer_index:
            features['content'] = x
    return features

# Compute gram matrix for style representation
def gram_matrix(tensor):
    _, c, h, w = tensor.shape
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# Define loss functions
def compute_loss(target_features, content_features, style_features):
    content_loss = torch.nn.functional.mse_loss(target_features['content'], content_features['content'])

    style_loss = 0
    for i in style_layer_indices:
        target_gram = gram_matrix(target_features[f'style_{i}'])
        style_gram = gram_matrix(style_features[f'style_{i}'])
        style_loss += torch.nn.functional.mse_loss(target_gram, style_gram)

    return content_loss + 1e6 * style_loss

# Initialize target image (copy of content image)
target = content_image.clone().requires_grad_(True)

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Style Transfer Process
num_steps = 20
for step in range(num_steps):
    optimizer.zero_grad()

    target_features = extract_features(target, vgg)
    loss = compute_loss(target_features, extract_features(content_image, vgg), extract_features(style_image, vgg))

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}/{num_steps}, Loss: {loss.item()}")

# Convert tensor to image for display
def tensor_to_image(tensor):
    image = tensor.clone().detach()
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Denormalization
    image = image.clamp(0, 1)
    return image.numpy()

# Display final stylized image
plt.figure(figsize=(8, 8))
plt.imshow(tensor_to_image(target))
plt.axis("off")
plt.title("Stylized City Skyline")
plt.show()
