import os
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm

# Config
output_dir = 'lung_dataset'
image_dir = os.path.join(output_dir, 'images')
mask_dir = os.path.join(output_dir, 'masks')
num_samples = 20
img_size = (128, 128)

os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

def generate_image_and_mask(img_size):
    # Create a noisy grayscale background (simulating CT texture)
    base = np.random.normal(loc=100, scale=30, size=img_size).astype(np.uint8)

    # Create a synthetic "lesion" as a white circle on black mask
    mask = Image.new('L', img_size, 0)
    draw = ImageDraw.Draw(mask)

    # Random circle (diseased region)
    r = random.randint(10, 25)
    x = random.randint(r, img_size[0] - r)
    y = random.randint(r, img_size[1] - r)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    # Add brighter intensity in the lesion area on the image
    mask_np = np.array(mask) / 255.0
    lesion_intensity = np.random.randint(140, 180)
    lesion = (mask_np * lesion_intensity).astype(np.uint8)
    base = np.maximum(base, lesion)

    return Image.fromarray(base), mask

print("🧪 Generating synthetic lung CT images and masks...")
for i in tqdm(range(num_samples)):
    img, mask = generate_image_and_mask(img_size)
    img.save(os.path.join(image_dir, f"img{i}.png"))
    mask.save(os.path.join(mask_dir, f"img{i}.png"))

print("✅ Done. Data saved in 'lung_dataset/'")
