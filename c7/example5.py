# [Compare real vs. generated faces and discuss how the discriminator learns]
import os
import zipfile
import random
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIG ---
zip_path = 'img_align_celeba.zip'
extract_dir = 'celeba_images'
num_samples = 4

# --- Step 1: Extract ZIP if not done already ---
if not os.path.exists(extract_dir):
    print("🔄 Extracting images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# --- Step 2: Load real images ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image_files = sorted(os.listdir(extract_dir + '/img_align_celeba/'))[:1000]  # Use a subset to avoid scanning 200k files
print("📁 Found", len(image_files), "image files")

sample_files = random.sample(image_files, num_samples)

real_images = []
for fname in sample_files:
    img = Image.open(os.path.join(extract_dir + '/img_align_celeba/', fname)).convert("RGB")
    img = transform(img)
    real_images.append(img)

real_batch = torch.stack(real_images)

# --- Step 3: Load pretrained DCGAN generator ---
print("🔍 Loading generator...")
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo', 'DCGAN', model_name='celeba', pretrained=True)
generator = model.netG
generator.eval()

# --- Step 4: Generate fake images ---
nz = 120 # latent dimension used by this model
latent_vectors = torch.randn(num_samples, nz, 1, 1)
with torch.no_grad():
    fake_batch = generator(latent_vectors)

# --- Step 5: Denormalize for viewing ---
def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

# --- Step 6: Plot ---
fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))

for i in range(num_samples):
    axes[0, i].imshow(denorm(real_batch[i]).permute(1, 2, 0))
    axes[0, i].axis('off')
    axes[0, i].set_title("Real")

    axes[1, i].imshow(denorm(fake_batch[i]).permute(1, 2, 0))
    axes[1, i].axis('off')
    axes[1, i].set_title("Fake")

plt.tight_layout()
plt.savefig("real_vs_fake_faces.jpg")
print("✅ Saved comparison as real_vs_fake_faces.jpg")
