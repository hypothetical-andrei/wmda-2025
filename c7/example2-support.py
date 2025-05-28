import zipfile
import os
import shutil
import random
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
zip_path = 'covid19-radiography-database.zip'
extract_dir = 'covid19_raw'
output_dir = 'covid19_prepared'
img_size = (224, 224)  # for ResNet

# --- Step 1: Extract ZIP ---
print("🔄 Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# --- Step 2: Locate the base dataset folder ---
print("🔍 Locating main dataset folder...")
base_data_dir = None
for root, dirs, files in os.walk(extract_dir):
    for name in dirs:
        if name.lower().startswith("covid-19_radiography_dataset"):
            base_data_dir = os.path.join(root, name)
            break
    if base_data_dir:
        break

if not base_data_dir:
    raise RuntimeError("❌ Could not find the 'COVID-19_Radiography_Dataset' folder inside the zip.")

print(f"✅ Found base data directory: {base_data_dir}")

# --- Step 3: Class mapping ---
# Adjust folder names if needed based on your dataset
class_map = {
    'COVID': 'COVID',
    'NORMAL': 'Normal',
    'VIRAL': 'Viral Pneumonia'
}

max_images_per_class = 10
train_ratio = 0.8

# --- Step 4: Create output dirs ---
for phase in ['train', 'val']:
    for cls in class_map:
        os.makedirs(os.path.join(output_dir, phase, cls), exist_ok=True)

# --- Step 5: Prepare limited samples ---
def prepare_class_images(class_label, folder_name):
    print(f"\n📁 Processing class: {class_label}")
    candidate_folder = os.path.join(base_data_dir, folder_name)
    full_folder = os.path.join(candidate_folder, 'images') if os.path.exists(os.path.join(candidate_folder, 'images')) else candidate_folder

    if not os.path.exists(full_folder):
        print(f"❌ Folder not found: {full_folder}")
        return

    images = [f for f in os.listdir(full_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"⚠️ No images found in: {full_folder}")
        return

    selected_images = random.sample(images, min(max_images_per_class, len(images)))
    random.shuffle(selected_images)
    split_idx = int(len(selected_images) * train_ratio)
    split_data = {
        'train': selected_images[:split_idx],
        'val': selected_images[split_idx:]
    }

    for phase, img_list in split_data.items():
        out_class_dir = os.path.join(output_dir, phase, class_label)
        for img_name in tqdm(img_list, desc=f"{phase.capitalize()} - {class_label}"):
            img_path = os.path.join(full_folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img.save(os.path.join(out_class_dir, img_name))
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")

# --- Step 6: Process all classes ---
for label, folder in class_map.items():
    prepare_class_images(label, folder)

print("\n✅ Demo dataset prepared at:", output_dir)
