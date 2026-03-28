import os
import random
import shutil

# Source
PARASITIZED_DIR = "data/raw/cell_images/Parasitized"
UNINFECTED_DIR = "data/raw/cell_images/Uninfected"

# Destination
TRAIN_DIR = "data/validator_dataset/train/blood_smear"
VAL_DIR = "data/validator_dataset/val/blood_smear"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Charger toutes les images
images = []

for folder in [PARASITIZED_DIR, UNINFECTED_DIR]:
    for file in os.listdir(folder):
        images.append(os.path.join(folder, file))

# Mélange
random.shuffle(images)

# Split
split_index = int(0.8 * len(images))
train_images = images[:split_index]
val_images = images[split_index:]

# Copie
def copy_images(image_list, dest):
    for i, img_path in enumerate(image_list):
        new_name = f"img_{i}.jpg"
        shutil.copy(img_path, os.path.join(dest, new_name))

copy_images(train_images, TRAIN_DIR)
copy_images(val_images, VAL_DIR)

print("Dataset blood_smear prêt")

