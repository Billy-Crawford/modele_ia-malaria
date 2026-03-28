import os
import random
import shutil

# Source
SRC_DIR = "data/validator_dataset/not_blood_smear"

# Destinations
TRAIN_DIR = "data/validator_dataset/train/not_blood_smear"
VAL_DIR = "data/validator_dataset/val/not_blood_smear"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Charger images
images = [os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR)]

# Mélange
random.shuffle(images)

# Split
split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# Copie
def copy_images(imgs, dest):
    for i, img_path in enumerate(imgs):
        shutil.copy(img_path, os.path.join(dest, f"img_{i}.jpg"))

copy_images(train_imgs, TRAIN_DIR)
copy_images(val_imgs, VAL_DIR)

print("not_blood_smear split terminé")

