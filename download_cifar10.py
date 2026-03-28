# import tensorflow_datasets as tfds
import os

from PIL import Image
import numpy as np
import tensorflow_datasets as tfds

# Dossier de sortie
OUTPUT_DIR = "data/validator_dataset/not_blood_smear"

# Créer dossier si inexistant
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger CIFAR-10
dataset = tfds.load("cifar10", split="train", as_supervised=True)

count = 0

for image, label in tfds.as_numpy(dataset):

    # Sauvegarder uniquement certaines classes (optionnel)
    # Ici on garde tout → parfait pour "not blood smear"

    img = Image.fromarray(image)
    img = img.resize((64, 64))  # important pour ton modèle

    img.save(f"{OUTPUT_DIR}/img_{count}.jpg")

    count += 1

    if count >= 5000:
        break  # limiter le dataset pour éviter explosion de stockage

print(f"{count} images téléchargées dans {OUTPUT_DIR}")

