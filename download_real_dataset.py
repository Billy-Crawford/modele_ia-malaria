# download_real_dataset.py

import os
import requests
from tqdm import tqdm

OUTPUT_DIR = "data/validator_dataset/not_blood_smear"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Génère plein d’images via picsum (images réalistes aléatoires)
TOTAL_IMAGES = 2000

for i in tqdm(range(TOTAL_IMAGES)):
    url = f"https://picsum.photos/200/200?random={i}"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(f"{OUTPUT_DIR}/img_{i}.jpg", "wb") as f:
                f.write(response.content)
    except:
        continue

