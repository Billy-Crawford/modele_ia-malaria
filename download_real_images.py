# download_real_images.py

import os
import requests
from tqdm import tqdm

# Dossier où les images seront stockées
OUTPUT_DIR = "data/validator_dataset/not_blood_smear"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Liste d'URLs d'images libres
urls = [
    "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d",
    "https://images.unsplash.com/photo-1498050108023-c5249f4df085",
    "https://images.unsplash.com/photo-1518770660439-4636190af475",
    "https://images.unsplash.com/photo-1534081333815-ae5019106622",
    "https://images.unsplash.com/photo-1519985176271-adb1088fa94c",
    "https://images.unsplash.com/photo-1522202176988-66273c2fd55f",
    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
]

for i, url in enumerate(urls):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        path = os.path.join(OUTPUT_DIR, f"img_{i}.jpg")
        with open(path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"[+] Image téléchargée: {path}")
    else:
        print(f"[-] Erreur téléchargement URL: {url}")


