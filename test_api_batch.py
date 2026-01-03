import os
import requests
from pathlib import Path

# ==============================
# CONFIG
# ==============================
API_URL = "http://127.0.0.1:8000/predict"  # URL locale de l'API
IMAGES_DIR = Path("test_images")           # Dossier contenant les images à tester

# Vérification dossier
if not IMAGES_DIR.exists():
    print(f"Le dossier {IMAGES_DIR} n'existe pas !")
    exit(1)

# ==============================
# Fonction d'envoi d'une image à l'API
# ==============================
def send_image(file_path):
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "image/jpeg")}
        response = requests.post(API_URL, files=files)
        return response.json()


# ==============================
# Boucle sur toutes les images du dossier
# ==============================
results = []

for img_file in sorted(IMAGES_DIR.iterdir()):
    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        try:
            prediction = send_image(img_file)
            results.append({
                "image": img_file.name,
                "result": prediction["result"],
                "confidence": prediction["confidence"]
            })
            print(f"{img_file.name} → {prediction['result']} ({prediction['confidence']}%)")
        except Exception as e:
            print(f"Erreur pour {img_file.name} : {e}")

# ==============================
# Résumé final
# ==============================
print("\n===== RÉSUMÉ GLOBAL =====")
infectee = sum(1 for r in results if r["result"] == "Cellule infectee")
saine = sum(1 for r in results if r["result"] == "Cellule saine")
total = len(results)

print(f"Total images testées : {total}")
print(f"Cellules infectées : {infectee}")
print(f"Cellules saines    : {saine}")
