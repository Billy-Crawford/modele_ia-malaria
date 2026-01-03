# app/test_api_batch.py

import os
import requests

url = "http://127.0.0.1:8000/predict"
test_dirs = {
    "Parasitized": "../data/splits/test/Parasitized",
    "Uninfected": "../data/splits/test/Uninfected"
}

total, correct = 0, 0

for label, folder in test_dirs.items():
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files).json()
            total += 1
            if (response['result'] == "Cellule infectee" and label=="Parasitized") or \
               (response['result'] == "Cellule saine" and label=="Uninfected"):
                correct += 1

print(f"Exactitude approximative : {correct}/{total} = {correct/total*100:.2f}%")