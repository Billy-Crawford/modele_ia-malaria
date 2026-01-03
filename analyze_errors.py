# analyze_errors.py
import os
import numpy as np
from PIL import Image
import shutil
import tensorflow as tf

# ===============================
# Configuration
# ===============================
MODEL_PATH = "models/malaria_model.tflite"
TEST_DIR = "data/splits/test/Parasitized"
OUTPUT_DIR = "results/faux_negatifs"
IMG_SIZE = (64, 64)
THRESHOLD = 0.35  # seuil actuel

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Charger le modèle TFLite
# ===============================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===============================
# Fonction de prédiction
# ===============================
def predict(image_path):
    img = Image.open(image_path).resize(IMG_SIZE).convert("RGB")
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# ===============================
# Analyse
# ===============================
total = 0
false_negatives = 0

for filename in os.listdir(TEST_DIR):
    path = os.path.join(TEST_DIR, filename)
    prob = predict(path)

    total += 1

    # Faux négatif : infectée prédite saine
    if prob >= THRESHOLD:
        false_negatives += 1
        shutil.copy(path, os.path.join(OUTPUT_DIR, filename))

print("\n===== ANALYSE TERMINÉE =====")
print(f"Total cellules infectées testées : {total}")
print(f"Faux négatifs détectés          : {false_negatives}")
print(f"Taux de faux négatifs           : {false_negatives / total * 100:.2f}%")
print(f"Images sauvegardées dans        : {OUTPUT_DIR}")

