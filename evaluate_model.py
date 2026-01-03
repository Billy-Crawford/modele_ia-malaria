# evaluate_model.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# === Chemins ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "malaria_model.tflite"
DATA_PATH = BASE_DIR / "data" / "splits" / "test"

# === Charger le modèle ===
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Fonction de prédiction ===
def predict_prob(image_path):
    img = Image.open(image_path).resize((64, 64)).convert("RGB")
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# === Charger les données ===
probs = []
y_true = []

for label, folder in enumerate(["Parasitized", "Uninfected"]):
    folder_path = DATA_PATH / folder
    for file in os.listdir(folder_path):
        p = folder_path / file
        prob = predict_prob(p)
        probs.append(prob)
        y_true.append(label)  # 0 = infectée, 1 = saine

probs = np.array(probs)
y_true = np.array(y_true)

# === Tester plusieurs seuils ===
thresholds = np.arange(0.1, 0.9, 0.05)

for t in thresholds:
    y_pred = np.where(probs >= t, 1, 0)

    print(f"\n==============================")
    print(f"SEUIL = {t:.2f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Infectee", "Saine"],
        digits=4
    ))
