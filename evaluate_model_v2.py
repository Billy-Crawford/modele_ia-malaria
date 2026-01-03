# evaluate_model_v2.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from analyze_errors import TEST_DIR, IMG_SIZE
from evaluate_model import thresholds

# ============================
# Configuration
# ============================
MODEL_PATH = "models/malaria_cnn_model_med_v2.keras"
TEST_DIR = "data/splits/test"
IMG_SIZE = (64, 64)

# ============================
# Chargement modèle
# ============================
model = tf.keras.models.load_model(MODEL_PATH)

# ============================
# Prédiction brute
# ============================
def predict_prob(image_path):
    img = Image.open(image_path).resize(IMG_SIZE).convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return float(model.predict(img, verbose=0)[0][0])

# ============================
# Collecte des probabilités
# ============================
probs = []
y_true = []

for folder in ["Parasitized", "Uninfected"]:
    label = 0 if folder == "Parasitized" else 1
    folder_path = os.path.join(TEST_DIR, folder)

    for file in os.listdir(folder_path):
        p = os.path.join(folder_path, file)
        prob = predict_prob(p)
        probs.append(prob)
        y_true.append(label)

probs = np.array(probs)
y_true = np.array(y_true)

# ============================
# Analyse par seuil
# ============================
thresholds = np.arange(0.05, 0.095, 0.05)

print("\n=== EVALUATION MEDICALE - MODELE V2 ===")

for t in thresholds:
    y_pred = (probs >= t).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Infectee", "Saine"],
        output_dict=True
    )

    recall_infected = report["Infectee"]["recall"]
    fn = cm[0][1]

    print(f"\nSEUIL = {t:.2f}")
    print(cm)
    print(f"Recall infectee : {recall_infected:.4f}")
    print(f"Faux negatifs   : {fn}")