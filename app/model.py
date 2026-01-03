# app/model.py
import numpy as np
from PIL import Image
import tensorflow as tf

# ==============================
# CONFIGURATION MEDICALE OFFICIELLE
# ==============================
MODEL_PATH = "models/malaria_model_med_v2.tflite"
MEDICAL_THRESHOLD = 0.35  # seuil medical certifie v2
MODEL_VERSION = "v2-medical"

IMG_SIZE = (64, 64)

# ==============================
# CHARGEMENT DU MODELE TFLite
# ==============================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ==============================
# PREDICTION D'UNE CELLULE
# ==============================
def predict_cell(image: Image.Image, threshold: float = MEDICAL_THRESHOLD):
    """
    Prediction medicale d'une cellule sanguine.

    Regle medicale :
    - prediction > threshold  -> Cellule saine
    - prediction <= threshold -> Cellule infectee
    """

    # ---- Pretraitement image (STRICTEMENT identique au training) ----
    img = image.resize(IMG_SIZE).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # normalisation obligatoire
    img_array = np.expand_dims(img_array, axis=0)

    # ---- Inference TFLite ----
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0][0]

    # ---- Decision medicale ----
    if prediction > threshold:
        result = "Cellule saine"
        confidence = prediction
    else:
        result = "Cellule infectee"
        confidence = 1.0 - prediction

    # ---- Reponse medicale normalisee ----
    return {
        "result": result,
        "confidence": round(float(confidence * 100), 2),
        "threshold": threshold,
        "raw_score": round(float(prediction), 4),
        "model_version": MODEL_VERSION
    }
