# app/predict.py

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


class MalariaPredictor:
    def __init__(self, malaria_model_path, validator_model_path):
        """
        Initialisation des modèles :
        - malaria_model_path : TFLite pour prédiction parasitée/non
        - validator_model_path : Keras pour détecter si c'est un frottis
        """
        # 🔹 Modèle malaria (TFLite)
        self.malaria_interpreter = tf.lite.Interpreter(model_path=malaria_model_path)
        self.malaria_interpreter.allocate_tensors()
        self.malaria_input = self.malaria_interpreter.get_input_details()
        self.malaria_output = self.malaria_interpreter.get_output_details()

        # 🔹 Modèle validator (Keras)
        self.validator_model = tf.keras.models.load_model(validator_model_path)

        # ⚠️ Tailles d'image différentes pour les modèles
        self.VALIDATOR_SIZE = (128, 128)
        self.MALARIA_SIZE = (64, 64)

        # 🔹 Seuils stricts pour validator
        self.THRESHOLD_STRICT = 0.001  # < strict → OK
        self.THRESHOLD_WARNING = 0.01  # < warning → douteux
        self.THRESHOLD_ERROR = 0.02    # > error → clairement pas un frottis

    # =========================
    # 🔹 Prétraitement validator
    # =========================
    def preprocess_validator(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            return None
        img = img.resize(self.VALIDATOR_SIZE)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    # =========================
    # 🔹 Prétraitement malaria
    # =========================
    def preprocess_malaria(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.MALARIA_SIZE)
        img = np.array(img)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    # =========================
    # 🔹 Prédiction
    # =========================
    def predict(self, image_path):
        """
        Retourne un dict :
        - status : success / warning / error
        - type : 'blood_smear'
        - prediction : Parasitized / Uninfected (si success)
        - confidence : confiance modèle malaria
        - validator_confidence : confiance modèle validator
        """

        # -------- VALIDATOR --------
        validator_img = self.preprocess_validator(image_path)
        if validator_img is None:
            return {
                "status": "error",
                "message": "Image corrompue ou illisible"
            }

        validator_pred = float(self.validator_model.predict(validator_img)[0][0])
        print("Validator raw output:", validator_pred)

        # -------- Seuils stricts validator --------
        if validator_pred > self.THRESHOLD_ERROR:
            # Image clairement non médicale
            return {
                "status": "error",
                "message": "Image non valide (pas un frottis sanguin)",
                "validator_confidence": validator_pred
            }
        elif validator_pred > self.THRESHOLD_WARNING:
            # Image douteuse
            return {
                "status": "warning",
                "message": "Image douteuse, veuillez envoyer un vrai frottis",
                "validator_confidence": validator_pred
            }
        elif validator_pred > self.THRESHOLD_STRICT:
            # Très léger doute, mais OK
            return {
                "status": "warning",
                "message": "Image probablement valide mais vérifiez",
                "validator_confidence": validator_pred
            }

        # -------- MALARIA --------
        malaria_img = self.preprocess_malaria(image_path)
        self.malaria_interpreter.set_tensor(self.malaria_input[0]['index'], malaria_img)
        self.malaria_interpreter.invoke()
        malaria_pred = float(self.malaria_interpreter.get_tensor(self.malaria_output[0]['index'])[0][0])

        # Interprétation du score malaria
        if malaria_pred < 0.5:
            result = "Parasitized"
            confidence = 1 - malaria_pred
        else:
            result = "Uninfected"
            confidence = malaria_pred

        confidence = round(confidence, 4)

        return {
            "status": "success",
            "type": "blood_smear",
            "prediction": result,
            "confidence": confidence,
            "validator_confidence": validator_pred
        }

