# app/main.py
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from app.model import predict_cell

app = FastAPI(
    title="Malaria Detection API",
    version="2.0-medical",
    description="API médicale stable pour la détection du paludisme"
)


@app.get("/")
def read_root():
    return {
        "message": "Malaria Detection API v2 - Medical Stable",
        "threshold": 0.35,
        "model": "malaria_model_med_v2.tflite"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get the prediction:

    - "Cellule infectee" or "Cellule saine"
    - Confidence %
    """
    # Lire l'image uploadée
    image = Image.open(file.file)
    prediction = predict_cell(image)  # utilise le seuil défini dans model.py
    return prediction
