from fastapi import FastAPI, File, UploadFile
from PIL import Image
from model import predict_cell

app = FastAPI(title="Malaria Detection API")

@app.post("/predict", response_model=None)
async def predict(file: UploadFile = File(...)):
    # Lire l'image uploadée
    image = Image.open(file.file)
    prediction = predict_cell(image)  # utilise le seuil défini dans model.py
    return prediction

@app.get("/")
def read_root():
    return {"message": "API Malaria Detection - Upload an image to /predict"}
