# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os

from app.predict import MalariaPredictor

app = FastAPI()

# =========================
# LOAD MODEL (AU DEMARRAGE)
# =========================
predictor = MalariaPredictor(
    malaria_model_path="models/malaria_model.tflite",
    validator_model_path="models/validator_model.keras"
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================
# ROUTE TEST
# =========================
@app.get("/")
def home():
    return {"message": "Malaria AI API is running"}


# =========================
# ROUTE PREDICT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 📥 Sauvegarde temporaire
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 🧠 Prediction
        result = predictor.predict(file_path)

        # 🧹 Nettoyage
        os.remove(file_path)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

