# app/predict.py

from app.predict import MalariaPredictor

predictor = MalariaPredictor(
    malaria_model_path="models/malaria_model.tflite",
    validator_model_path="models/validator_model.keras"
)
#   Test avec image
# 1-    non frottis
result = predictor.predict("test_images/C1_thinF_IMG_20150604_104722_cell_123.png")

# 2-  frottis
# result = predictor.predict("data/validator_dataset/not_blood_smear/img_0.jpg")

print(result)

