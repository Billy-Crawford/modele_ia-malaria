# test_validator.py

from app.predict import MalariaPredictor

predictor = MalariaPredictor(
    malaria_model_path="models/malaria_model.tflite",
    validator_model_path="models/validator_model.keras"
)

# result = predictor.predict("test_images/C1_thinF_IMG_20150604_104722_cell_123.png")
result = predictor.predict("data/validator_dataset/not_blood_smear/img_0.jpg")

print(result)


