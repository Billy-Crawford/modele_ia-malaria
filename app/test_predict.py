from predict import MalariaPredictor

MODEL_PATH = "../models/malaria_model.tflite"
# IMAGE_PATH = "../data/splits/test/Parasitized/C39P4thinF_original_IMG_20150622_110115_cell_115.png"
IMAGE_PATH = "../data/splits/test/Uninfected/C225ThinF_IMG_20151112_113836_cell_290.png"

predictor = MalariaPredictor(MODEL_PATH)
label, confidence = predictor.predict(IMAGE_PATH)

print(f"Résultat : {label}")
print(f"Confiance : {confidence:.2f} %")
