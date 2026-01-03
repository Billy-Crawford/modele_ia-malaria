import tensorflow as tf

MODEL_IN = "models/malaria_cnn_model_med_v2.keras"
MODEL_OUT = "models/malaria_model_med_v2.tflite"

model = tf.keras.models.load_model(MODEL_IN)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(MODEL_OUT, "wb") as f:
    f.write(tflite_model)

print("Modèle TFLite médical v2 exporté avec succès")
