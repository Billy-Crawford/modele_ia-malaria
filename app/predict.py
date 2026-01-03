import numpy as np
import tensorflow as tf
from PIL import Image


class MalariaPredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((64, 64))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            img_array
        )
        self.interpreter.invoke()

        prediction = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0][0]

        if prediction < 0.5:
            return "Cellule infectee", (1 - prediction) * 100
        else:
            return "Cellule saine", prediction * 100