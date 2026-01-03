import requests

url = "http://127.0.0.1:8000/predict"
# image_path = "../data/splits/test/Parasitized/C157P118ThinF_IMG_20151115_163915_cell_173.png"
image_path = "../data/splits/test/Uninfected/C217ThinF_IMG_20151106_141326_cell_13.png"

with open(image_path, "rb") as f:
    files = {"file": (image_path, f, "image/png")}
    response = requests.post(url, files=files)

print(response.json())