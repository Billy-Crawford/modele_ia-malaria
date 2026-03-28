from PIL import Image
import os

def is_image_valid(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

def clean_folder(folder):
    removed = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)

            if not is_image_valid(path):
                print("Suppression :", path)
                os.remove(path)
                removed += 1

    print("Total supprimé :", removed)

clean_folder("data/")

