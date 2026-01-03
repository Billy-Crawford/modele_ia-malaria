# app/utils.py

from PIL import Image
from fastapi import HTTPException
from io import BytesIO

ALLOWED_FORMATS = ["PNG", "JPEG", "JPG"]

def load_and_validate_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(file_bytes))
        image.verify()  #verification interne PIL
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Image invalide ou corrompue. Veuillez reesayer.",
        )

    image = Image.open(BytesIO(file_bytes))

    if image.format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=415,
            details=f"Le format de l'image n'est pas prise en charge: {image.format}"
        )

    return image
