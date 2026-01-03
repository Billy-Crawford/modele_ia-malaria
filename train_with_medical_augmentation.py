import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ============================
# Configuration générale
# ============================
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30

TRAIN_DIR = "data/splits/train"
VAL_DIR = "data/splits/val"

BASE_MODEL_PATH = "models/malaria_cnn_model.keras"
MODEL_OUT = "models/malaria_cnn_model_med_v2.keras"

# ============================
# Data Augmentation médicale
# ============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.15,
    brightness_range=[0.6, 1.1],
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ============================
# Pondération médicale
# ============================
# Classe 0 = Parasitized (priorité clinique)
# Classe 1 = Uninfected
class_weight = {
    0: 2.5,
    1: 1.0
}

# ============================
# Chargement du modèle de base
# ============================
if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(
        f"Modèle de base introuvable : {BASE_MODEL_PATH}"
    )

print("Chargement du modèle de base :", BASE_MODEL_PATH)
model = load_model(BASE_MODEL_PATH)

# ============================
# Callbacks professionnels
# ============================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_OUT,
        monitor="val_loss",
        save_best_only=True
    )
]

# ============================
# Entraînement médicalement orienté
# ============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

print("Modèle médicalement amélioré entraîné avec succès")
print("Sauvegardé sous :", MODEL_OUT)
