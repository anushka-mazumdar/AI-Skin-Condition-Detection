import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# === Paths ===
DATASET_DIR = "data/dataset/train"
MODEL_DIR = "models/skin_condition"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Parameters ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

# === Data Generators ===
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

# === Save label map ===
label_map = {v: k for k, v in train_data.class_indices.items()}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=4)

# === Build Model (Transfer Learning with MobileNetV2) ===
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(len(label_map), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# === Callbacks ===
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.h5"),
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# === Train ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# === Save history ===
import pickle
with open(os.path.join(MODEL_DIR, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

print(f"✅ Model training complete. Saved in: {MODEL_DIR}")
