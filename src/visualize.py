import os
import json
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# === Paths ===
DATASET_DIR = "data/dataset"
MODEL_DIR = "models/skin_condition"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

# === Load model and label map ===
print("🔍 Loading model and metadata...")
model = load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}
inv_label_map = {v: k for k, v in label_map.items()}

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

# === Plot training curves ===
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history["accuracy"], label="train_acc")
plt.plot(history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history["loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# === Evaluate on validation set ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    subset="validation",
    shuffle=False
)

loss, acc = model.evaluate(val_data, verbose=1)
print(f"✅ Validation Accuracy: {acc:.4f}")
print(f"✅ Validation Loss: {loss:.4f}")

# === Show random sample predictions ===
x, y_true = next(val_data)
y_pred = model.predict(x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

plt.figure(figsize=(12,6))
for i in range(6):
    idx = random.randint(0, len(x)-1)
    img = x[idx]
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.axis('off')
    true_label = label_map[y_true_classes[idx]]
    pred_label = label_map[y_pred_classes[idx]]
    color = "green" if true_label == pred_label else "red"
    plt.title(f"T: {true_label}\nP: {pred_label}", color=color)
plt.tight_layout()
plt.show()
