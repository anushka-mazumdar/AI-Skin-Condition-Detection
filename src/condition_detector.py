import os
import json
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Paths ===
MODEL_DIR = "models/skin_condition"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# === Load model and label map ===
print("🔍 Loading skin condition model...")
model = load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

IMG_SIZE = (224, 224)

# === Predict Function ===
def predict_condition(image_path):
    """
    Predicts the skin condition from an enhanced face image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read and preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    class_name = label_map[class_idx]

    return class_name, float(confidence)

# === Optional: Visualize Prediction ===
def visualize_prediction(image_path):
    condition, conf = predict_condition(image_path)
    img = cv2.imread(image_path)
    label = f"{condition} ({conf*100:.1f}%)"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === Test Run ===
if __name__ == "__main__":
    test_image = "data/dataset/testing/Eczemaa/eczema-chronic-14.jpg"  # example test path
    condition, confidence = predict_condition(test_image)
    print(f"🩺 Detected condition: {condition} ({confidence*100:.2f}% confidence)")
