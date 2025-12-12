import os
import json
import torch
import numpy as np
import cv2
from torchvision import transforms, models
import torch.nn.functional as F

MODEL_DIR = "models/skin_condition"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cached model
_model = None
_label_map = None
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _lazy_load():
    global _model, _label_map
    if _model is not None and _label_map is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")

    data = torch.load(MODEL_PATH, map_location=DEVICE)
    # Rebuild a MobileNetV2 and load state dict
    num_classes = data.get("num_classes", None)
    if num_classes is None:
        raise ValueError("Saved checkpoint missing 'num_classes' field.")

    model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(data["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _model = model

    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            _label_map = json.load(f)
        # ensure int keys
        _label_map = {int(k): v for k, v in _label_map.items()}
    else:
        # fallback: build simple index->str map
        _label_map = {i: str(i) for i in range(num_classes)}


def predict_condition(image_path):
    """
    Returns (class_name, confidence) where confidence is in [0,1].
    """
    _lazy_load()

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image: " + image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = _transform(img)
    inp = inp.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = _label_map.get(idx, str(idx))
    return label, conf


def visualize_prediction(image_path):
    label, conf = predict_condition(image_path)
    img = cv2.imread(image_path)
    txt = f"{label} ({conf*100:.1f}%)"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_img = "data/dataset/val/Basal Cell Carcinoma/basal-cell-carcinoma-face-11.jpg"
    print("Testing with:", test_img)
    try:
        lab, c = predict_condition(test_img)
        print("Result:", lab, c)
    except Exception as e:
        print("Error:", e)
