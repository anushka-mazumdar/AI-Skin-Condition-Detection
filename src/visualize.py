import os
import json
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# === Paths ===
DATASET_DIR = "data/dataset"
MODEL_DIR = "models/skin_condition"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# === Load label map ===
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}
inv_label_map = {v: k for k, v in label_map.items()}

# === Load training history ===
with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

# === Plot training curves ===
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# === Data transforms ===
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# === Load validation dataset ===
VAL_DIR = os.path.join(DATASET_DIR, "val")
val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# === Load model ===
print("🔍 Loading PyTorch model...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes = checkpoint["num_classes"]

model = models.mobilenet_v2(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(in_features, num_classes),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# === Evaluate model ===
correct = 0
total = 0
losses = []

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total
val_loss = sum(losses) / len(losses)

print(f"✅ Validation Accuracy: {val_acc:.4f}")
print(f"✅ Validation Loss: {val_loss:.4f}")

# === Show random sample predictions ===
print("🔍 Displaying random prediction samples...")

# Load 1 batch manually
images, labels = next(iter(val_loader))
images_cpu = images.cpu()

with torch.no_grad():
    outputs = model(images.to(DEVICE))
    probs = F.softmax(outputs, dim=1).cpu().numpy()
    pred_classes = np.argmax(probs, axis=1)

plt.figure(figsize=(12,6))
for i in range(6):
    idx = random.randint(0, len(images)-1)
    img = images_cpu[idx].numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # unnormalize
    img = np.clip(img, 0, 1)

    true_label = label_map[int(labels[idx])]
    pred_label = label_map[int(pred_classes[idx])]

    color = "green" if true_label == pred_label else "red"

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"T: {true_label}\nP: {pred_label}", color=color)

plt.tight_layout()
plt.show()
