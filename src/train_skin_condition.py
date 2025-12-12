import os
import json
import time
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# Paths
DATASET_DIR = "data/dataset"        # expects train/ and val/ inside OR train used with validation_split
MODEL_DIR = "models/skin_condition"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparams
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4 if os.cpu_count() > 4 else 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 50

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def get_dataloaders(dataset_dir, batch_size=BATCH_SIZE):
    """
    Expects structure:
    data/dataset/train/<class>/*.jpg
    data/dataset/val/<class>/*.jpg
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        raise FileNotFoundError("Please ensure data/dataset/train and data/dataset/val exist and contain class folders.")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # label_map: index -> class name
    label_map = {v: k for k, v in train_ds.class_to_idx.items()}
    return train_loader, val_loader, label_map


def build_model(num_classes, feature_extract=True):
    # MobileNetV2 backbone from torchvision
    model = models.mobilenet_v2(pretrained=True)
    if feature_extract:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # usually 1280
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)


def train():
    train_loader, val_loader, label_map = get_dataloaders(DATASET_DIR)
    num_classes = len(label_map)
    print(f"Classes ({num_classes}):", label_map)

    model = build_model(num_classes, feature_extract=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val_acc = 0.0
    history = defaultdict(list)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        start = time.time()

        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)

            if step % PRINT_EVERY == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step {step}/{len(train_loader)} Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        elapsed = time.time() - start

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)

        val_loss = val_loss / val_total
        val_acc = val_corrects / val_total

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch} summary: Train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f} | time {elapsed:.1f}s")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "num_classes": num_classes
            }, save_path)
            print(f"Saved new best model to {save_path} (val_acc={val_acc:.4f})")

    # Save label map (index -> class) so inference can use it easily
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)

    # Save training history
    with open(os.path.join(MODEL_DIR, "training_history.pkl"), "wb") as f:
        pickle.dump(dict(history), f)

    print("Training complete. Best val acc:", best_val_acc)
    print("Model + metadata saved to:", MODEL_DIR)


if __name__ == "__main__":
    train()
