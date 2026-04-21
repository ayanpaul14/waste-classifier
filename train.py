"""
train.py  —  Train MobileNetV3-Small on the Gary Thung Garbage Classification dataset
Run locally (not on Render) then upload waste_classifier.pth to your repo or cloud storage.

Dataset: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
        OR: https://github.com/garythung/trashnet
        Folder structure expected:
            dataset/
              cardboard/  (403 images)
              glass/      (501 images)
              metal/      (410 images)
              paper/      (594 images)
              plastic/    (482 images)
              trash/      (137 images)

Install deps first:
    pip install torch torchvision Pillow tqdm
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset"          # path to your downloaded dataset folder
MODEL_OUT     = "models/waste_classifier.pth"
BATCH_SIZE    = 32
EPOCHS        = 15
LR            = 1e-3
VAL_SPLIT     = 0.2
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
WASTE_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# ────────────────────────────────────────────────────────────────────────

print(f"Using device: {DEVICE}")

# ── Transforms ──────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Dataset ─────────────────────────────────────────────────────────────
full_dataset = ImageFolder(root=DATASET_DIR, transform=train_tf)
print(f"Total images: {len(full_dataset)}")
print(f"Classes found: {full_dataset.classes}")

# Warn if class order differs from WASTE_CLASSES
if full_dataset.classes != WASTE_CLASSES:
    print(f"WARNING: Dataset classes {full_dataset.classes} differ from expected {WASTE_CLASSES}")
    print("Update WASTE_CLASSES in app.py to match:", full_dataset.classes)

val_size  = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
val_ds.dataset = ImageFolder(root=DATASET_DIR, transform=val_tf)  # val uses val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── Model ────────────────────────────────────────────────────────────────
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(WASTE_CLASSES))
model = model.to(DEVICE)

# ── Loss + Optimizer ─────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# Phase 1: only train the new classifier head (frozen backbone)
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Training loop ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100. * correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
    return total_loss / total, 100. * correct / total


print("\n── Phase 1: Training classifier head (5 epochs) ──")
best_val_acc = 0
for epoch in range(1, 6):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc = val_epoch(model, val_loader, criterion)
    scheduler.step()
    print(f"Epoch {epoch:2d} | Train loss {tr_loss:.4f} acc {tr_acc:.1f}% | "
          f"Val loss {vl_loss:.4f} acc {vl_acc:.1f}% | {time.time()-t0:.1f}s")
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc

# Phase 2: Unfreeze backbone and fine-tune everything with lower LR
print("\n── Phase 2: Fine-tuning full model (10 more epochs) ──")
for param in model.features.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(6, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc = val_epoch(model, val_loader, criterion)
    scheduler.step()
    print(f"Epoch {epoch:2d} | Train loss {tr_loss:.4f} acc {tr_acc:.1f}% | "
          f"Val loss {vl_loss:.4f} acc {vl_acc:.1f}% | {time.time()-t0:.1f}s")
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  ✅ Saved best model (val acc {vl_acc:.1f}%)")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
print(f"Model saved to: {MODEL_OUT}")
print("\nNext steps:")
print("  1. Copy models/waste_classifier.pth to your project")
print("  2. Commit it to Git LFS  OR  upload to a CDN and download at startup")
print("  3. Re-deploy on Render")