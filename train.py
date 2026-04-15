import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'models/waste_classifier.pth'
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224

print(f"Training on: {DEVICE}")

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=os.path.join(DATASET_DIR, 'train'), transform=train_transforms)
val_dataset   = ImageFolder(root=os.path.join(DATASET_DIR, 'val'),   transform=val_transforms)
test_dataset  = ImageFolder(root=os.path.join(DATASET_DIR, 'test'),  transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

if 'organic' not in train_dataset.classes:
    print("WARNING: organic class not found! Check dataset/train/organic folder.")
else:
    print("Organic class found successfully.")

def build_model(num_classes):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

model = build_model(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0

def train_epoch(loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc='Train', leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

def eval_epoch(loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val', leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total

print("\n── Phase 1: Train head only (5 epochs) ──")
for epoch in range(5):
    tr_loss, tr_acc = train_epoch(train_loader)
    vl_loss, vl_acc = eval_epoch(val_loader)
    scheduler.step()
    print(f"Epoch {epoch+1:2d} | Train {tr_acc:.3f} | Val {vl_acc:.3f}")
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)

print("\n── Phase 2: Fine-tune full model (15 epochs) ──")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - 5)

for epoch in range(5, NUM_EPOCHS):
    tr_loss, tr_acc = train_epoch(train_loader)
    vl_loss, vl_acc = eval_epoch(val_loader)
    scheduler.step()
    print(f"Epoch {epoch+1:2d} | Train {tr_acc:.3f} | Val {vl_acc:.3f}")
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  Saved best model (val_acc={vl_acc:.4f})")

print(f"\n── Test Evaluation ──")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
test_loss, test_acc = eval_epoch(test_loader)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Best Val Acc  : {best_val_acc:.4f}")

with open('models/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)
print("Saved training_history.json")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'],   label='Val')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'],   label='Val')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()
plt.tight_layout()
plt.savefig('models/training_curve.png', dpi=150)
print("Saved training_curve.png")